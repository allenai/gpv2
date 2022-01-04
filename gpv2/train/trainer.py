import json
import logging
import math
import os
import re
import socket
from collections import defaultdict, Counter
from datetime import datetime
from os import mkdir, makedirs, getpid
from time import perf_counter

import h5py
import torch
from numbers import Number
from os.path import join, exists, dirname
from typing import List, Optional, Dict, Any, Union, Tuple

from allennlp.common import FromParams, Params, Registrable
from allennlp.common.util import lazy_groups_of
from dataclasses import dataclass, replace
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch import distributed as dist
import numpy as np

from gpv2.data.dataset import Dataset, Task
from gpv2.data.gpv_datasets import GpvDataset
from gpv2.data.webqa_dataset import WebQaDataset
from gpv2.model.model import PredictionArg, GPVExampleOutput, GPVModel, BEST_STATE_NAME, \
  BeamSearchSpec
from gpv2.train.evaluator import Evaluator, ResultKey, CaptionEvaluator
from gpv2.train.optimizer import OptimizerBuilder, TrainingScheduleBuilder
from gpv2.train.runner import DataLoaderBuilder, run_model, save_gpv_output, CollateWithBatch
from gpv2.train.samplers import SubsetSampler, DistributedSubsetSampler
from gpv2.train.stratified_subset_sampler import StratifiedSubsetSampler
from gpv2.utils import py_utils, pytorch_utils
from gpv2.utils.py_utils import select_run_dir, load_json_object, dump_json_object, duration_to_str
from gpv2.utils.to_params import to_params


@dataclass
class EvaluationSetup(FromParams):
  """Specifies how to evaluate a task"""

  @classmethod
  def from_params(
    cls,
    params: Params,
    constructor_to_call=None,
    constructor_to_inspect=None,
    **extras
  ):
    if "iterator" in params:
      assert params.pop("iterator") is None

    # Manually build the troublesome  `prediction_args` field becaues allennlp
    # does not handle the dictionary-of-union case
    if "beam_search" in params:
      bs = params.pop_bool("beam_search")
      prediction_args = dict(allennlp_spec=BeamSearchSpec(**bs))
    else:
      prediction_args = params.pop("prediction_args")
    params["prediction_args"] = None
    out = super().from_params(params, constructor_to_call, constructor_to_inspect)
    for k, v in prediction_args.items():
      if isinstance(v, (int, float, str) or v is None):
        pass
      else:
        prediction_args[k] = PredictionArg.from_params(v)
    out.prediction_args = prediction_args
    return out

  evaluator: Evaluator
  prediction_args: Dict[str, Union[PredictionArg, int, float, str, None]]


@dataclass
class TrainerDataset(FromParams):
  """Dataset with meta-data that will be used during training"""

  dataset: Dataset
  logging_name: str = None
  eval_sample: Optional[int] = None
  train_sample: Union[int, float, None] = None
  eval_setup: EvaluationSetup = None

  def get_name(self):
    if self.logging_name is None:
      return self.dataset.get_name()
    else:
      return self.logging_name


@dataclass
class RunArgs(FromParams):

  @classmethod
  def from_params(cls, params: Params, constructor_to_call=None, constructor_to_inspect=None, **other):
    if "send_model" in params:
      del params["send_model"]
    return super().from_params(params, constructor_to_call, constructor_to_inspect, **other)

  """Specifies what devices/distributed setting to train on"""
  devices: Union[str, int, List[int], None]
  seed: int
  dist_backend: str = "nccl"
  dist_url: str = 'tcp://localhost:10001'
  grad_accumulation: int = 1
  num_workers: Optional[int] = None

  @property
  def distributed(self):
    return isinstance(self.devices, list)

  @staticmethod
  def build(args: 'DeviceArgsType', force_one_process=False,
            grad_accumulation=1, num_workers=None, seed=None, dist_port=None, dist_backend="nccl"):
    if isinstance(args, RunArgs):
      return args
    if args is None:
      if torch.cuda.is_available():
        logging.info("cuda found, defaulting to cuda")
        args = 'cuda'
      else:
        logging.info("cuda not found, using cpu")
        args = 'cpu'

    elif isinstance(args, list) and len(args) == 1 and not force_one_process:
      args = args[0]
    if seed is None:
      seed = np.random.randint(0, 2**28)

    if dist_port is not None:
      dist_port = f'tcp://localhost:{dist_port}'
    elif "PYTORCH_DIST_PORT" in os.environ:
      dist_port = f'tcp://localhost:{os.environ["PYTORCH_DIST_PORT"]}'
    else:
      dist_port = f'tcp://localhost:10001'
    return RunArgs(args, grad_accumulation=grad_accumulation, dist_backend=dist_backend,
                   num_workers=num_workers, seed=seed, dist_url=dist_port)


# Arguements we can use to specify the device
DeviceArgsType = Union[RunArgs, int, str, List[int], None]


@dataclass
class _TrainingState:
  """Internal training state used for checkpointing"""
  global_step: int = 0
  epoch: int = 0
  best_save_score: Optional[float] = None
  loss_ema: float = 0.0
  monitor_ema: Optional[Dict] = None
  optimizer_state: Optional[Dict] = None
  scheduler_state: Optional[Dict] = None
  epoch_scheduler_state: Optional[Dict] = None
  model_state: Optional[Dict] = None


@dataclass
class _EvaluationRunner:
  """Internal class to run evaluations"""
  evaluator: Evaluator
  prediction_args: Dict[str, Any]
  examples: List
  data_loader: DataLoader
  dataset: Dataset
  eval_sample: int
  desc: str = "eval"
  nopbar: bool = False
  distributed_evaluator: bool = False

  def get_predictions(self, model) ->  Dict[str, GPVExampleOutput]:
    return run_model(
      model, self.data_loader, beams_to_keep=1,
      model_device=None, desc=self.desc, nopbar=self.nopbar,
      prediction_args=self.prediction_args
    )


def select_subdir(output_dir, target=None):
  prefix = "" if target is None else target + "-"
  i = 0
  while True:
    candidate = join(output_dir, prefix + "r" + str(i))
    if not exists(candidate):
      try:
        mkdir(candidate)
        return candidate
      except FileExistsError:
        pass
    i += 1


def is_distributed():
  return dist.is_initialized()


def is_primary():
  return not dist.is_initialized() or dist.get_rank() == 0


def get_lrs(optimizer):
  lrs = []
  for param_group in optimizer.param_groups:
    lrs.append(param_group['lr'])
  return lrs


def _train_worker(rank, *args, **kwargs):
  return Trainer._train_worker(*args, **kwargs, rank=rank)


def _remove_tensors(monitor: Dict) -> Dict:
  for k, v in monitor.items():
    if isinstance(v, torch.Tensor):
      monitor[k] = v.item()
  return monitor


@dataclass
class Trainer(FromParams):
  """Class to run the training loop for our models"""

  train_datasets: List[TrainerDataset]
  eval_datasets: List[TrainerDataset]
  optimizer: OptimizerBuilder
  epochs: int
  train_loader: DataLoaderBuilder

  # Additional optimization settings
  step_schedule: TrainingScheduleBuilder = None
  clip_grad_norm: Optional[float] = None

  # Data iterator parameters
  find_unused_parameters: bool = True
  end_at_epoch: Optional[int] = None
  eval_loader: DataLoaderBuilder = None

  # Should we balance the different train dataset between batches
  stratify: bool = True

  # Saving the results
  save_evaluation_results: bool = True
  save_prediction_samples: Optional[int] = 0
  save_each_epoch: int = True
  best_model_key: Union[ResultKey, None, List[Union[ResultKey, Tuple[ResultKey, float]]]] = None
  eval_at_start: bool = False
  checkpoint: bool = False
  train_val_log: Optional[List[Tuple[ResultKey, str]]] = None

  # Cosmetic/Logging
  tb_log_intervals: int = 20
  tb_log: bool = True
  log_lr: bool = True
  log_frozen_parameters = True
  loss_logging_ema: float = 0.99
  monitor_ema: float = 0.99
  eval_pbar: bool = True
  epoch_pbar: bool = True
  eval_at: int = None

  # Should we log to tensorboard the stats from just hte main process, or
  # sync the results between all processes
  sync_monitor: bool = True

  def train(self, model: GPVModel, output_dir: Optional[str],
            device: DeviceArgsType = None, override=False):
    """Train a model

    :param model: Model to train
    :param output_dir: directory to save reults
    :param device: GPU devices to train on
    :param override: Override `output_dir` if it exists
    """
    if output_dir is not None:
      logging.info(f"Initializing model dir {output_dir}")
      py_utils.clear_if_nonempty(output_dir, override)
      makedirs(output_dir, exist_ok=True)
      Params(to_params(self, None)).to_file(join(output_dir, "trainer.json"))
      Params(to_params(model, GPVModel)).to_file(join(output_dir, "model.json"))
    else:
      logging.info(f"No output dir, model will not be saved")
    device = RunArgs.build(device)
    self._train(model, output_dir, device)

  @staticmethod
  def resume_from_checkpoint(run_dir: str, device: DeviceArgsType = None, save=True):
    """Continue training a model from checkpoint

    :param run_dir: Run directory to resume
    :param device: Devices to run on
    :param save: Continue saving the results in `run_dir`
    """
    logging.info(f"Resuming training for {run_dir}")

    run_dir = select_run_dir(run_dir)

    status = load_json_object(join(run_dir, "status.json"))
    if status["done"]:
      logging.info(f"{run_dir} is already marked as done")
      return

    logging.info("Loading trainer")
    output_dir = dirname(run_dir)
    with py_utils.DisableLogging():
      trainer = Trainer.from_params(Params.from_file(join(output_dir, "trainer.json")))
    model_file = join(output_dir, "model.json")

    checkpoint_file = join(run_dir, "checkpoint.pth")
    run_args = RunArgs.build(device)
    if not save:
      logging.info("Save is false, so no results will be recorded")
      run_dir, output_dir = None, None
    trainer._train(model_file, output_dir, run_args, checkpoint_file, run_dir)

  @staticmethod
  def train_another_model(output_dir: str, device: DeviceArgsType = None, save=True):
    """Train another run of the model stored in `output_dir`

    :param output_dir:
    :param device: Devices to train on
    :param save: Save the new run in `output_dir`, otherwise do not save anything
    """
    logging.info(f"Starting another run for {output_dir}")
    logging.info("Getting trainer/model")
    with py_utils.DisableLogging():
      trainer = Trainer.from_params(Params.from_file(join(output_dir, "trainer.json")))
      model = GPVModel.from_params(Params.from_file(join(output_dir, "model.json")))

    run_args = RunArgs.build(device)
    if not save:
      logging.info("Save is false, so no results will be recorded")
      output_dir = None
    trainer._train(model, output_dir, run_args)

  def _init_eval(
      self,
      model: GPVModel,
      train_examples: List[List],
      eval_examples: List[List]
  ) -> Dict[str, _EvaluationRunner]:
    """Build the `_EvaluationRunner` we will use for eval"""

    runners = {}
    collate_fn = CollateWithBatch(model.get_collate())

    to_eval = list(zip(eval_examples, self.eval_datasets))
    to_eval += list(zip(train_examples, self.train_datasets))

    builder = self.eval_loader
    if builder is None:
      builder = self.train_loader

    batch_size = builder.batch_size

    if is_distributed():
      assert batch_size % dist.get_world_size() == 0
      batch_size = batch_size // dist.get_world_size()

    total_eval = 0
    for examples, ds in to_eval:
      if ds.eval_sample == 0:
        continue
      # Slightly more efficient to group by query length
      prepped = [model.preprocess_example(x) for x in examples]

      eval_spec = ds.eval_setup

      if is_distributed():
        sampler = DistributedSubsetSampler(
          len(prepped), ds.eval_sample, dist.get_rank(), dist.get_world_size())
      else:
        if ds.eval_sample:
          sampler = SubsetSampler(len(prepped), ds.eval_sample)
        else:
          sampler = None

      loader = builder.build(
        prepped, collate_fn, shuffle=False, sampler=sampler, batch_size=batch_size)
      total_eval += ds.eval_sample if ds.eval_sample else len(examples)

      pbar = is_primary() and self.eval_pbar
      runner = _EvaluationRunner(
        eval_spec.evaluator, eval_spec.prediction_args, examples,
        eval_sample=ds.eval_sample, data_loader=loader, dataset=ds.dataset,
        desc=f"{ds.get_name()}", nopbar=not pbar,
        # TODO this is a hack
        distributed_evaluator=not isinstance(eval_spec.evaluator, CaptionEvaluator)
      )
      if ds.get_name() in runners:
        raise ValueError("Datasets have identical logging names")
      runners[ds.get_name()] = runner

    return runners

  def _get_optimizers(self, model: GPVModel, epoch_size: int, train_state: _TrainingState):
    """Construct the optimizer and step_sheduler"""

    optimizer = self.optimizer.build(model, epoch_size, self.epochs)
    if train_state.optimizer_state:
      optimizer.load_state_dict(train_state.optimizer_state)
      train_state.optimizer_state = None

    step_scheduler = None
    if self.step_schedule is not None:
      num_steps = epoch_size * self.epochs
      step_scheduler = self.step_schedule.build(optimizer, num_steps, train_state.global_step - 1)
      if train_state.scheduler_state is not None:
        step_scheduler.load_state_dict(train_state.scheduler_state)

    return optimizer, step_scheduler

  def _get_train_loader(self, model: GPVModel, training_examples: List[List],
                        runtime: RunArgs):
    """Returns the train batch iterator, size of the iterator, and the sampler if one is used"""

    all_train = []
    for grp, ds in zip(training_examples, self.train_datasets):
      preprocessed = []
      for source_ex in grp:
        for ex in model.preprocess_example_train(source_ex):
          if ex.meta is None:
            ex.meta = {}
          # Used for logging the losses to tensorboard
          # We manually map datasets to more compact keys here
          if isinstance(ds, GpvDataset) and ds.split == "train":
            ex.meta["loss-logging"] = str(ds.task) + "-loss"
          elif isinstance(ds, WebQaDataset) and ds.split == "train":
            ex.meta["loss-logging"] = "webqa-loss"
          else:
            ex.meta["loss-logging"] = ds.get_name() +"-loss"
          preprocessed.append(ex)
      all_train.append(preprocessed)
    all_train_sizes = [len(x) for x in all_train]
    all_train = py_utils.flatten_list(all_train)

    if len(set(x.id for x in all_train)) != len(all_train):
      raise ValueError("Repeated IDs in train")

    shuffle = True

    batch_size = self.train_loader.batch_size

    if (any(x.train_sample is not None for x in self.train_datasets) or
        self.stratify or is_distributed()):
      # Use our custom sampler that handles all these cases
      if is_distributed():
        world_size, rank = dist.get_world_size(), dist.get_rank()
        if batch_size % world_size != 0:
          raise ValueError("Batch size not divisible by world size")
        batch_size = batch_size // world_size
        logging.info(f"Using batch size {batch_size} since there "
                     f"are {world_size} workers with base size of {self.train_loader.batch_size}")
      else:
        world_size, rank = None, None

      samples = [x.train_sample for x in self.train_datasets]
      sampler = StratifiedSubsetSampler(
        all_train_sizes, runtime.seed, self.stratify, samples, batch_size, rank, world_size)
      shuffle = False   # Sampler does shuffling
      loader_batch_size = 1  # Sampler does batching
    else:
      loader_batch_size = batch_size
      sampler = None

    batch_groups = runtime.grad_accumulation
    if batch_groups > 1:
      if batch_size % batch_groups != 0:
        raise NotImplementedError("Batch size not divisible by grad accumulation steps")
      prev_batch_size = batch_size
      batch_size = batch_size // batch_groups
      logging.info(f"Accumulating total of {prev_batch_size} through {batch_groups} size {batch_size} batches")

    loader = self.train_loader.build(
      all_train, model.get_collate(True),
      batch_size=loader_batch_size, shuffle=shuffle, batch_sampler=sampler)

    if batch_groups == 1:
      return loader, len(loader), sampler
    else:
      batch_group_generator = lazy_groups_of(loader, batch_groups)
      num_training_batches = math.ceil(len(loader) / batch_groups)
      return batch_group_generator, num_training_batches, sampler

  def _get_train_eval_dir(self, run_dir, epochs, step) -> Optional[str]:
    """Where to save predictions we recorded during evaluation"""

    if run_dir is None or (not self.save_evaluation_results and self.save_prediction_samples == 0):
      return None
    if not exists(join(run_dir, "train-evals")):
      mkdir(join(run_dir, "train-evals"))
    out = join(run_dir, "train-evals", f"ep{epochs}-st{step}")
    if not exists(out):
      mkdir(out)
    return out

  def _get_task_eval_dir(self, eval_dir, eval_name: str):
    task_dir = join(eval_dir, eval_name)
    if not exists(task_dir):
      mkdir(task_dir)

    return task_dir

  def _run_eval(self, model, runners: Dict[str, _EvaluationRunner],
                global_step: int, seed: int, eval_dir) -> Dict[ResultKey, Number]:
    if is_distributed():
      all_results = self._run_eval_dist(model, runners, global_step, seed, eval_dir)
    else:
      all_results = {}
      for name, eval in runners.items():
        outputs = eval.get_predictions(model)
        results = eval.evaluator.evaluate(eval.examples, outputs, allow_partial=True)
        if self.save_prediction_samples != 0 and eval_dir is not None:
          to_save = py_utils.sample_dict(outputs, self.save_prediction_samples)
          save_gpv_output(to_save, self._get_task_eval_dir(eval_dir, name))

        for k, v in results.items():
          all_results[replace(k, dataset_name=name)] = v

    return all_results

  def _run_eval_dist(self, model, runners: Dict[str, _EvaluationRunner],
                     global_step, seed, eval_dir) -> Dict[ResultKey, Number]:
    # Need to give any samplers an identical seed that is different from seeds
    # used in previous evaluations
    for runner in runners.values():
      if runner.data_loader.sampler is not None:
        runner.data_loader.sampler.set_seed(global_step*78 + seed*13)

    n_to_save = self.save_prediction_samples

    all_results = {}
    predictions = {}
    for name, eval in runners.items():
      pred = eval.get_predictions(model)
      if eval_dir and n_to_save != 0:
        # Primary saves its predictions
        out = self._get_task_eval_dir(eval_dir, name)
        to_save = py_utils.sample_dict(pred, self.save_prediction_samples)
        save_gpv_output(to_save, out)

      if not eval.distributed_evaluator:
        # Each process sends over all its predictions
        predictions[name] = pred
      else:
        # Each process runs its own evaluation and just sends the results
        results = eval.evaluator.evaluate(
              eval.examples, pred, allow_partial=True, mean=False)
        for k, v in results.items():
          all_results[replace(k, dataset_name=name)] = v

    # Workers run the evaluations completely independently, now gather the results on the primary
    logging.info(f"Aggregating distributed results")

    if len(all_results) > 0:
      # Gather scores where the each process ran its own evaluator
      # We don't use `gather_object` since it is not supported on NCCL backend
      output = [None for _ in range(dist.get_world_size())]
      dist.all_gather_object(output, all_results)

      if is_primary():
        # Aggregate distributed results, which were reported in (total, count) form
        all_results = py_utils.transpose_list_of_dicts(output)
        for k, v in all_results.items():
          sums, ns = py_utils.transpose_lists(v)
          all_results[k] = sum(sums) / sum(ns)

    if len(predictions) > 0:
      # Gather scores where each process produced predictions, and the primary
      # runs the evaluator
      output = [None for _ in range(dist.get_world_size())]
      dist.all_gather_object(output, predictions)
      if is_primary():
        for evaluation_name in output[0]:
          all_pred = {}
          for pred in output:
            assert not any(x in all_pred for x in pred)
            all_pred.update(pred[evaluation_name])

          runner = runners[evaluation_name]
          results = runner.evaluator.evaluate(runner.examples, all_pred, True)
          for k, v in results.items():
            all_results[replace(k, dataset_name=evaluation_name)] = v

    if is_primary():
      return all_results
    else:
      return None

  def _log_results(
      self, result: Dict[ResultKey, Number], summary_writer: SummaryWriter, global_step,
      eval_time, eval_dir):
    if not is_primary():
      return

    if self.save_evaluation_results and eval_dir is not None:
      to_save = {str(k): v for k, v in result.items()}
      to_save["step"] = global_step
      with open(join(eval_dir, "eval.json"), "w") as fh:
        json.dump(to_save, fh, indent=2)

    grouped = defaultdict(lambda: defaultdict(dict))
    name_ds = {x.logging_name: x for x in (self.train_datasets + self.eval_datasets)}
    train_names = set(k.logging_name for k in self.train_datasets)
    for k, r in result.items():
      src = name_ds[k.dataset_name].dataset.get_source_name()
      is_train = k.dataset_name in train_names
      if k.subset_name is None:
        key = k.metric_name
      else:
        key = k.subset_name + "-" + k.metric_name
      grouped[src]["tr" if is_train else "val"][key] = r

    result_str = f"Evaluation took {duration_to_str(eval_time)}, showing results"
    for name, table in grouped.items():
      result_str += "\n"
      result_str += py_utils.dict_of_dicts_as_table_str(table, "%.2f", top_right=name)
    logging.info(result_str)

    if summary_writer:
      for k, r in result.items():
        if k.subset_name is None:
          key = k.metric_name
        else:
          key = k.subset_name + "-" + k.metric_name
        key = k.dataset_name + "/" + key
        summary_writer.add_scalar(key, r, global_step)

        if self.train_val_log:
          train_val_log = {k: v for k, v in self.train_val_log}
          if k not in train_val_log:
            continue
          summary_writer.add_scalar(train_val_log[k], r, global_step)

  def _init_run_dir(self, output_dir, run_args: RunArgs):
    """Initialize the subdirectory for this particular run"""

    run_dir = select_subdir(output_dir)
    logging.info("Saving run to %s" % run_dir)

    with open(join(run_dir, "runtime.json"), "w") as f:
      json.dump(dict(
        hostname=socket.gethostname(),
        date=datetime.now().strftime("%m%d-%H%M%S"),
        device=to_params(run_args, RunArgs)
      ), f, indent=2)
    dump_json_object(dict(done=False), join(run_dir, "status.json"))
    return run_dir

  def _log_continue(self, run_dir, run_args: RunArgs, train_state: _TrainingState):
    """Log a continue-from-checkpoint note to the runtime file"""

    runtime_f = join(run_dir, "runtime.json")
    prev = load_json_object(runtime_f)
    if "continue" not in prev:
      prev["continue"] = []
    prev["continue"].append(dict(
        global_step=train_state.global_step,
        epoch=train_state.epoch,
        hostname=socket.gethostname(),
        date=datetime.now().strftime("%m%d-%H%M%S"),
        device=to_params(run_args, RunArgs)
    ))
    with open(runtime_f, "w") as f:
      json.dump(prev, f, indent=2)

  def _get_model_score(self, results: Dict[ResultKey, Number]):
    """Get the overall model selection score from the computed results"""

    if isinstance(self.best_model_key, ResultKey):
      return results[self.best_model_key]

    total = 0
    for k in self.best_model_key:
      if isinstance(k, tuple):
        k, w = k
      else:
        w = 1.0
      total += results[k]*w
    return total

  def _load_and_log_train(self):
    """Load the training and log dataset sizes"""
    training_examples = [x.dataset.load() for x in self.train_datasets]
    total = sum(len(x) for x in training_examples)

    logging.info(f"Have {total} train examples")
    for ex, ds in zip(training_examples, self.train_datasets):
      logging.info(f"\t{len(ex)} from {ds.get_name()}")
    return training_examples

  def _load_and_log_eval(self, training_examples):
    """Load eval data and log the dataset sizes"""
    eval_examples = [x.dataset.load() for x in self.eval_datasets]

    total_eval = 0
    all_eval = (list(zip(eval_examples, self.eval_datasets)) +
                list(zip(training_examples, self.train_datasets)))
    for (examples, ds) in all_eval:
      if ds.eval_sample:
        total_eval += ds.eval_sample
      else:
        total_eval += len(examples)

    logging.info(f"Have {total_eval} eval examples")
    for (examples, ds) in all_eval:
      if ds.eval_sample == 0:
        pass
      elif ds.eval_sample is not None:
        logging.info(f"\t{ds.eval_sample} samples of {len(examples)} from {ds.get_name()}")
      else:
        logging.info(f"\t{len(examples)} from {ds.get_name()}")
    return eval_examples

  def _train(self, model: Union[str, GPVModel], output_dir, runtime: RunArgs,
             train_state_file: Optional[str]=None, run_dir=None):
    """Train with the output dir already initialized, and possibly a checkpoint file"""

    if train_state_file is not None:
      assert isinstance(model, str)
    else:
      assert isinstance(model, GPVModel)

    if not runtime.distributed:
      # Load train since we can pass it directly to the worker method
      logging.info("Loading training data")
      training_examples = self._load_and_log_train()
    else:
      # Only load train if we need to initialize the moel
      training_examples = None

    if output_dir is not None:
      if run_dir is None:
        logging.info("Initializing run dir")
        run_dir = self._init_run_dir(output_dir, runtime)
      log_file = join(run_dir, "out.log")
      record_log_handle = logging.FileHandler(log_file)
      logging.getLogger().addHandler(record_log_handle)
    else:
      run_dir = None
      record_log_handle = None

    devices = runtime.devices
    if isinstance(devices, list):
      world_size = len(devices)

      logging.info(f"Given {len(devices)} devices, launching {len(devices)} worker processes")
      if isinstance(model, str):
        to_send = model
      else:
        to_send = to_params(model, GPVModel)
      args = (
        self, to_send, None,
        run_dir, runtime, train_state_file, world_size
      )
      context = torch.multiprocessing.spawn(_train_worker, nprocs=world_size, args=args, join=False)

      del training_examples

      while not context.join():
        pass
    else:
      self._train_worker(model, training_examples, run_dir, runtime, train_state_file)

    if run_dir is not None:
      dump_json_object(dict(done=True), join(run_dir, "status.json"))

    if record_log_handle is not None:
      logging.getLogger().removeHandler(record_log_handle)

  def _train_worker(self, model, training_examples,
                    run_dir, runtime: RunArgs,
                    train_state_file: Optional[str], world_size=None, rank=0):
    """The main train loop that can be used in a distributed setting,"""
    # Set the device, and do some setup if we are distributed
    if world_size is not None:

      # Need to reset the logging
      py_utils.add_stdout_logger()

      if rank == 0 and run_dir:
        log_file = join(run_dir, "primary.log")
        record_log_handle = logging.FileHandler(log_file)
        logging.getLogger().addHandler(record_log_handle)

      device = runtime.devices[rank]
      if rank == 0:
        suffix = " (will log to stdout)"
      else:
        suffix = ""
      logging.info(f"Worker {rank} proc={getpid()} starting for device {device}{suffix}")

      if not isinstance(model, GPVModel):
        # Make sure everything we need to build the model with from_params is imported
        py_utils.import_all()

      if rank > 0:
        logging.disable(logging.WARNING)  # Non-primary only logs critical messages
        run_dir = None  # Only the primary saves anything

      dist.init_process_group(
        backend=runtime.dist_backend,
        init_method=runtime.dist_url,
        world_size=world_size,
        rank=rank)
      torch.cuda.set_device(device)
    else:
      device = runtime.devices

    # Now get the train state and model as objects
    if train_state_file is not None:
      # resuming training
      logging.info("Loading train state")
      train_state: _TrainingState = torch.load(train_state_file, map_location="cpu")

      # model is passed in as a file in this case
      logging.info("Loading model")
      with py_utils.DisableLogging():
        model = GPVModel.from_params(Params.from_file(model))
      model.load_state_dict(train_state.model_state)
      train_state.model_state = None

      if run_dir is not None:
        self._log_continue(run_dir, runtime, train_state)
    else:
      train_state = _TrainingState()

      # Get the model
      if isinstance(model, dict):
        logging.info("Loading model")
        # Models was sent as parameters
        with py_utils.DisableLogging():
          model = GPVModel.from_params(Params(model))
      else:
        # Should have been sent the full model
        assert isinstance(model, GPVModel)

      if is_primary():
        logging.info("Initializing model")
        model.initialize()
      else:
        model.initialize(False)

    if train_state.epoch != 0:
      assert train_state.global_step != 0

    device = torch.device(device)

    # Finish setting up the model
    model.to(device)

    _base_model = model
    if world_size is not None:
      model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], find_unused_parameters=self.find_unused_parameters)

    logging.info("Preparing training loader")
    if training_examples is None:
      training_examples = self._load_and_log_train()

    train_loader, n_train_batches, tr_sampler = self._get_train_loader(_base_model, training_examples, runtime)
    logging.info("Preparing evaluation")
    eval_examples = self._load_and_log_eval(training_examples)
    eval_runners = self._init_eval(_base_model, training_examples, eval_examples)

    logging.info("Preparing optimizers")
    optimizer, step_scheduler = self._get_optimizers(
      _base_model, n_train_batches, train_state)
    if self.clip_grad_norm is not None:
      clip_params = [p for n, p in _base_model.named_parameters()]
      logging.info(f"Clipping grad norm for {len(clip_params)} parameters")
    else:
      clip_params = list(model.parameters())

    # Other stuff we need to track during training
    if run_dir and self.tb_log:
      summary_writer = SummaryWriter(join(run_dir, "log"))
    else:
      summary_writer = None
    best_saved_score = train_state.best_save_score
    monitor_ema = train_state.monitor_ema
    global_step = train_state.global_step
    if monitor_ema is None:
      monitor_ema = {}
    loss_ema = train_state.loss_ema

    # Do initial eval if asked
    if self.eval_at_start and global_step == 0:
      logging.info("Starting initial eval")
      start = perf_counter()
      eval_dir = self._get_train_eval_dir(run_dir, 0, global_step)
      results = self._run_eval(_base_model, eval_runners, global_step, runtime.seed, eval_dir)
      self._log_results(results, summary_writer, global_step, perf_counter() - start, eval_dir)
      if is_primary() and self.best_model_key:
        best_saved_score = self._get_model_score(results)

    # Ready to start!
    if global_step > 0:
      logging.info(f"Resume training from ep={train_state.epoch} global_step={global_step}")
    else:
      logging.info(f"Start training")

    n_train = sum(p.requires_grad for p in model.parameters())
    n_freeze = sum(not p.requires_grad for p in model.parameters())
    logging.info(f"Have {n_train} params and {n_freeze} frozen parameters")

    for epoch in range(train_state.epoch, self.epochs):
      ep_start = perf_counter()
      model.train()

      if hasattr(tr_sampler, "set_epoch"):
        # Some samplers need set_epoch to be set each epoch
        tr_sampler.set_epoch(epoch)

      if is_primary():
        pbar = tqdm(train_loader, disable=not self.epoch_pbar, ncols=100, desc="loss=", total=n_train_batches)
      else:
        # We don't just use tqdm(... disable=True) since that seems to cause visual
        # issues in multi-process settings
        pbar = train_loader

      for batch in pbar:
        # Backprop/collect monitor information
        if runtime.grad_accumulation > 1:
          n = len(batch)
          monitor = defaultdict(float)
          total_loss = 0

          if not is_distributed():
            for sub_batch in batch:
              sub_batch = pytorch_utils.to_device(sub_batch, device)
              loss, sub_monitor = model(**sub_batch)
              for k, v in _remove_tensors(sub_monitor).items():
                monitor[k] += v/n
              total_loss += loss.item()
              (loss / n).backward()
          else:
              for sub_i, sub_batch in enumerate(batch):
                sub_batch = pytorch_utils.to_device(sub_batch, device)
                if sub_i < n - 1:
                  with model.no_sync():
                    loss, sub_monitor = model(**sub_batch)
                    total_loss += loss.item()
                    (loss / n).backward()
                else:
                  loss, sub_monitor = model(**sub_batch)
                  total_loss += loss.item()
                  (loss / n).backward()
                for k, v in _remove_tensors(sub_monitor).items():
                  monitor[k] += v/n
          loss = total_loss / n
          monitor = {k: v for k, v in monitor.items()}
        else:
          batch = pytorch_utils.to_device(batch, device)
          loss, monitor = model(**batch)
          monitor = _remove_tensors(monitor)
          loss.backward()
          loss = loss.item()

        if self.clip_grad_norm is not None:
          torch.nn.utils.clip_grad_norm_(clip_params, self.clip_grad_norm)

        optimizer.step()
        if not np.isfinite(loss):
          raise ValueError(f"non-finite foss {loss}")

        # Manually remove gradients, slightly faster then `optimizer.zero_grad`
        for group in optimizer.param_groups:
          for p in group['params']:
            p.grad = None

        global_step += 1
        if step_scheduler is not None:
          step_scheduler.step()

        if is_distributed() and self.sync_monitor:
          # Gather `monitor` from each work to primary so we can log the global average
          # We use `all_gather_object` so things work even if monitor had different
          # keys on different processes
          out = [None] * world_size
          dist.all_gather_object(out, (loss, monitor))
          if is_primary():
            loss = np.mean([x[0] for x in out])
            monitor = py_utils.transpose_list_of_dicts([x[1] for x in out])
            monitor = {k: np.mean(v) for k, v in monitor.items()}

        if is_primary():
          for k, v in monitor.items():
            if k not in monitor_ema:
              monitor_ema[k] = v
              to_show = v
            else:
              cur = monitor_ema[k]
              ema = cur * self.monitor_ema + v * (1 - self.monitor_ema)
              monitor_ema[k] = ema
              to_show = (ema / (1 - self.monitor_ema ** global_step))

            if summary_writer is not None and global_step % self.tb_log_intervals == 0:
              summary_writer.add_scalar(f"train/{k}", to_show, global_step)

          loss_ema = loss_ema * self.loss_logging_ema + loss * (1 - self.loss_logging_ema)
          corrected_loss_ema = (loss_ema / (1 - self.loss_logging_ema ** global_step))
          pbar.set_description("loss=%.4f" % corrected_loss_ema, refresh=False)

          if summary_writer is not None and global_step % self.tb_log_intervals == 0:
            summary_writer.add_scalar("train/loss-smoothed", corrected_loss_ema, global_step)
            summary_writer.add_scalar("train/loss", loss, global_step)

            if self.log_lr:
              for j, group in enumerate(optimizer.param_groups):
                name = group.get("name", f"group_{j}")
                summary_writer.add_scalar(f'lr/{name}', group["lr"], global_step)

            if self.log_frozen_parameters:
              for j, group in enumerate(optimizer.param_groups):
                name = group.get("name", f"group_{j}")
                n_frozen = sum(not x.requires_grad for x in group["params"]) / len(group["params"])
                if n_frozen > 0:
                  summary_writer.add_scalar(f'lr/{name}-frozen', n_frozen, global_step)

      ep_end = perf_counter()
      if self.eval_at is not None and (epoch+1) % self.eval_at != 0:
        continue

      logging.info(f"Epoch {epoch + 1} took {py_utils.duration_to_str(ep_end - ep_start)}, starting evaluation")

      eval_start = perf_counter()
      # Just eval the base model since we don't need any synchronization between models
      eval_dir = self._get_train_eval_dir(run_dir, epoch, global_step)
      results = self._run_eval(_base_model, eval_runners, global_step, runtime.seed, eval_dir)
      eval_end = perf_counter()

      self._log_results(results, summary_writer, global_step, eval_end-eval_start, eval_dir)

      if summary_writer:
        summary_writer.add_scalar("time/train", ep_end-ep_start, epoch+1)
        summary_writer.add_scalar("time/eval", eval_end - eval_start, epoch + 1)

      if self.best_model_key and is_primary():
        score = self._get_model_score(results)
        if best_saved_score is None or best_saved_score < score:
          prefix = "Saving as" if run_dir else "Found"
          if best_saved_score is None:
            logging.info(f"{prefix} best model ({score:.5f}) ep={epoch+1}")
          else:
            logging.info(f"{prefix} best model ({score:.5f} > {best_saved_score:.5f}) ep={epoch+1}")
          best_saved_score = score
          if run_dir:
            best_model_file = join(run_dir, BEST_STATE_NAME)
            torch.save(_base_model.state_dict(), best_model_file)

      if (run_dir is not None and
          self.save_each_epoch and
          epoch % self.save_each_epoch == 1):
        state_file = join(run_dir, f"state-ep{epoch+1}.pth")
        logging.info(f"Saving state as {state_file}")
        torch.save(_base_model.state_dict(), state_file)

      if run_dir is not None and self.checkpoint:
        checkpoint = _TrainingState(
            epoch=epoch+1,  # +1 because the epoch has ended
            loss_ema=loss_ema,
            global_step=global_step,
            monitor_ema=monitor_ema,
            scheduler_state=None if step_scheduler is None else step_scheduler.state_dict(),
            optimizer_state=optimizer.state_dict(),
            best_save_score=best_saved_score,
            model_state=_base_model.state_dict()
        )
        checkpoint_file = join(run_dir, "checkpoint.pth")
        logging.info(f"Checkpointing to {checkpoint_file}")
        torch.save(checkpoint, checkpoint_file)

      if epoch == self.end_at_epoch:
        logging.info(f"Hit epoch {self.end_at_epoch}, ending early")
        break
