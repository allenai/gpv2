import logging
from argparse import ArgumentParser

from gpv2.data.dataset import Task
from gpv2.data.gpv_datasets import GpvDataset, CocoCategories
from gpv2.model.model import BeamSearchSpec
from gpv2.train import evaluator
from gpv2.train.evaluator import ResultKey
from gpv2.train.runner import DataLoaderBuilder
from gpv2.train.trainer import Trainer, TrainerDataset, EvaluationSetup, RunArgs
from gpv2.utils.pytorch_utils import get_devices


def add_train_args(parser: ArgumentParser, batch_size=32, num_workers=6,
                   epochs=8, tasks="all", clip_grad_norm=None, find_unused_parameters=True):
  """Add a bunch of args to `parser` that are likely to be useful for multiple models

  The extra args specify the defaults
  """
  parser.add_argument("--task", nargs="+", default=tasks,
                      help="Tasks for train on")
  parser.add_argument("--sce", action="store_true")

  # Performance args
  parser.add_argument("--device", nargs="+", default=None,
                      help="List of integer GPU devices to train on")
  parser.add_argument("--dist_port", default=None, type=int,
                      help="Port to syn distributed training on")
  parser.add_argument("--grad_accumulation", type=int, default=1,
                      help="Accumulate gradients over n step, the effective batch_size will "
                           "be unchanged")
  parser.add_argument("--force_one_worker", action="store_true",
                      help="For debugging, run distributed code even only 1 device is given")
  parser.add_argument("--nopin_memory", action="store_true",
                      help="Turn off memory pinning")
  parser.add_argument("--num_workers", default=num_workers, type=int,
                      help="Number of workers to use")
  parser.add_argument("--find_unused_parameters", action="store_true")

  # Other training args
  parser.add_argument("--clip_grad_norm", default=clip_grad_norm, type=float)
  parser.add_argument("--batch_size", default=batch_size, type=int)
  parser.add_argument("--epochs", default=epochs, type=int)
  parser.add_argument("--debug", choices=["tiny", "small", "med", "large"], default=None)

  # Output args
  parser.add_argument("--eval_start", action="store_true")
  parser.add_argument("--override", action="store_true")
  parser.add_argument("--output_dir")


def run_train(args, model, **kwargs):
  trainer = get_trainer_from_args(args, **kwargs)
  run_trainer_from_args(trainer, model, args)


COCO_EVAL = {
  Task.VQA: EvaluationSetup(
    evaluator.VqaEvaluator(),
    dict(beam_search_spec=BeamSearchSpec(1, 10))
  ),
  Task.CAPTIONING: EvaluationSetup(
    evaluator.CaptionEvaluator(),
    dict(beam_search_spec=BeamSearchSpec(1, 30))
  ),
  Task.LOCALIZATION: EvaluationSetup(
    evaluator.LocalizationEvaluator(),
    dict(beam_search_spec=None)
  ),
  Task.CLS: EvaluationSetup(
    evaluator.ClsEvaluator(),
    dict(beam_search_spec=BeamSearchSpec(1, 5), answer_options=CocoCategories())
  ),
  Task.CLS_IN_CONTEXT: EvaluationSetup(
    evaluator.ClsEvaluator(),
    dict(beam_search_spec=BeamSearchSpec(1, 5), answer_options=CocoCategories())
  ),
}


def get_trainer_from_args(
    args, optimizer, logging_ema=0.99, sync_monitor=True, scheduler=None) -> Trainer:
  batch_size, num_workers = args.batch_size, args.num_workers

  if args.debug:
    dbg_batch_size, dbg_num_workers = {
      "tiny": (2, 0),
      "small": (8, 0),
      "med": (24, 4),
      "large": (60, 4),
    }[args.debug]
    if not hasattr(args, "batch_size_not_default"):
      batch_size = dbg_batch_size
    if not hasattr(args, "num_workers_not_default"):
      num_workers = dbg_num_workers

  logging.info(f"batch size={batch_size}")
  logging.info(f"num_workers={num_workers}")
  logging.info(f"lr={args.lr}")
  if args.grad_accumulation != 1:
    logging.info(f"grad acc={args.grad_accumulation}")

  train_datasets = []
  eval_datasets = []
  tasks = {}  # Use a dictionary to preserve ordering with uniqueness
  for dataset in args.task:
    if dataset == "all":
      tasks.update({x: None for x in Task})
    elif dataset == "non-cls":
      tasks.update({x: None for x in [Task.VQA, Task.CAPTIONING, Task.DETECTION]})
    else:
      tasks[Task(dataset)] = None

  for task in tasks:
    train_datasets.append(TrainerDataset(
      GpvDataset(task, "train", args.sce), str(task) + "-tr", eval_setup=COCO_EVAL[task]))
    eval_datasets.append(TrainerDataset(
      GpvDataset(task, "val", args.sce), str(task) + "-val", eval_setup=COCO_EVAL[task]))

  best_model_key = [
    ResultKey("accuracy", dataset_name="cls-val"),
    ResultKey("accuracy", dataset_name="cic-val"),
    ResultKey("score", dataset_name="vqa-val"),
    ResultKey("cider", dataset_name="cap-val"),
    ResultKey("AP", dataset_name="det-val"),
    ResultKey("accuracy", dataset_name="webqa-val"),
  ]
  best_model_key = [x for x in best_model_key if any(x.dataset_name.startswith(str(t)) for t in tasks)]

  if args.debug == "tiny":
    for x in train_datasets:
      x.dataset.sample = 5
      if x.eval_sample != 0:
        x.eval_sample = 0
    for x in eval_datasets:
      x.dataset.sample = 5
      if x.eval_sample != 0:
        x.eval_sample = 4

  elif args.debug == "small":
    for x in train_datasets:
      x.dataset.sample = 120
      x.eval_sample = 30
    for x in eval_datasets:
      x.dataset.sample = 120
      x.eval_sample = 30

  elif args.debug == "med":
    for x in train_datasets:
      x.dataset.sample = 2000
      x.eval_sample = 500
    for x in eval_datasets:
      x.dataset.sample = 2000
      x.eval_sample = 500

  elif args.debug == "large":
    for x in train_datasets:
      x.dataset.sample = 10000
      x.eval_sample = 2000
    for x in eval_datasets:
      x.eval_sample = 2000

  else:
    for x in train_datasets:
      if x.dataset.task == Task.CAPTIONING:
        x.eval_sample = 5000
      else:
        x.eval_sample = 8000
    for x in eval_datasets:
      if x.dataset.task == Task.CAPTIONING:
        x.eval_sample = 8000
      else:
        x.eval_sample = 12000

  train_loader = DataLoaderBuilder(batch_size, num_workers, not args.nopin_memory,
                                   prefetch_factor=2, persist_workers=num_workers > 0)

  # other_log specifies additional tensorboard logging outputs, we use it to
  # have a second tab with results grouped by train/eval rather than by dataset
  other_log = {}
  evals = [(x, True) for x in train_datasets] + [(x, False) for x in eval_datasets]
  for ds, is_train in evals:
    task = ds.dataset.task
    if task == Task.CAPTIONING:
      metric_name, name = "cider", "cider"
      k = evaluator.ResultKey(metric_name="bleu4", dataset_name=ds.get_name())
      other_log[k] = "bleu4"
    elif task == Task.CLS:
      metric_name, name = "accuracy", "cls"
    elif task == Task.VQA:
      metric_name, name = "score", "vqa"
    elif task == Task.LOCALIZATION:
      metric_name, name = "AP", "loc"
    elif task == Task.CLS_IN_CONTEXT:
      metric_name, name = "accuracy", "ident"
    elif task == Task.WEBQA:
      metric_name, name = "accuracy", "webqa"
    elif task == Task.HOI:
      continue
    else:
      raise NotImplementedError(task)
    name = f"train-evals/{name}" if is_train else f"val-evals/{name}"
    other_log[evaluator.ResultKey(metric_name=metric_name, dataset_name=ds.get_name())] = name

  trainer = Trainer(
    train_datasets,
    eval_datasets,
    optimizer,

    train_loader=train_loader,
    step_schedule=scheduler,

    save_evaluation_results=True,
    save_prediction_samples=500,

    train_val_log=list(other_log.items()),
    find_unused_parameters=not args.find_unused_parameters,

    epochs=args.epochs,
    best_model_key=best_model_key,
    clip_grad_norm=args.clip_grad_norm,
    tb_log_intervals=20,
    checkpoint=True,
    sync_monitor=sync_monitor,
    eval_at_start=args.eval_start,
    loss_logging_ema=logging_ema,
    monitor_ema=logging_ema,
  )

  return trainer


def run_trainer_from_args(trainer, model, args):
  devices = RunArgs.build(get_devices(args.device), args.force_one_worker, args.grad_accumulation)
  trainer.train(model, args.output_dir, devices, override=args.override)
