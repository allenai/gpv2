
import argparse
import json
import logging
import os

import h5py

from gpv2.data.dataset import Task, Dataset
from gpv2.data.dce_dataset import DceDataset
from gpv2.data.gpv_datasets import GpvDataset
from gpv2.data.webqa_dataset import WebQaDataset
from gpv2.eval.dataset_cli import get_datasets_from_args, add_dataset_args
from gpv2.eval.evaluation import get_evaluator, save_evaluation
from gpv2.model.model import BeamSearchSpec
from gpv2.train.evaluator import ResultKey
from gpv2.train.runner import run, save_gpv_output, prediction_args_to_json
from gpv2.utils import py_utils, pytorch_utils
from gpv2.utils.to_params import to_params

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime
from os.path import join, exists, dirname
from shutil import rmtree


# These make sense for T5 based on the train data, maybe not for other models
DEFAULT_MAX_SEQ_LEN = {
  Task.VQA: 20,
  Task.CLS: 8,
  Task.CLS_IN_CONTEXT: 8,
  Task.CAPTIONING: 30
}


def get_default_seq_len(ds: Dataset):
  if isinstance(ds, (GpvDataset, DceDataset)):
    return DEFAULT_MAX_SEQ_LEN[ds.task]
  elif isinstance(ds, WebQaDataset):
    return 8
  else:
    raise NotImplementedError()


def eval_on(args, run_dir, dataset, devices, skip_existing=True):
  if args.output_dir:
    output_dir = args.output_dir

  elif args.output_name:
    name = f"{dataset.get_name()}--{args.output_name}"
    eval_dir = join(run_dir, "eval")
    if not exists(eval_dir):
      os.mkdir(eval_dir)
    output_dir = join(eval_dir, name)
  else:
    output_dir = None

  if output_dir is not None:
    if exists(output_dir):
      if len(os.listdir(output_dir)) > 0:
        if skip_existing:
          logging.info(f"{output_dir} already exists, skipping")
          return

        if args.override or py_utils.get_yes_no(f"{output_dir} exists, delete (y/n)?"):
          logging.info(f"Deleting {output_dir}")
          rmtree(output_dir)
        else:
          logging.info("No override, not stopping")
          return
    elif not exists(dirname(output_dir)):
      raise ValueError(f"Parent folder {dirname(output_dir)} does not exist")
    else:
      logging.info(f"Will save to {output_dir}")
  else:
    logging.info(f"Not saving the output")

  if output_dir:
    if not exists(output_dir):
      os.mkdir(output_dir)
    logging.info(f"Saving output to {output_dir}")

  if isinstance(dataset, (GpvDataset, DceDataset)):
    task = dataset.task
  else:
    task = None

  logging.info("Setting up...")
  examples = dataset.load()

  if args.rank_answer_options == "always":
    do_rerank = task in {Task.CLS, Task.CLS_IN_CONTEXT}
  elif args.rank_answer_options == "never":
    do_rerank = False
  else:
    raise NotImplementedError(args.rank_answer_options)

  prediction_args = {}
  beams_to_keep = args.beams_to_keep
  batch_size = args.batch_size

  if isinstance(dataset, (GpvDataset, DceDataset)):
    task = dataset.task
    if task in {Task.CLS, Task.CLS_IN_CONTEXT}:
      answer_options = dataset.get_answer_options(False)
      prediction_args["answer_options"] = answer_options
      logging.info("Classification so keeping 20 beams")
      beams_to_keep = 20

  if do_rerank and prediction_args.get("answer_options"):
    logging.info(f"Re-ranking answer options")
    logging.info(f"Reducing batch size to 5")
    batch_size = 5
    prediction_args["rerank_answer_options"] = True
  else:
    if args.max_seq_len:
      max_seq_len = args.max_seq_len
    elif task == Task.LOCALIZATION:
      max_seq_len = None
    else:
      max_seq_len = get_default_seq_len(dataset)
      logging.info(f"Defaulting to max_seq_len {max_seq_len} for task {task}")

    if max_seq_len is not None:
      bs = BeamSearchSpec(beam_size=args.beam_size, max_seq_len=max_seq_len)
    else:
      bs = None
    prediction_args["beam_search_spec"] = bs

  if args.dry_run:
    logging.info("Skipping running the model since this is a dry run")
    return

  output = run(
    run_dir, examples, devices, batch_size, args.num_workers,
    prediction_args, beams_to_keep=beams_to_keep)

  if output_dir is not None:
    logging.info(f"Saving output to {output_dir}")
    save_gpv_output(output, output_dir)

    config = dict(
      batch_size=batch_size,
      num_workers=args.num_workers,
      predictions_args=prediction_args_to_json(prediction_args),
      dataset=to_params(dataset, Dataset),
      beams_to_keep=beams_to_keep,
      date=datetime.now().strftime("%m%d-%H%M%S"),
    )

    with open(output_dir + "/config.json", "w") as f:
      json.dump(config, f, indent=2)

  if args.eval:
    if isinstance(dataset, GpvDataset) and not dataset.gpv_split and dataset.split == "test":
      logging.info("Skip evaluating since no labels for COCO test")
      return
    if isinstance(dataset, DceDataset) and dataset.task == Task.CAPTIONING:
      logging.info("Skip evaluating since no labels for DCE captioning")
      return
    else:
      logging.info("Evaluating...")
    evaluator, subsets = get_evaluator(dataset)

    results = evaluator.evaluate(examples, output, allow_partial=True, subset_mapping=subsets)
    k = [k for k in results if k.metric_name == "n"]
    if len(k) == 1:
      del results[k[0]]

    if output_dir is not None:
      results[ResultKey("n", None)] = len(output)
      logging.info(f"Caching evaluation to {output_dir}")
      save_evaluation(output_dir, evaluator, results)

    if task != Task.CAPTIONING:
      factor = 100
    else:
      factor = 1
    results = {str(k): v*factor for k, v in results.items()}
    print(json.dumps(results, indent=2))


def main():
  parser = argparse.ArgumentParser(description="Compute predictions for a GPV model")
  parser.add_argument("model")
  add_dataset_args(parser)
  parser.add_argument("--device", nargs="+", default=[None], help="GPU devices to use")
  parser.add_argument("--batch_size", type=int, default=30)
  parser.add_argument("--num_workers", type=int, default=4)
  parser.add_argument("--beams_to_keep", type=int, default=5, help="Number of predictions to save")
  parser.add_argument("--max_seq_len", type=int, default=None)
  parser.add_argument("--beam_size", type=int, default=20)
  parser.add_argument("--eval", action="store_true", help="Evaluate the results")
  parser.add_argument("--override", action="store_true", help="Delete output dir if it exists")
  parser.add_argument("--output_dir", help="Save to this directory")
  parser.add_argument("--output_name",
                      help="Save results in model/run/eval/{dataset_name}--{output_name}")
  parser.add_argument("--dry_run", action="store_true")
  parser.add_argument("--rank_answer_options", default="always",
                      choices=["never", "always"],
                      help="When to use answer options ranking for classification")
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  if args.output_dir and args.output_name:
    raise ValueError("Cannot specify output_name and output_dir")

  models = py_utils.find_models(args.model)
  if len(models) == 0:
    logging.info("No models selected")
    return

  devices = pytorch_utils.get_devices(args.device)
  if args.output_dir:
    models = py_utils.flatten_list(x[1] for x in models.values())
    if len(models) > 1:
      raise ValueError("Cannot use one output dir if more than one model selected!")
    model = models[0]

    datasets = get_datasets_from_args(args, model)
    if len(datasets) > 1:
      raise ValueError("Cannot use one output dir if more than one dataset is selected!")
    if len(datasets) == 0:
      raise ValueError("No datasets is selected!")
    eval_on(args, model, datasets[0], devices, skip_existing=False)

  else:
    targets = []
    for model_name, (model_dir, runs) in models.items():
      for ds in get_datasets_from_args(args, model_dir):
        for run_dir in runs:
          targets.append((run_dir, ds))

    if len(targets) == 0:
      raise ValueError("No datasets to evaluate on found!")

    for i, (run_dir, dataset) in enumerate(targets):
      if len(targets) > 1:
        logging.info(f"Evaluating on {run_dir} {dataset.get_name()} ({i+1}/{len(targets)})")
      else:
        logging.info(f"Evaluating on {run_dir} {dataset.get_name()}")
      eval_on(args, run_dir, dataset, devices, skip_existing=len(targets) > 1)


if __name__ == '__main__':
  main()
