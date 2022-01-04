"""Evaluates pre-computed predictions for a model"""

import argparse
import logging
from collections import defaultdict
from os import listdir
from os.path import join, exists

from gpv2.data.dataset import Task
from gpv2.data.dce_dataset import DceDataset
from gpv2.data.gpv_datasets import GpvDataset
from gpv2.eval.dataset_cli import add_dataset_args, get_datasets_from_args
from gpv2.eval.evaluation import get_evaluator, save_evaluation
from gpv2.train.evaluator import ResultKey
from gpv2.train.runner import load_gpv_predictions
from gpv2.utils import py_utils
from gpv2.utils.py_utils import load_json_object, val_to_str


def find_eval_files(run_dir, prefix):
  output = defaultdict(list)
  eval_dir = join(run_dir, "eval")
  if not exists(eval_dir):
    return output
  for subdir_name in listdir(eval_dir):
    subdir = join(eval_dir, subdir_name)
    if subdir_name.startswith(prefix):
      if exists(join(subdir, "predictions.json")) or exists(join(subdir, "eval.json")):
        eval_name = subdir_name.split("--")[-1]
        config = load_json_object(join(subdir, "config.json"))
        ds = config["dataset"]
        n_sample = (ds["sample"], ds.get("seen_sample", None), ds.get("unseen_sample", None))
        n_sample = sum(0 if x is None else x for x in n_sample)
        output[eval_name].append((subdir, None if n_sample == 0 else n_sample))

  def _get_order(x):
    return 1e9 if x[1] is None else x[1]

  consolidated_out = {}
  for k, v in output.items():
    v.sort(key=_get_order, reverse=True)
    consolidated_out[k] = v[0][0]

  return consolidated_out


def get_eval_if_cached(eval_dir):
  cache_file = join(eval_dir, "eval.json")
  if exists(cache_file):
    cached = load_json_object(cache_file)
    if isinstance(cached, list) and "in-domain" in cached[0]:
      # a nocaps eval file
      stats = {}
      for part in cached:
        for subset, r in part.items():
          for metric_name, val in r.items():
            stats[ResultKey(metric_name, subset)] = val
      return stats

    stats_str = cached["stats"]
    stats = {}
    for k, v in stats_str.items():
      subset_name, metric_name = k.split("/")
      if subset_name == "all":
        subset_name = None
      stats[ResultKey(metric_name, subset_name)] = v
    return stats
  return None


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  add_dataset_args(parser, sample=False)

  parser.add_argument("--eval_name", default=None)
  parser.add_argument("--sample", type=int, help="Evaluate on a subset of examples")
  parser.add_argument("--nocache", action="store_true",
                      help="Re-run the evaluation even if it has already been pre-computed")
  args = parser.parse_args()
  py_utils.add_stdout_logger()

  args.cache = not args.nocache
  if args.sample and args.cache:
    raise ValueError("Cannot sample if caching")

  run_dir = py_utils.select_run_dir(args.model)

  datasets = get_datasets_from_args(args, run_dir, sample=False)

  for dataset in datasets:
    if args.eval_name:
      eval_dir = join(run_dir, "eval", dataset.get_name() + "--" + args.eval_name)
      if not exists(eval_dir):
        logging.info(f"Missing expected eval dir: {eval_dir}")
        continue
    else:
      eval_dir = [x for x in listdir(join(run_dir, "eval")) if x.startswith(dataset.get_name() + "--")]
      if len(eval_dir) == 0:
        logging.info(f"No predictions for model {dataset.get_name()}: {run_dir}")
        continue
      elif len(eval_dir) > 1:
        logging.info(f"Multiple predictions for model {dataset.get_name()}: {run_dir}, "
                     f"please specify `eval_name`")
        continue
      else:
        eval_dir = join(run_dir, "eval", eval_dir[0])

      if args.nocache:
        cached = None
      else:
        cached = get_eval_if_cached(eval_dir)

      if cached:
        logging.info(f"Loaded cached stats for {eval_dir}")
        stats = cached
      else:
        logging.info(f"Evaluating {dataset.get_name()}")
        evaluator, get_subsets = get_evaluator(dataset)
        if evaluator is None:
          continue
        load_boxes = False
        if isinstance(dataset, (GpvDataset, DceDataset)):
          load_boxes = dataset.task == Task.LOCALIZATION

        pred = load_gpv_predictions(eval_dir, load_boxes)

        stats = evaluator.evaluate(dataset.load(), pred, subset_mapping=get_subsets,
                                   allow_partial=True)
        if not args.nocache:
          save_evaluation(eval_dir, evaluator, stats)

      k = [k for k in stats if k.metric_name == "n"]
      if len(k) == 1:
        del stats[k[0]]

      table = [[str(r) for r in stats], [val_to_str(k, v) for k, v in stats.items()]]
      print("*" * 20 + " " + dataset.get_name() + " " + "*"*20)
      print(py_utils.table_string(table))


if __name__ == '__main__':
  main()