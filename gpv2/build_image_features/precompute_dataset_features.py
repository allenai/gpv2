import argparse
import logging
from collections import defaultdict

from gpv2.data.dataset import Task
from gpv2.data.dce_dataset import DceDataset
from gpv2.data.gpv_datasets import GpvDataset
from gpv2.data.webqa_dataset import WebQaDataset
from gpv2.build_image_features.compute_features import add_args, ExtractionTarget, run
import numpy as np


def main():
  parser = argparse.ArgumentParser(description="Extract features for a dataset")
  parser.add_argument("dataset", help="Dataset to extract features for")
  parser.add_argument("--sample", type=int, default=None,
                      help="Sample the datasets, used for debugging")
  add_args(parser)

  args = parser.parse_args()

  default_query_box = None if args.no_query else (0, 0, 1.0, 1.0)
  targets = []
  queries = defaultdict(set)

  if args.dataset == "web":
    for split in ["train", "test", "val"]:
      for ex in WebQaDataset(split).load():
        queries[(ex.image_id, None)].add(default_query_box)

  elif args.dataset.startswith("opensce"):
    logging.info("Running on OpenSCE")
    if args.image_source == "opensce":
      tasks = list(Task)
    else:
      tasks = [Task(args.image_source.split("-")[-1])]
    queries = defaultdict(set)
    for task in tasks:
      for part in ["val", "test"]:
        for ex in DceDataset(task, part, sample=args.sample).load():
          crop, qbox = None, default_query_box
          if task == Task.CLS_IN_CONTEXT:
            qbox = tuple(ex.query_box)
          elif task == Task.CLS:
            crop = tuple(ex.crop)
          queries[(ex.image_id, crop)].add(qbox)

  elif args.dataset.startswith("coco"):
    logging.info(f"Running on {args.dataset}")
    tasks = {"coco": list(Task)}
    tasks.update({f"coco-{t}": [t] for t in Task})
    queries = defaultdict(set)
    for task in tasks[args.dataset]:
      parts = ["train", "val"] if task in {Task.CLS, Task.CLS_IN_CONTEXT, Task.LOCALIZATION} else ["train", "val", "test"]
      for part in parts:
        for ex in GpvDataset(task, part, False, sample=args.sample).load():
          crop, qbox = None, default_query_box
          if task == Task.CLS_IN_CONTEXT and not args.no_query:
            qbox = tuple(ex.query_box)
          elif task == Task.CLS:
            crop = tuple(ex.crop)
          queries[(ex.image_id, crop)].add(qbox)
  else:
    raise NotImplementedError(args.image_source)

  for (image_id, crop), parts in queries.items():
    parts = [x for x in parts if x is not None]
    qboxes = np.array(parts, dtype=np.float32) if parts else None
    targets.append(ExtractionTarget(image_id, None, crop, qboxes))
  logging.info(f"Running on {len(targets)} images")
  run(targets, args)


if __name__ == '__main__':
  main()
