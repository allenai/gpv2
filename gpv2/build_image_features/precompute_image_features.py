import argparse
import logging
from collections import defaultdict
from os import walk
from os.path import join, relpath

import numpy as np

from gpv2.build_image_features.compute_features import add_args, ExtractionTarget, run
from gpv2.utils import py_utils


def main():
  parser = argparse.ArgumentParser(description="Extract features for images in a directory")
  parser.add_argument("image_source", help="Directory containing images or name of dataset")
  parser.add_argument("dataset_name", help="Directory containing images or name of dataset")
  add_args(parser)


  args = parser.parse_args()

  default_query_box = None if args.no_query else (0, 0, 1.0, 1.0)
  targets = []
  queries = defaultdict(set)

  py_utils.add_stdout_logger()

  default_query_arr = np.array([default_query_box])
  assert default_query_arr.shape == (1, 4)
  for dirpath, dirnames, filenames in walk(args.image_source):
    for filename in filenames:
      filename = join(dirpath, filename)
      image_id = relpath(filename, args.image_source)
      assert ".." not in image_id  # Sanity check the relpath
      image_id = args.dataset_name + "/" + image_id
      targets.append(ExtractionTarget(image_id, filename, None, default_query_arr))

  for (image_id, crop), parts in queries.items():
    parts = [x for x in parts if x is not None]
    qboxes = np.array(parts, dtype=np.float32) if parts else None
    targets.append(ExtractionTarget(image_id, None, crop, qboxes))
  logging.info(f"Running on {len(targets)} images")
  run(targets, args)


if __name__ == '__main__':
  main()