"""Script to download all the data we need for GPV experiments"""

import argparse
import logging
import tarfile
import tempfile
from os import makedirs
from os.path import exists, join

from gpv2 import file_paths
from gpv2.data.dataset import Task
from gpv2.data.dce_dataset import DceDataset
from gpv2.utils import py_utils, image_utils
from gpv2.utils.downloader import download_zip, download_from_s3, download_s3_folder, \
  download_to_file, download_files

GPV_DATA = "https://ai2-prior-gpv.s3-us-west-2.amazonaws.com/public/coco_original_and_sce_splits.zip"


def download_gpv_json():
  if not exists(join(file_paths.GPV_DATA_DIR, "coco_captions")):
    logging.info(f"Download GPV json")
    download_zip(GPV_DATA, file_paths.GPV_DATA_DIR, True)
  else:
    logging.info("GPV annotation files already exists")


COCO_URLS = {
  "train2014": "http://images.cocodataset.org/zips/train2014.zip",
  "val2014": "http://images.cocodataset.org/zips/val2014.zip",
  "test2014": "http://images.cocodataset.org/zips/test2014.zip",
  "test2015": "http://images.cocodataset.org/zips/test2015.zip"
}


def download_coco_images(subset):
  output_dir = join(file_paths.COCO_IMAGES, subset)
  if not exists(output_dir):
    logging.info(f"Download COCO images {subset}")
    download_zip(COCO_URLS[subset], file_paths.COCO_IMAGES, True)
  else:
    logging.info(f"Already have {subset} COCO images")


DCE_IMAGES = "s3://ai2-prior-git/images/opensce_v1_images.tar.gz"


def download_dce_images(n_procs=15):
  if not exists(join(file_paths.DCE, "samples")):
    raise ValueError("Download DCE annotations firsts")

  targets = []
  for task in Task:
    for part in ["val", "test"]:
      for ex in DceDataset(task, part).load():
        image_file = image_utils.get_image_file(ex.image_id)
        if exists(image_file):
          continue
        if task == Task.VQA:
          # URL for visual genome images
          url = f"https://cs.stanford.edu/people/rak248/{'/'.join(ex.image_id.split('/')[-2:])}"
        else:
          # TODO Downloading s3 data from the URL seems pretty slow
          # URL for open-images
          url = f"https://open-images-dataset.s3.amazonaws.com/validation/{ex.image_id.split('/')[-1]}"
        targets.append((url, image_file))

  if len(targets) == 0:
    logging.info(f"DCE images already exist")
    return

  logging.info("Downloading DCE images...")
  download_files(targets, n_procs=n_procs)


VAL_NOCAPS_URL = "https://s3.amazonaws.com/nocaps/nocaps_val_image_info.json"
TEST_NOCAPS_URL = "https://s3.amazonaws.com/nocaps/nocaps_test_image_info.json"


def download_dce_annotations():
  if exists(join(file_paths.DCE, "samples")):
    logging.info(f"DCE annotations are already downloaded")
    return
  logging.info(f"Downloading DCE annotations")
  download_s3_folder("ai2-prior-dce", "v1", file_paths.DCE)

  logging.info(f"Downloading nocaps image info")
  download_to_file(VAL_NOCAPS_URL, file_paths.NOCAPS_VAL_IMAGE_INFO)
  download_to_file(TEST_NOCAPS_URL, file_paths.NOCAPS_TEST_IMAGE_INFO)


VINVL_FEATURE_BUCKET = "ai2-prior-gpv"
VINVL_FEATURE_KEYS = {
  "coco": "precomputed-image-features/coco/vinvl.hdf5",
  "dce": "precomputed-image-features/opensce/vinvl.hdf5",
}


def download_vinvl_features(dataset):
  out_file = join(file_paths.PRECOMPUTED_FEATURES_DIR, "vinvl", f"{dataset}.h5py")
  if not exists(out_file):
    logging.info(f"Downloading VinVL features for {dataset}")
    download_from_s3(
      VINVL_FEATURE_BUCKET, VINVL_FEATURE_KEYS[dataset],
      join(file_paths.PRECOMPUTED_FEATURES_DIR, "vinvl", f"{dataset}.h5py"), True)
  else:
    logging.info(f"Already have VinVL features for {dataset}")


VINVL_URL = "https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth"


def download_vinvl():
  if exists(file_paths.VINVL_STATE):
    logging.info("VinVL state already exists")
  else:
    logging.info("Downloading VinVL")
    download_to_file(VINVL_URL, file_paths.VINVL_STATE, True)


DOWNLOAD_FNS = {
  "dce-anno": download_dce_annotations,
  "dce-images": download_dce_images,
  "gpv-anno": download_gpv_json,
  "vinvl": download_vinvl,
}


for x in COCO_URLS:
  DOWNLOAD_FNS[f"coco-{x}"] = lambda x=x: download_coco_images(x)
for x in VINVL_FEATURE_KEYS:
  DOWNLOAD_FNS[f"vinvl-{x}"] = lambda x=x: download_vinvl_features(x)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--targets",
    choices=["all"] + list(DOWNLOAD_FNS),
    default="all"
  )
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  target = args.targets
  if target == "all":
    for v in DOWNLOAD_FNS.values():
      v()
  else:
    DOWNLOAD_FNS[target]()


if __name__ == '__main__':
  main()