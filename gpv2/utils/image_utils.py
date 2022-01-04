from os import listdir
from os.path import join, exists
from typing import Union

import numpy as np

import imagesize
import torch
from PIL import Image

from gpv2 import file_paths
from gpv2.utils import py_utils

_IMAGE_ID_TO_SIZE_MAP = {}

IMAGE_SOURCE_MAP = {
  "imsitu": file_paths.IMSITU_IMAGE_DIR,
  "coco": file_paths.COCO_IMAGES,
  "web": file_paths.WEB_IMAGES_DIR,
  "dce": file_paths.DCE_IMAGES,
}


def get_image_file(image_id) -> str:
  """Returns the filepath of an image corresponding to an input image id

  To support multiple datasets, we prefix image_ids with {source/}
  """
  source, key = image_id.split("/", 1)
  if source in IMAGE_SOURCE_MAP:
    return join(IMAGE_SOURCE_MAP[source], key)
  raise ValueError(f"Unknown image id {image_id}")


def get_box_key(box):
  crop = np.array(box, dtype=np.float32).tobytes()
  return py_utils.consistent_hash(crop)


def get_hdf5_files(feature_name):
  if exists(join(file_paths.PRECOMPUTED_FEATURES_DIR, feature_name)):
    src = join(file_paths.PRECOMPUTED_FEATURES_DIR, feature_name)
    return [join(src, x) for x in listdir(src)]

  # Old format, one folder per dataset
  out = []
  for ds_dir in listdir(file_paths.PRECOMPUTED_FEATURES_DIR):
    ds_dr = join(file_paths.PRECOMPUTED_FEATURES_DIR, ds_dir, feature_name + ".hdf5")
    if exists(ds_dr):
      out.append(ds_dr)
  return out


BUILD_SUBGROUPS_FOR = ["web", "coco"]


def get_hdf5_key_for_image(image_id, crop=None) -> str:
  """Returns the key we would use in HDF5 for the given image_id

  This mapping is a bit convoluted since we are have tried to keep hdf5 feature files
  backward-compatible with newer image_id formats.
  """

  prefix, key = image_id.split("/", 1)

  if prefix == "coco":
    if key.startswith("test2015"):
      hdf5_key = f"coco-{key.split('.')[-1]}"
    else:
      hdf5_key = key.split("_")[-1].split(".")[0].lstrip("0")
  else:
    # Slashes lead to nested HDF5 groups, so replace with "-"
    hdf5_key = key.replace("/", "-")

  if crop is None:
    return hdf5_key
  return f"{hdf5_key}-{get_box_key(crop)}"


def crop_img(img: Union[np.ndarray, Image.Image], crop):
  if crop is None:
    return img
  x, y, w, h = crop

  if isinstance(img, np.ndarray):
    H, W = img.shape[:2]
  else:
    W, H = img.size

  if all(c <= 1.0 for c in crop):
    # Assume the crop is in normalized coordinates
    x, y, w, h = x*W, y*H, w*W, h*H

  if w < 5: w = 5
  if h < 5: h = 5
  x1 = x - 0.2 * w
  x2 = x + 1.2 * w
  y1 = y - 0.2 * h
  y2 = y + 1.2 * h
  x1, x2 = [min(max(0, int(z)), W) for z in [x1, x2]]
  y1, y2 = [min(max(0, int(z)), H) for z in [y1, y2]]
  if isinstance(img, np.ndarray):
    return img[y1:y2, x1:x2]
  else:
    return img.crop((x1, y1, x2, y2))


def get_image_size(image_id):
  if image_id in _IMAGE_ID_TO_SIZE_MAP:
    return _IMAGE_ID_TO_SIZE_MAP[image_id]

  img_file = get_image_file(image_id)
  size = imagesize.get(img_file)

  _IMAGE_ID_TO_SIZE_MAP[image_id] = size
  return size


def get_coco_int_id(image_id) -> int:
  if isinstance(image_id, int):
    return image_id
  return int(image_id.split("_")[-1].split(".")[0])


def normalize_boxes(boxes, image_id):
  w, h = get_image_size(image_id)
  div = torch.as_tensor([w, h, w, h])
  return boxes / div.unsqueeze(0)
