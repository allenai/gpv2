import logging
import re
from contextlib import ExitStack
from os import listdir
from os.path import join, exists
from typing import List, Tuple, Any, Dict

import h5py
import torch
import torchvision
import numpy as np
from allennlp.common import Params
from dataclasses import dataclass

from gpv2 import file_paths
from gpv2.image_featurizer.image_featurizer import BoxTargets, ImageRegionFeatures, ImageCollater, \
  ImageFeatureExtractor, gather_qboxes_and_targets
from gpv2.model.gpv_example import GPVExample
from gpv2.model.layers import Layer
from gpv2.utils import image_utils


def find_query_boxes(hdf5_group, query_boxes, image_id=None, extract_features=True):
  saved_boxes = hdf5_group["query_bboxes"][:]

  matches = np.abs(np.expand_dims(query_boxes.numpy(), 1) - np.expand_dims(saved_boxes, 0)) < 1e-6
  matches = np.all(matches, -1)
  q_ixs = []
  for i, box in enumerate(query_boxes):
    match_ix = np.where(matches[i])[0]
    if len(match_ix) == 0:
      if image_id is None:
        print(query_boxes)
        print(saved_boxes)
        raise ValueError(f"Unable to locate a required query box")
      else:
        print(query_boxes)
        print(saved_boxes)
        raise ValueError(f"Unable to locate a required query box for image {image_id}")
    q_ixs.append(match_ix[0])

  # TODO could avoid loading the entire array
  objectness = torch.as_tensor(hdf5_group["query_objectness"][:][q_ixs])
  if extract_features:
    return objectness, torch.as_tensor(hdf5_group["query_features"][:][q_ixs])
  else:
    return objectness


_COCO_IMAGE_ID_MAP = {}

#
# def _id_to_coco(coco_image_id):
#   if not _COCO_IMAGE_ID_MAP:
#     for part in ["val2014", "test2014", "train2014", "test2015"]:
#       for file in listdir(join(file_paths.COCO_IMAGES, part)):
#         _COCO_IMAGE_ID_MAP[image_utils.get_coco_int_id(file)] = f"coco-{part}-{file}"
#   return _COCO_IMAGE_ID_MAP[coco_image_id]
#
#
# _NUM_REGEX = re.compile("^[0-9]+(-[0-9]+)?$")
#
#
# def convert_legacy_key(hdf5_key: str) -> str:
#   """Returns the image_id for a given hdf5 key
#
#   Note this supports some legacy HDF5 keys formats that will fail if passed back to
#   `get_hdf5_key_for_image`
#   """
#   parts = hdf5_key.split("-")
#   if parts[0] in {"coco", "dce", "web"}:
#     return hdf5_key
#
#   if parts[0] in {"test", "val"} and len(parts) > 1 and parts[1] in {"nocaps", "open_images", "visual_genome"}:
#     # Legacy DCE image_id that did not have the dce prefix
#     return "dce-" + hdf5_key
#
#   if _NUM_REGEX.match(hdf5_key):
#     # Legacy COCO image_id that was just an int
#     if "-" in hdf5_key:
#       key, crop = hdf5_key.split("-")
#       return _id_to_coco(int(key)) + "-" + crop
#     else:
#       return _id_to_coco(int(hdf5_key))
#
#   raise NotImplementedError(hdf5_key)

class PrecomputedDataLoader:
  """Utility class that gathers pre-computed data and targets of a batch"""

  def __init__(self, box_sources, extract_features=False, extract_objectness=True):
    self.box_sources = box_sources
    self.extract_objectness = extract_objectness
    self.extract_features = extract_features
    self.key_to_ix = None
    if self.box_sources != "debug" and len(self.box_sources) > 1:
      self.key_to_ix = {}
      for ix, file in enumerate(self.box_sources):
        logging.info(f"Building key/file map for {file}...")
        with h5py.File(file, "r") as f:
          for key in f.keys():
            if key == "box_format":
              continue
            self.key_to_ix[key] = ix

  def __call__(self, batch: List[GPVExample], hflipped=None) -> Tuple[ImageRegionFeatures, BoxTargets]:
    query_boxes, targets = gather_qboxes_and_targets(batch, qbox_format="xyxy")
    if hflipped is None:
      hflipped = [None for _ in batch]

    if self.box_sources == "debug":
      batch_size = len(batch)
      n_boxes = 50
      boxes = torch.empty(batch_size, n_boxes, 4).uniform_(0.00001, 0.5)
      for i, q in enumerate(query_boxes):
        if q is not None:
          boxes[i, :q.shape[0]] = torch.as_tensor(q)
      return ImageRegionFeatures(
        boxes, None,
        torch.log(torch.empty(batch_size, n_boxes).uniform_(1e-6, 1.0 - 1e-6))
      ), targets

    # For backwards compat
    # TODO remove when no longer needed
    extract_objectness = getattr(self, "extract_objectness", True)

    # Additionally load the boxes
    with ExitStack() as stack:
      all_features = []
      all_bboxes = []
      all_objectness = []
      files = [stack.enter_context(h5py.File(name, "r")) for name in self.box_sources]
      for ex, is_flipped, qboxes in zip(batch, hflipped, query_boxes):
        key = image_utils.get_hdf5_key_for_image(ex.image_id, ex.crop)
        parts = key.split("/")
        if len(self.box_sources) == 1:
          grp = files[0][parts[0]]
          if len(parts) == 2:
            grp = grp[parts[1]]

        elif len(parts) == 1:
          ix = self.key_to_ix[key]
          grp = files[ix][key]

        else:
          assert len(parts) == 2
          fs = [f[parts[0]] for f in files if parts[0] in f]
          fs = [f[parts[1]] for f in fs if parts[1] in f]
          assert len(fs) == 1
          grp = fs[0]

        bboxes = torch.as_tensor(grp["bboxes"][:])

        if extract_objectness:
          objectness = torch.as_tensor(grp['objectness'][:])

        if self.extract_features:
          features = torch.as_tensor(grp['features'][:])
        if qboxes is not None:
          if isinstance(qboxes, dict):
            ixs = np.array([qboxes["ix"]])
            qboxes = qboxes["qboxes"]
            bboxes = torch.cat([bboxes, qboxes, bboxes[ixs]], 0)
            if self.extract_features:
              qobj, qfeatures = find_query_boxes(grp, qboxes, extract_features=True)
              features = torch.cat([features, qfeatures, features[ixs]], 0)
            else:
              qobj = find_query_boxes(grp, qboxes, extract_features=False)
            if extract_objectness:
              objectness = torch.cat([objectness, objectness[ixs], qobj], 0)
          elif isinstance(qboxes, int) or qboxes.shape == ():
            ixs = np.array([qboxes])
            bboxes = torch.cat([bboxes, bboxes[ixs]], 0)
            if self.extract_features:
              features = torch.cat([features, features[ixs]], 0)
            if extract_objectness:
              objectness = torch.cat([objectness, objectness[ixs]], 0)
          else:
            bboxes = torch.cat([bboxes, qboxes], 0)
            if self.extract_features:
              qobj, qfeatures = find_query_boxes(grp, qboxes, extract_features=True)
              features = torch.cat([features, qfeatures], 0)
            else:
              qobj = find_query_boxes(grp, qboxes, extract_features=False)
            if extract_objectness:
              objectness = torch.cat([objectness, qobj], 0)
        all_bboxes.append(torchvision.ops.box_convert(bboxes, "xyxy", "cxcywh"))
        if extract_objectness:
          all_objectness.append(objectness)
        if self.extract_features:
          all_features.append(features)

    for i, flip in enumerate(hflipped):
      if flip:
        # TODO flip target and all_bboxes
        raise NotImplementedError()

    all_features = all_features if self.extract_features else None
    all_objectness = all_objectness if extract_objectness else None
    regions = ImageRegionFeatures.build_from_lists(all_bboxes, all_features, all_objectness)
    return regions, targets



@dataclass
class MultiHdf5FeatureExtractorCollate2(ImageCollater):
  source: PrecomputedDataLoader

  def collate(self, batch: List[GPVExample]) -> Tuple[Dict[str, Any], List]:
    regions, targets = self.source(batch)
    return dict(regions=regions), targets


@ImageFeatureExtractor.register("hdf5-feature-extractor")
class Hdf5FeatureExtractor(ImageFeatureExtractor):
  """Loadings features from HDF5"""

  @classmethod
  def from_params(
      cls,
      params: Params,
      **extras,
  ):
    if "box_format" in params:
      box_format = params.pop("box_format")
      assert box_format is None or box_format == "xyxy"
    return super().from_params(params, **extras)

  def __init__(self, source, box_embedder: Layer=None, extract_objectness=True,
               image_id_version=1):
    super().__init__()
    self.box_embedder = box_embedder
    self.extract_objectness = extract_objectness
    self.source = source
    self.image_id_version = image_id_version
    if isinstance(source, str):
      src = image_utils.get_hdf5_files(self.source)
    else:
      src = [join(file_paths.PRECOMPUTED_FEATURES_DIR, x) for x in source]
    self.extractor = PrecomputedDataLoader(src, True, extract_objectness=extract_objectness)

  def get_collate(self, is_train=False) -> 'ImageCollater':
    return MultiHdf5FeatureExtractorCollate2(self.extractor)

  def forward(self, regions) -> ImageRegionFeatures:
    if self.box_embedder:
      box_embed = self.box_embedder(regions.boxes)
      regions.features = torch.cat([
        regions.features,
        box_embed
      ], -1)
    return regions

