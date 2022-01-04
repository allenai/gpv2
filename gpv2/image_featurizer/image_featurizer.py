import logging
from contextlib import ExitStack
from os.path import join, exists
from typing import List, Dict, Any, Tuple, Optional, NewType, Union

import logging
from typing import List, Dict, Any, Tuple, Optional

import h5py
import numpy as np
import torch
import torchvision
from allennlp.common import Registrable, Params, FromParams
from dataclasses import dataclass, replace
from torch import nn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import ColorJitter, RandomGrayscale, Normalize
from torchvision.transforms.functional import hflip, to_tensor
from torch.nn import functional as F

from gpv2 import file_paths
from gpv2.model.gpv_example import GPVExample
from gpv2.model.layers import Layer
from gpv2.utils import pytorch_utils, image_utils


@dataclass
class ImageRegionFeatures:
  """Object boxes, features, and objectness scores for objects in an image"""

  @staticmethod
  def build_from_lists(boxes, features, objectness):
    n_boxes = [len(x) for x in boxes]
    max_len = max(n_boxes)
    n_boxes = torch.as_tensor(n_boxes, dtype=torch.long, device=boxes[0].device)
    return ImageRegionFeatures(
      pytorch_utils.stack_and_pad(boxes, max_len),
      None if features is None else pytorch_utils.stack_and_pad(features, max_len),
      # -10000 so the padding is a valid log-probability
      None if objectness is None else pytorch_utils.stack_and_pad(objectness, max_len, -10000),
      n_boxes,
    )

  """[batch, n_regions, 4] boxes in [cx, cy, w, y] format normalized between 0 and 1"""
  boxes: torch.Tensor

  """[batch, n_regions, n_features] region features"""
  features: Optional[torch.Tensor]

  """[batch, n_regions] or [batch, n_regions, n_classes] objectness log-probability"""
  objectness: Optional[torch.Tensor]

  """[batch] number of boxes for each batch if batches can have differing numbers of boxes"""
  n_boxes: Optional[torch.Tensor] = None

  def numpy(self):
    return ImageRegionFeatures(
      self.boxes.cpu().numpy(),
      None if self.features is None else self.features.cpu().numpy(),
      None if self.objectness is None else self.objectness.cpu().numpy(),
      None if self.n_boxes is None else self.n_boxes.cpu().numpy()
    )

  def get_boxes(self, i: int):
    if self.boxes is not None:
      return self.boxes[i, :self.n_boxes[i]]
    else:
      return self.boxes[i]

  def to(self, device):
    return ImageRegionFeatures(
      self.boxes.to(device),
      None if self.features is None else self.features.to(device),
      None if self.objectness is None else self.objectness.to(device),
      None if self.n_boxes is None else self.n_boxes.to(device)
    )

  def get_n_boxes(self):
    if self.n_boxes is None:
      batch, n = self.boxes.size()[:2]
      return torch.full((batch,), n,
                        device=self.boxes.device, dtype=torch.long)
    else:
      return self.n_boxes


BoxTargets = NewType('BoxTargets', List[Optional[torch.Tensor]])
"""Batch of target boxes in cxcywh format, normalized between 0 and 1"""


class ImageFeatureExtractor(nn.Module, Registrable):
  """Extracts regions and region feature vectors for an image

  This class does the visual feature extraction for our models. In order to support this,
  this class provides a custom collate function to use on the images and then
  a forward method that builds the features from the output of that collate function.
  """

  def get_collate(self, is_train=False) -> 'ImageCollater':
    raise NotImplementedError()

  def forward(self, **kwargs) -> ImageRegionFeatures:
    raise NotImplementedError()


class ImageCollater:

  def collate(self, batch: List[GPVExample]) -> Dict[str, Any]:
    """
    return:
      image_inputs: Inputs to pass to `ImageFeatureExtractor.forward`
    """
    raise NotImplementedError()


class ROIFeatureExtractor(Layer):
  """Extract image features for a given set of regions"""

  def forward(self, x: torch.Tensor, boxes: torch.Tensor):
    """
    x: Tensor of images
    boxes: [batch, n, 4] boxes that NOT normalized and in xyxy format
    """
    raise NotImplementedError()


@ROIFeatureExtractor.register("box-embed-feature-extractor")
class BoBoxEmbedFeatureExtractor(ROIFeatureExtractor):
  """Does ROI pooling to get features for image regions"""

  def __init__(
      self,
      box_coordinate_embed: Optional[Layer] = None,
      pre_rio: Layer = None,
      post_rio: Layer = None,
      return_objectness = True,
      rio_processor: str = "mean",
      box_coordinate_join: str = "concat",
      rio_size=7,
  ):
    super().__init__()
    self.box_coordinate_embed = box_coordinate_embed
    self.pre_rio = pre_rio
    self.post_rio = post_rio
    self.return_objectness = return_objectness
    self.rio_processor = rio_processor
    self.box_coordinate_join = box_coordinate_join
    self.rio_size = rio_size

  def extract_roi(self, features, boxes: torch.Tensor):
    B, C, W, H = features.size()
    N = boxes.size(1)

    div = torch.as_tensor([W, H, W, H], device=boxes.device, dtype=boxes.dtype)
    scaled_boxes = boxes * div
    scaled_boxes = torchvision.ops.box_convert(scaled_boxes, "cxcywh", "xyxy")
    scaled_boxes = torch.unbind(scaled_boxes)

    roi_features = torchvision.ops.roi_align(
      features, scaled_boxes, output_size=self.rio_size, aligned=True)
    if self.rio_processor == "mean":
      roi_features = roi_features.view(B, N, C, -1).mean(-1)
    elif self.rio_processor == "max":
      roi_features = roi_features.view(B, N, C, -1).max(-1)
    else:
      raise NotImplementedError(self.rio_processor)
    return roi_features

  def forward(self, images, boxes) -> ImageRegionFeatures:
    if self.pre_rio is not None:
      images = self.pre_rio(images)

    roi_features = self.extract_roi(images, boxes)

    if self.post_rio is not None:
      roi_features = self.post_rio(roi_features)

    if self.box_coordinate_embed:
      box_embed = self.box_coordinate_embed(boxes)
      if self.box_coordinate_join == "concat":
        roi_features = torch.cat([roi_features, box_embed], -1)
      else:
        raise NotImplementedError(self.box_coordinate_join)

    return roi_features


def build_scaled_boxes(features, boxes):
  B, C, H, W = features.size()
  div = torch.as_tensor([W, H, W, H], device=boxes.device, dtype=boxes.dtype)
  boxes = boxes * div
  return torchvision.ops.box_convert(boxes, "cxcywh", "xyxy")


def gather_qboxes_and_targets(batch, hflipped=None, qbox_format="cxcywh",
                              normalize_targets=True, target_format="cxcywh"):
  """Utility method that gather the query_boxes and targets of a batch"""
  targets = []
  query_boxes = []
  if hflipped is None:
    hflipped = [None for _ in batch]
  for ex, is_flipped in zip(batch, hflipped):
    if ex.target_boxes is None:
      targets.append(None)
    else:
      if ex.crop:
        raise ValueError("Box target on cropped images no supoorted")
      # Normalize the target boxes to be between 0 and 1 and to be
      # cxcywh format
      # TODO it would be nice to do this in the pre-preprocesing step
      boxes = torch.tensor(ex.target_boxes, dtype=torch.float)
      boxes = torchvision.ops.box_convert(boxes, "xywh", target_format)
      if normalize_targets:
        boxes = image_utils.normalize_boxes(boxes, ex.image_id)
      if is_flipped:
        boxes[:, 0] = 1.0 - boxes[:, 0]
      targets.append(boxes)

    if ex.query_boxes is None:
      query_boxes.append(None)
    else:
      if isinstance(ex.query_boxes, dict):
        qboxes = ex.query_boxes["qboxes"]
        qboxes = torch.tensor(qboxes, dtype=torch.float)
        if torch.any(qboxes > 1.0):
          qboxes = image_utils.normalize_boxes(qboxes, ex.image_id)
        qboxes = torchvision.ops.box_convert(qboxes, "xywh", qbox_format)
        if is_flipped:
          qboxes[:, 0] = 1.0 - qboxes[:, 0]
        q = dict(ex.query_boxes)
        q["qboxes"] = qboxes
        query_boxes.append(q)
      elif isinstance(ex.query_boxes, int) or ex.query_boxes.shape == ():
        # Query box reference to a stored bounding box
        query_boxes.append(ex.query_boxes)
      else:
        # Convert query boxes
        qboxes = torch.tensor(ex.query_boxes, dtype=torch.float)
        if torch.any(qboxes > 1.0):
          qboxes = image_utils.normalize_boxes(qboxes, ex.image_id)
        qboxes = torchvision.ops.box_convert(qboxes, "xywh", qbox_format)
        if is_flipped:
          qboxes[:, 0] = 1.0 - qboxes[:, 0]
        query_boxes.append(qboxes)
  return query_boxes, targets


@ImageFeatureExtractor.register("debug")
class DebugFeaturizer(ImageFeatureExtractor, ImageCollater):

  def __init__(self, n_boxes=4, dim=32):
    super().__init__()
    self.n_boxes = n_boxes
    self.dim = dim

  def get_collate(self, is_train=False) -> 'ImageCollater':
    return self

  def collate(self, batch):
    _, targets = gather_qboxes_and_targets(batch)
    return dict(batch_size=torch.as_tensor(len(batch))), targets

  def forward(self, batch_size) -> ImageRegionFeatures:
    device = batch_size.device
    return ImageRegionFeatures(
      torch.empty(batch_size, self.n_boxes, 4, device=device).uniform_(0.00001, 0.5),
      torch.empty(batch_size, self.n_boxes, self.dim, device=device).uniform_(0.00001, 0.5),
      torch.log(torch.empty(batch_size, self.n_boxes, self.dim, device=device).uniform_(1e-6, 1.0 - 1e-6))
    )

