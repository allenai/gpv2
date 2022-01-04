import base64
import io
import json
import logging
from os.path import join, exists
from typing import List, Tuple, Dict, Any

import cv2
import h5py
import torch
import torchvision.ops
from PIL import Image
from attr import dataclass
import torchvision.transforms as T

import numpy as np

from gpv2 import file_paths
from gpv2.image_featurizer.image_featurizer import ImageCollater, ImageRegionFeatures, \
  ImageFeatureExtractor, gather_qboxes_and_targets
from gpv2.model.gpv_example import GPVExample
from gpv2.utils import pytorch_utils, image_utils, py_utils
from gpv2.vinvl.get_vinvl import get_vinvl
from gpv2.vinvl.structures.bounding_box import BoxList
from gpv2.vinvl.structures.image_list import to_image_list


class VinVLTSVReader:
  """Knows how to reading VinVL's precomputed feature TSV format"""

  def __init__(self, src):
    logging.info("Computing vinvl image-id-to-offsets")
    feature_lineidx = []
    with open(join(src, "features.lineidx")) as f:
      for line in f:
        feature_lineidx.append(int(line))

    pred_lineidx = []
    with open(join(src, "predictions.lineidx")) as f:
      for line in f:
        pred_lineidx.append(int(line))

    with open(join(src, "imageid2idx.json")) as f:
      image_id_to_idx = json.load(f)

    image_to_offset = {}
    for image_id, idx in image_id_to_idx.items():
      image_to_offset[int(image_id)] = feature_lineidx[idx], pred_lineidx[idx]
    self.image_to_offset = image_to_offset
    self.feature_file = join(src, "features.tsv")
    self.prediction_file = join(src, "predictions.tsv")

  def get(self, image_ids, return_features=True) -> List[Dict[str, Any]]:
    preds = []
    with open(self.feature_file, "r+b") as feature_f, open(self.prediction_file, "r") as pred_f:
      for image_id in image_ids:
        feature_off, pred_off = self.image_to_offset[image_id]

        pred_f.seek(pred_off)
        pred_image_id, pred = pred_f.readline().split("\t")
        assert pred_image_id == str(image_id)
        pred = json.loads(pred)

        if return_features:
          feature_f.seek(feature_off)
          parts = feature_f.readline().split(b"\t")
          assert str(image_id) == str(int(parts[0]))
          n_boxes = int(parts[1])
          ex_features = np.frombuffer(
            base64.decodebytes(parts[2]),
            dtype=np.float32).reshape((n_boxes, -1))
          # Copy to avoid trigger in annoying warning when using `torch.as_tensor` on a read-only
          # numpy array
          pred["features"] = ex_features.copy()

        preds.append(pred)
      return preds


@dataclass
class VinVLPrecomputedFeaturesCollate(ImageCollater):
  """Loads VinVL features and convert them to `ImageRegionFeatures`"""
  reader: VinVLTSVReader

  def collate(self, batch: List[GPVExample]) -> Tuple[Dict[str, Any], List]:
    boxes = []
    image_sizes = []
    conf = []

    if any(x.crop is not None for x in batch) or any(x.query_boxes is not None for x in batch):
      raise NotImplementedError()

    preds = self.reader.get([x.image_id for x in batch])

    for pred in preds:
      w, h = pred["image_w"], pred["image_h"]
      image_sizes.append((h, w))
      image_bboxes = np.array([x["rect"] for x in pred["objects"]], dtype=np.float32)
      image_conf = torch.log(torch.as_tensor([x["conf"] for x in pred["objects"]], dtype=torch.float32))
      conf.append(image_conf)
      image_bboxes = numpy_xyxy_to_cxcywh(image_bboxes, w, h)
      boxes.append(torch.as_tensor(image_bboxes))

    n_boxes = torch.as_tensor([len(x) for x in boxes])
    box_len = n_boxes.max()

    fe = ImageRegionFeatures(
      pytorch_utils.stack_and_pad(boxes, box_len),
      pytorch_utils.stack_and_pad([x["features"] for x in preds], box_len),
      # -1000 so the padding is a valid value in log-probability space
      pytorch_utils.stack_and_pad(conf, box_len, pad=-1000),
      n_boxes=n_boxes
    )
    box_targets = get_box_targets(batch, image_sizes, "cxcywh")
    return dict(features=fe), box_targets


@ImageFeatureExtractor.register("vinvl-precomputed")
class VinVLPrecomputedFeatures(ImageFeatureExtractor):
  """Returns pre-computed VinVL features"""

  def __init__(self, model="release", dataset="coco2014trainval"):
    super().__init__()
    self._collater = None
    self.model = model
    self.dataset = dataset

  def get_collate(self, is_train=False) -> 'ImageCollater':
    if self._collater is None:
      src = join(file_paths.VINVL_SOURCE, self.model, self.dataset)
      self._collater = VinVLTSVReader(src)
    return VinVLPrecomputedFeaturesCollate(self._collater)

  def forward(self, features) -> ImageRegionFeatures:
    return features


@dataclass
class VinvlCollate(ImageCollater):
  """Collates images in way that can be passed into a VinVL model"""
  transform: Any
  is_train: bool
  read_image_mode: str = "vinvl"

  def collate(self, batch: List[GPVExample]) -> Tuple[Dict[str, Any], List]:
    image_tensors = []
    for example in batch:
      # Different image reading methods return subtly different outputs
      # for the same image, so we mimic the one used in vimvl (cv2.imread) by default
      if self.read_image_mode == "load_image_data":
        img, size = image_utils.load_image_data(example, None)
        img = F.to_pil_image(img)
      elif self.read_image_mode == "pil":
        img = image_utils.load_image_pil(example.image_id, example.crop)
      elif self.read_image_mode == "vinvl":
        if hasattr(example, "image_file"):
          image_f = example.image_file
        else:
          image_f = image_utils.get_image_file(example.image_id)
        if not exists(image_f):
          raise ValueError(f"Missing image file {image_f}/{example.image_id}")
        try:
          # VinVL encodes and then decodes the image due to its pre-processing setup, I have seen
          # the encoding/decoding procedure slightly alter the image (maybe due to the jpg encoding?)
          # so we do it here to 100% match with image format VinVL is trained on
          tmp = cv2.imread(image_f)
          img = Image.open(io.BytesIO(cv2.imencode('.jpg', tmp)[1])).convert('RGB')
          img = image_utils.crop_img(img, example.crop)
        except cv2.error:
          # This load method fails for some formats (i.e., GIFs) due to limited support
          # of cv2.imread, we fall back to the more general load_image_data
          img, size = image_utils.load_image_data(example, None)
          img = F.to_pil_image(img).convert("RGB")
      else:
        raise NotImplementedError()
      image_tensors.append(img)

    qboxes, targets = gather_qboxes_and_targets(
      batch, qbox_format="xyxy", normalize_targets=False, target_format="xyxy")
    for ix, t in enumerate(targets):
      if t is not None:
        h, w = image_tensors[ix].size[-2:]
        targets[ix] = BoxList(t, (w, h), "xyxy")

    out = [self.transform(img, target) for img, target in zip(image_tensors, targets)]
    image_tensors, targets = py_utils.transpose_lists(out)

    for ix, t in enumerate(targets):
      if t is not None:
        bbox = torchvision.ops.box_convert(t.resize((1.0, 1.0)).bbox, "xyxy", "cxcywh")
        targets[ix] = bbox

    # VinVL expects boxes to be relative to the image tensors
    for qbox, image in zip(qboxes, image_tensors):
      tensor_h, tensor_w = image.size()[-2:]
      size_f = torch.tensor([tensor_w, tensor_h, tensor_w, tensor_h])
      qbox *= size_f.unsqueeze(0)

    images = to_image_list(image_tensors)
    return dict(images=images, query_boxes=qboxes), targets


class VinvlImageFeaturizer(ImageFeatureExtractor):
  """Builds by features by running a VinVL model end-to-end

  Note I am currently not sure if a loss can backprop through its outputs effectively or not
  """

  def __init__(self, model="release", box_embedder=None):
    super().__init__()
    self.box_embedder = box_embedder
    self.model = model
    self.vinvl, transforms = get_vinvl(model)
    self.transforms = transforms

  def get_collate(self, is_train=False) -> 'ImageCollater':
    return VinvlCollate(self.transforms, is_train)

  def forward(self, images, query_boxes=None) -> ImageRegionFeatures:
    device = images.tensors.device

    with torch.no_grad():
      out, backbone_features = self.vinvl(images, None, True)

    if query_boxes is not None and any(x is not None for x in query_boxes):
      # Build BoxLists for the extra boxes we want features for
      extra_boxes = []
      for batch_ix in range(len(images.tensors)):
        extra_boxes.append(BoxList(query_boxes[batch_ix], images.image_sizes[batch_ix]))

      # Run those through the feature extractor/classification pipeline manually
      # We don't try to get these in the main `self.vinvl` call to ensure getting
      # this doesn't mess with box selection in the main method
      box_head = self.vinvl.roi_heads['box']
      query_features = box_head.feature_extractor(backbone_features, extra_boxes)
      extra_class_logits, _ = box_head.predictor(query_features)
      extra_class_logprobs = torch.log_softmax(extra_class_logits, -1)
      query_features = box_head.post_processor.avgpool(query_features).squeeze(-1).squeeze(-1)
      n_query_boxes = [len(x.bbox) for x in extra_boxes]
      query_objectness = torch.split(torch.max(extra_class_logprobs[:, 1:], -1)[0], n_query_boxes)
      query_features = torch.split(query_features, n_query_boxes)
    else:
      extra_boxes = None
      query_objectness = None
      query_features = None

    all_boxes = []
    all_features = []
    conf = []
    for batch_ix, bbox in enumerate(out):
      boxes = bbox.bbox
      features = bbox.get_field("box_features")
      scores = torch.log(bbox.get_field("scores"))
      w, h = bbox.size
      scale = torch.as_tensor([w, h, w, h], dtype=bbox.bbox.dtype, device=device).unsqueeze(0)
      boxes = boxes/scale

      if query_features is not None:
        # Append the query boxes/features/scores
        q_bbox = extra_boxes[batch_ix].bbox / scale

        # Some of the boxes can exceed 1 due to float point issues, so we clip them here
        q_bbox[:, 2] = torch.clip(q_bbox[:, 2], 0, 1.0)
        q_bbox[:, 3] = torch.clip(q_bbox[:, 3], 0, 1.0)
        boxes = torch.cat([boxes, q_bbox], 0)
        features = torch.cat([features, query_features[batch_ix]], 0)
        scores = torch.cat([scores, query_objectness[batch_ix]], 0)

      boxes = torchvision.ops.box_convert(boxes, bbox.mode, "cxcywh")
      all_boxes.append(boxes)
      all_features.append(features)
      conf.append(scores)

    regions = ImageRegionFeatures.build_from_lists(all_boxes, all_features, conf)

    if self.box_embedder:
      box_embed = self.box_embedder(regions.boxes)
      regions.features = torch.cat([
        regions.features,
        box_embed
      ], -1)

    return regions
