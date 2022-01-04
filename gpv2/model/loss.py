import collections
from typing import List, Optional, Tuple, Dict

import torch
import torchvision
from allennlp.common import Registrable, FromParams, Params
from dataclasses import asdict, dataclass
from torch import nn

from gpv2.detr.matcher import HungarianMatcher
from gpv2.detr.set_criterion import SetCriterion
from torch.nn import functional as F


@dataclass
class BoxPredictions:
  """Predicted output boxes for a batch"""
  boxes: torch.Tensor
  rel: torch.Tensor
  n_boxes: Optional[torch.Tensor]

  def __getitem__(self, indices):
    return BoxPredictions(
      self.boxes[indices],
      None if self.rel is None else self.rel[indices],
      None if self.n_boxes is None else self.n_boxes[indices]
    )


class LocalizationLoss(nn.Module, Registrable):

  def forward(self, box_pred: BoxPredictions, targets):
    raise NotImplementedError()


@LocalizationLoss.register("box-cls-loss")
class BoxClsLoss(LocalizationLoss):
  def __init__(self, thresh, mode="cls"):
    super().__init__()
    self.thresh = thresh
    self.mode = mode

  def forward(self, box_pred: BoxPredictions, targets):
    losses = []
    for i, t in enumerate(targets):
      example_boxes = torchvision.ops.box_convert(box_pred.boxes[i], "cxcywh", "xyxy")
      target_boxes = torchvision.ops.box_convert(targets[i], "cxcywh", "xyxy")
      # [boxes, targets]
      iou = torchvision.ops.box_iou(example_boxes, target_boxes) > self.thresh
      if self.mode == "cls":
        target = 1 - torch.any(iou, 1).to(dtype=torch.int64)
        example_rel = box_pred.rel[i]
        if box_pred.n_boxes is not None:
          target = target[:box_pred.n_boxes[i]]
          example_rel = example_rel[:box_pred.n_boxes[i]]
        losses.append(F.cross_entropy(example_rel, target))
      else:
        raise NotImplementedError()
    loss = torch.stack(losses, -1).mean()
    return loss, dict(box_cls=loss)


@LocalizationLoss.register("detr-loss")
class DetrLocalizationLoss(LocalizationLoss):

  def __init__(
      self,
      cost_class: float,
      cost_bbox: float,
      cost_giou: float,
      num_classes: int,
      eos_coef: float,
      class_w: float=None, bbox_w: float=None, giou_w: float=None,
      losses=('labels', 'boxes')
  ):
    super().__init__()
    self.class_w = cost_class if class_w is None else class_w
    self.bbox_w = cost_bbox if bbox_w is None else bbox_w
    self.giou_w = cost_giou if giou_w is None else giou_w
    self.cost_class = cost_class
    self.cost_bbox = cost_bbox
    self.cost_giou = cost_giou
    self.num_classes = num_classes
    self.eos_coef = eos_coef
    self.losses = losses
    self.matcher = HungarianMatcher(
      self.cost_class, self.cost_bbox, self.cost_giou
    )
    self.set_criterion = SetCriterion(
      num_classes=num_classes,
      matcher=self.matcher,
      weight_dict=None,
      eos_coef=eos_coef,
      losses=list(losses)
    )
    self.loc_weights = {
      "loss_ce": class_w,
      "loss_bbox": bbox_w,
      "loss_giou": giou_w,
    }

  def forward(self, box_pred: BoxPredictions, targets):
    n_boxes = box_pred.n_boxes
    boxes = box_pred.boxes
    rel = box_pred.rel
    if n_boxes is not None:
      # We make the masked boxes empty psuedo-boxes with a fixed, very low score
      # TODO do we need this now the criteion knows to account for n_boxes?
      for i, n in enumerate(n_boxes):
        boxes[i, n:, :2] = 0.0
        boxes[i, n:, 2:] = 0.001   # give boxes non-zero area so they don't NaN the loss if selected
        rel[i, n:, :-1] = -1000
        rel[i, n:, -1] = 1000

    outputs = dict(
      pred_relevance_logits=rel,
      pred_boxes=boxes
    )
    if n_boxes is not None:
      outputs[n_boxes] = n_boxes

    # Build the list-of-dictionary format the matcher expects
    target_dicts = []
    for target in targets:
      target_dicts.append(dict(
        boxes=target,
        labels=torch.zeros(target.size(0), device=target.device, dtype=torch.long)
      ))

    losses = self.set_criterion(outputs, target_dicts)
    to_return = ['loss_ce', 'loss_bbox', 'loss_giou']
    out = {}
    total_loss = 0
    for k in to_return:
      if k not in losses:
        continue
      v = losses[k]
      out[k] = v
      total_loss += v * self.loc_weights[k]
    return total_loss, out


class BasicGPVLoss(FromParams, nn.Module):

  @classmethod
  def from_params(
      cls,
      params: Params,
      **kwargs
  ):
    # Backwards compatibility hacks
    if "type" in params:
      del params["type"]
    if "task_weights" in params:
      assert params["task_weights"] is None
      del params["task_weights"]
    return super().from_params(params, **kwargs)

  def __init__(
      self,
      localization: LocalizationLoss, sum_seq_tokens=False,
      source_weights: Optional[Dict[str, float]] = None
  ):
    super().__init__()
    self.localization = localization
    self.sum_seq_tokens = sum_seq_tokens
    self.source_weights = source_weights

  def forward(self, text_logits: torch.Tensor, box_predictions: BoxPredictions, labels):
    target_text = labels["text_labels"]
    target_boxes = labels["box_targets"]
    if "loss_logging" in labels:
      box_losses = None
      sources = labels["loss_logging"]
      source_to_ix = collections.defaultdict(list)
      for i, src in enumerate(sources):
        source_to_ix[src].append(i)
      total_loss = 0
      losses = {}
      for source, ixs in source_to_ix.items():
        loss, _box_losses = self._compute_loss(
          text_logits[ixs],
          None if box_predictions is None else box_predictions[ixs],
          target_text[ixs],
          [target_boxes[i] for i in ixs]
        )
        if _box_losses:
          if box_losses:
            raise NotImplementedError("Currently at most one source can have a box loss")
          box_losses = box_losses
        w = 1 if self.source_weights is None else self.source_weights.get(source, 1.0)
        total_loss += loss*w
        losses[source] = loss
        if box_losses:
          for k, v in box_losses.items():
            losses[k] = losses.get(k, 0) + v
      return total_loss, losses
    else:
      assert self.source_weights is None
      loss, box_losses = self._compute_loss(
        text_logits,
        None if box_predictions is None else box_predictions,
        target_text,
        target_boxes
      ), {}
      return loss, box_losses

  def _compute_loss(self, text_logits: torch.Tensor, box_predictions: BoxPredictions,
                    target_text, target_boxes):
    loss = 0
    if not torch.all(target_text == -100):
      if self.sum_seq_tokens:
        task_loss = F.cross_entropy(
          text_logits.view(-1, text_logits.size(-1)), target_text.view(-1), reduction='sum')
        loss = task_loss / text_logits.size(0)
      else:
        loss = F.cross_entropy(text_logits.view(-1, text_logits.size(-1)), target_text.view(-1))

    if any(x is not None for x in target_boxes):
      assert all(x is not None for x in target_boxes)
      total, log = self.localization(box_predictions, target_boxes)
      loss += total
    else:
      log = {}

    return loss, log
