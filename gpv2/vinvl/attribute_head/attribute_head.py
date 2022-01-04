# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import os
import torch

from .roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from .roi_attribute_predictors import make_roi_attribute_predictor
from .inference import make_roi_attribute_post_processor
from .loss import make_roi_attribute_loss_evaluator
from ..structures.bounding_box import BoxList


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero(as_tuple=False).squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIAttributeHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIAttributeHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_attribute_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_attribute_post_processor(cfg)
        self.loss_evaluator = make_roi_attribute_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from box_head
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the attribute feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `attribute` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)

        # deal with the case when len(proposals)==0 in the inference time;
        # we assume that this will not happen in training
        num_dets = [len(box) for box in proposals]
        if not self.training and sum(num_dets)==0:
            return features, proposals, {}

        # normal process when there is at least one detected positive box in this batch
        if self.training and self.cfg.MODEL.ROI_ATTRIBUTE_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)

        labels = torch.cat(
                [boxes_per_image.get_field("labels").view(-1) for boxes_per_image in proposals], dim=0)
        attribute_logits, attribute_features = self.predictor(x, labels)

        if not self.training:
            result = self.post_processor(attribute_logits, proposals, attribute_features)
            return x, result, {}

        loss_attribute = self.loss_evaluator(proposals, attribute_logits, targets)

        return x, all_proposals, dict(loss_attribute=loss_attribute)


def build_roi_attribute_head(cfg, in_channels):
    return ROIAttributeHead(cfg, in_channels)
