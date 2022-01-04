from typing import Union, Optional, List, Callable, Any, Dict, Tuple
import numpy as np
import torch
import torchvision
from allennlp.common import Registrable
from allennlp.nn.beam_search import BeamSearch
from dataclasses import dataclass
from torch import nn

from gpv2.model.gpv_example import GPVExample


BEST_STATE_NAME = "best-state.pth"


@dataclass
class GPVExampleOutput:
    """GPV output for an example"""

    boxes: Union[torch.Tensor, np.ndarray, None]
    """[n_boxes, 4] box output in cxcywh format, normalized between 0 and 1"""

    relevance: Union[torch.Tensor, np.ndarray, None]
    """[n_boxes] Relevance score of each box, between 0 and 1"""

    text: Optional[List[str]]
    """top ranked text answers, sorted by score"""

    text_logprobs: Optional[List[float]]
    """score of each text answer"""

    def set_beams_to_keep(self, n):
        if self.text is None:
            return self
        return GPVExampleOutput(self.boxes, self.relevance, self.text[:n], self.text_logprobs[:n])


class PredictionArg(Registrable):
    """Generic super-type for arguments in GPVModel.set_prediction_args

    Have a registerable supertype allows to to use to_params/from_param to
    save those arguments in a human-readable format.
    """


@PredictionArg.register("beam-search-spec")
class BeamSearchSpec(PredictionArg):
    """Specifies how to do beam search"""

    def __init__(self, beam_size, max_seq_len, per_node_beam_size=None, sampler=None):
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size
        self.sampler = sampler
        self.max_seq_len = max_seq_len

    def build(self, end_index) -> BeamSearch:
        return BeamSearch(
            end_index, self.max_seq_len, self.beam_size,
            self.per_node_beam_size, self.sampler)


class GPVModel(nn.Module, Registrable):

    def initialize(self, load_params=True):
        """Initialize the model, used before training but not if loading a state dict

        This give the model a chance to load pre-trained parameters or do other setup that was
        not already in __init__, and does not need to be done if loading for a state_dict.

        if `load_params` is false, the model should still set up all its parameters and buffers,
        but does not need to fill them the initialized values (e.g., because it will load
        those parameters from a different distributed node).
        """
        raise NotImplementedError()

    def get_collate(self, is_train=False) -> Callable[[List[GPVExample]], Dict[str, Any]]:
        """Function that maps pre-processed examples to tensors suitable for `forward`

        The returned function might need to be distributed across multiple worker processes,
        so implementors should preferably return a light weight object rather than a method
        that is bound to `self`
        """
        raise NotImplementedError()

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the loss and any scalars to log using the outputs of `self.get_collate()(batch)`

        This is used during training.
        """
        raise NotImplementedError()

    def predict(self, *args, **kwargs) -> List[GPVExampleOutput]:
        """Computes the test-time example outputs for a batch of examples"""
        raise NotImplementedError()

    def set_prediction_args(
        self, *args: Union[str, int, float, PredictionArg],
        **kwargs: Union[str, int, float, PredictionArg]
    ):
        """Sets parameters used during prediction"""
        raise NotImplementedError()

    def preprocess_example_train(self, example) -> List[GPVExample]:
        """Convert a training example for a task into a universal/pre-processed format

        We support a one-to-many mapping for train examples
        """

        # By default, use the general method
        return [self.preprocess_example(example)]

    def preprocess_example(self, example) -> GPVExample:
        """Convert an eval example for a task into a universal/pre-processed format"""
        raise NotImplementedError()


def build_per_example_output(text, text_scores, boxes, rel, n_boxes=None, box_format="cxcywh", no_box_sort=False) -> List[GPVExampleOutput]:
    out = []
    if text_scores is not None:
        if isinstance(text_scores, torch.Tensor):
            text_scores = text_scores.cpu().numpy()

    if boxes is None:
        for txt, sc in zip(text, text_scores):
            out.append(GPVExampleOutput(None, None, txt, sc))
        return out

    if rel is None:
        n_boxes = [None]*len(boxes) if n_boxes is None else n_boxes.cpu().numpy()
        for txt, sc, box, n in zip(text, text_scores, boxes.cpu(), n_boxes):
            out.append(GPVExampleOutput(box[:n], None, txt, sc))
        return out

    if boxes.size()[:2] != rel.size():
        raise ValueError("Boxes and relevance have incompatible shapes")

    boxes = boxes.cpu()
    rel = rel.cpu().numpy()
    n_boxes = None if n_boxes is None else n_boxes.cpu().numpy()

    n = len(boxes)
    for i in range(n):
        if text is None:
            example_text, example_text_scores = None, None
        else:
            example_text, example_text_scores = text[i], text_scores[i]

        end = None if n_boxes is None else n_boxes[i]
        if no_box_sort:
            ixs = np.arange(end, dtype=np.int)
        else:
            ixs = np.argsort(rel[i, :end])
        example_boxes = torchvision.ops.box_convert(boxes[i, ixs], box_format, "cxcywh")

        out.append(GPVExampleOutput(
            example_boxes.numpy(), rel[i, ixs], example_text, example_text_scores))
    return out

