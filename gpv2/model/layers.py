import torch
from allennlp.common import Registrable, Params

from torch import nn
from torch.nn import functional as F
from transformers.models.t5.modeling_t5 import T5LayerNorm

from gpv2.utils import pytorch_utils
from gpv2.utils.to_params import to_params


class Layer(Registrable, nn.Module):
  """Generic class for modules that we need serialize with FromParams"""
  pass


@Layer.register("linear")
class Linear(nn.Linear, Layer):
  def to_params(self):
    return dict(
      in_features=self.in_features,
      out_features=self.out_features,
      bias=self.bias is not None,
    )


@Layer.register("relu")
class Relu(nn.ReLU, Layer):
  pass


@Layer.register("null")
class NullLayer(Layer):
  def forward(self, x):
    return x


@Layer.register("sequential")
class Sequence(nn.Sequential, Layer):

  @classmethod
  def from_params(
        cls,
        params: Params,
        constructor_to_call = None,
        constructor_to_inspect = None,
        **extras,
  ):
    return Sequence(*[Layer.from_params(x) for x in params["args"]])

  def to_params(self):
    return dict(args=[to_params(x, Layer) for x in self])


@Layer.register("linear-objectness")
class LinearObjectness(Layer):

  def __init__(self, n_in):
    super().__init__()
    self.n_in = n_in
    self.norm = T5LayerNorm(self.n_in)
    self.rel = nn.Linear(self.n_in, 1)

  def forward(self, encoder, objectness, boxes):
    n_images = boxes.size(1)
    image_rel = self.rel(self.norm(encoder[:, :n_images]))
    return image_rel.squeeze(-1)


@Layer.register("sum-with-objectness-v3")
class SumWithObjectness(Layer):

  def __init__(self, n_in, objectness_factor=False, multi_class_mode="any-object"):
    super().__init__()
    self.n_in = n_in
    self.multi_class_mode = multi_class_mode
    self.objectness_factor = objectness_factor
    self.norm = T5LayerNorm(self.n_in)
    self.rel = nn.Linear(self.n_in, 1 + objectness_factor)
    # Initialize so it just uses objectness at the start
    self.rel.weight.data[:] = 0
    if objectness_factor:
      self.rel.bias.data[1] = 1.0

  def forward(self, encoder, objectness, boxes):
    n_images = boxes.size(1)
    image_rel = self.rel(self.norm(encoder[:, :n_images]))
    if self.objectness_factor:
      image_rel, factor = torch.split(image_rel, [1, 1], -1)
      factor = factor.squeeze(-1)
    else:
      factor = None
    image_rel = image_rel.squeeze(-1)

    if len(objectness.size()) == 3:
      if self.multi_class_mode == "any-object":
        non_object_lp = F.log_softmax(objectness, -1)[:, :, -1]
        object_lp = torch.log1p(-torch.exp(non_object_lp))
      elif self.multi_class_mode == "max-object":
        object_lp = torch.max(F.log_softmax(objectness, -1)[:, :, :-1], 2)[0]
        non_object_lp = torch.log1p(-torch.exp(object_lp))
      else:
        raise NotImplementedError()
      objectness = object_lp - non_object_lp
    else:
      # Note we need eps=-1e-6 to stop NaN occurring if the objectness score is too close to log(1) = 0
      # This has occured in (very) rare cases for the VinVL model, in particular for images
      # 8cdae499db22a787a5274d4ee2255315964e1144ab3f95665144c90e24d79917
      # 597caa946a207ef96ede01a53321ff4fdf000a48707ea7c495627330b3ee4b90
      objectness = pytorch_utils.convert_logprob_to_sigmoid_logit(objectness, -1e-6)

    if factor is not None:
      objectness = objectness * factor

    return image_rel + objectness


@Layer.register("basic-box-embedder")
class BasicBoxEmbedder(Layer):
  def forward(self, boxes: torch.Tensor):
    cx, cy, w, h = [x.squeeze(-1) for x in boxes.split([1, 1, 1, 1], -1)]
    return torch.stack([cx, cy, w, h, w*h], -1)


@Layer.register("layer-norm")
class LayerNorm(Layer):
  def __init__(self, eps=1e-5):
    super().__init__()
    self.eps = eps

  def forward(self, x):
    return F.layer_norm(x, (x.size(-1),), eps=self.eps)
