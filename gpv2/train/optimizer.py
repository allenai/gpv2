import logging
import re

import torch
from typing import Dict, Tuple, List, Optional, Any, Union

from allennlp.common import FromParams, Registrable
from dataclasses import dataclass
from torch import nn
from torch.optim import AdamW, SGD, Optimizer
from transformers import Adafactor

from gpv2.train.fixed_adamw import FixedAdamW


class OptimizerBuilder(Registrable):
  """Builds an Optimizer

  We use this class rather then using an Optimizer directly since it can be
  serialized with FromParams, and can dynamically determine how to handle groups
  of parameters depending on the model
  """

  def build(self, model, epoch_size, n_epochs) -> Optimizer:
    raise NotImplementedError()


class TrainingScheduleBuilder(Registrable):
  """Builds an learning rate schedule"""

  def build(self, optimizer, num_steps, last_epoch):
    raise NotImplementedError()


def _per_or_int_to_int(x, total):
  if isinstance(x, int):
    return x
  return round(x*total)


class DelayedWarmupSchedule:
  """Learning rate scheduler that supports triangle learning rates, and
  delaying when particular parameter groups start training

  The relevant hypter-parameter are assumed to have been set in group dictionaries
  in the optimizer to support per-group settings.
  """

  def __init__(
      self,
      optimizer,
      warmup: Union[int, float],
      total: int,
      last_epoch=-1
  ):
    self.warmup = warmup
    self.total = total
    self.optimizer = optimizer
    self._step = 0
    self.last_epoch = last_epoch
    for group in optimizer.param_groups:
      if last_epoch != -1:
        # Should have been saved in the state if we are not initializing from scratch
        assert "initial_lr" in group
        assert "step" in group
      else:
        group["initial_lr"] = group["lr"]
        group["step"] = 0
        if group.get("delay"):
          group["lr"] = 0  # Start at zero, with gradients frozen for efficiency
          for param in group["params"]:
            param.requires_grad = False

  def state_dict(self):
    return dict(step=self._step)

  def load_state_dict(self, state):
    self._step = state["step"]

  def step(self):
    self._step += 1
    for group in self.optimizer.param_groups:
      delay = group.get("delay")
      if delay:
        delay = _per_or_int_to_int(delay, self.total)
        if delay > self._step:
          if not group.get("delay_lr_schedule", True):
            # group step is frozen until delay has passed
            group["step"] += 1
          continue
        if delay == self._step:
          logging.info(f"Unfreezing parameter group {group['name']}l")
          for param in group["params"]:
            param.requires_grad = True

      group_step = group["step"] + 1
      group["step"] = group_step
      wu = group.get("warmup", self.warmup)
      wu = _per_or_int_to_int(wu, self.total)
      if group_step < wu:
        group["lr"] = group["initial_lr"] * (group_step / wu)
      else:
        if "decay" in group and group["decay"] is None:
          factor = 1.0
        else:
          factor = (self.total - group_step) / (self.total - wu)
        min_lr = group.get("min_lr", 0)
        if min_lr:
          group["lr"] = min_lr + (group["initial_lr"] - min_lr) * factor
        else:
          group["lr"] = group["initial_lr"] * factor


@dataclass
@TrainingScheduleBuilder.register("delayed-warmup-linear")
class DelayedWarmupScheduleBuilder(TrainingScheduleBuilder):
  warmup: Union[int, float] = 0

  def build(self, optimizer, num_steps, last_epoch):
    return DelayedWarmupSchedule(optimizer, self.warmup, num_steps, last_epoch)


class ParameterSet(Registrable):
  """Select a subset of parameters in a model

  Used to specify groups of parameters without having to manually list the parameter names,
  and in a way that can be easily saved as part of a config file
  """
  def get(self, model) -> List[nn.Parameter]:
    raise ValueError()


@ParameterSet.register("all")
class AllParameters(ParameterSet):
  def get(self, model) -> List[nn.Parameter]:
    return list(model.parameters())


@ParameterSet.register("image-relevance")
class ImageRelevance(ParameterSet):
  def get(self, model) -> List[nn.Parameter]:
    return list(model.image_relevance.parameters())


@dataclass
class ParameterGroup(FromParams):
  """Hyper-parameters for a specific subset of parameter"""

  parameters: ParameterSet
  group_name: str
  overrides: Dict[str, Any]
  allow_empty: bool = False
  allow_overlap: bool = False


def get_param_groups(model, groups: List[ParameterGroup]) -> List[Dict]:
  """Compute the parameter groups for a model using `groups`"""

  parameters = []
  used_params = set()
  for group in groups:
    params = group.parameters.get(model)
    if not group.allow_overlap:
      if any(p in used_params for p in params):
        raise ValueError("Parameter was assigned to multiple groups")
    else:
      params = [p for p in params if p not in  used_params]

    if len(params) == 0 and not group.allow_empty:
      raise RuntimeError(f"Group {group.group_name} empty")

    if len(params) > 0:
      used_params.update(params)
      logging.info(f"Found {len(params)} in parameter group {group.group_name}")
      param_group = dict(group.overrides)
      param_group["params"] = params
      assert "name" not in param_group
      param_group["name"] = group.group_name
      parameters.append(param_group)

  default = [p for p in model.parameters() if p not in used_params]
  if len(default) > 0:
    logging.info(f"Found {len(default)} in remaining parameters in default group")
    if "default" in parameters or "name" in parameters:
      raise ValueError()
    default_params = dict(params=default)
    default_params["name"] = "default"
    parameters.append(default_params)
  return parameters


@OptimizerBuilder.register("adam-w")
@dataclass
class AdamWBuilder(OptimizerBuilder):
  lr: float
  weight_decay: float = 0.0
  betas: Tuple[float, float] = (0.9, 0.999)
  parameter_groups: Optional[List[ParameterGroup]] = None

  def build(self, model: torch.nn.Module, epoch_size, n_epochs):
    if self.parameter_groups is None:
      parameters = model.parameters()
    else:
      parameters = get_param_groups(model, self.parameter_groups)
      for group in parameters:
        if "beta" in group:
          raise ValueError("AdamW does not support groups with different betas")
    return self.build_for(parameters)

  def build_for(self, parameters):
    if torch.__version__.startswith("1.8"):
      return FixedAdamW(
        parameters, lr=self.lr,
        weight_decay=self.weight_decay, betas=self.betas
      )
    else:
      return AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay, betas=self.betas)


@OptimizerBuilder.register("adafactor")
@dataclass
class AdafactorBuilder(OptimizerBuilder):
  lr: float
  weight_decay: float = 0.0
  beta1: Optional[float] = None
  parameter_groups: Optional[List[ParameterGroup]] = None

  def build(self, model: torch.nn.Module, epoch_size, n_epochs):
    if self.parameter_groups is None:
      parameters = model.parameters()
    else:
      parameters = get_param_groups(model, self.parameter_groups)
      for group in parameters:
        if "beta" in group:
          raise ValueError("AdamW does not support groups with different betas")
    return self.build_for(parameters)

  def build_for(self, parameters):
    return Adafactor(parameters, lr=self.lr, weight_decay=self.weight_decay, beta1=self.beta1,
                     warmup_init=False, relative_step=False, scale_parameter=False)


@OptimizerBuilder.register("sgd")
@dataclass
class SgdBuilder(OptimizerBuilder):
  lr: float
  momentum: float = 0
  weight_decay: float = 0
  dampening: float = 0
  parameter_groups: List[ParameterGroup] = None

  def build(self, model: torch.nn.Module, epoch_size, n_epochs):
    if self.parameter_groups is None:
      parameters = model.parameters()
    else:
      parameters = get_param_groups(model, self.parameter_groups)
      for group in parameters:
        if "beta" in group:
          raise ValueError("AdamW does not support groups with different betas")
    return SGD(parameters, lr=self.lr, weight_decay=self.weight_decay,
               momentum=self.momentum, dampening=self.dampening)
