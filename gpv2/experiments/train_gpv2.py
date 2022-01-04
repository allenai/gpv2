import json
import logging
import os
from argparse import ArgumentParser
from copy import deepcopy

import torch.utils.data
from transformers import AutoConfig

from gpv2.data.dataset import Task
from gpv2.experiments.trainer_cli import add_train_args, get_trainer_from_args, \
  run_trainer_from_args
from gpv2.image_featurizer.precomputed_features import Hdf5FeatureExtractor
from gpv2.model.gpv2 import T5GpvPerBox
from gpv2.model.layers import Linear, BasicBoxEmbedder
from gpv2.model.loss import DetrLocalizationLoss, LocalizationLoss, BasicGPVLoss
from gpv2.train.optimizer import DelayedWarmupScheduleBuilder, AdamWBuilder, OptimizerBuilder, \
  ParameterGroup, AllParameters
from gpv2.train.trainer import Trainer
from gpv2.utils import py_utils
from gpv2.utils.to_params import to_params

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
  parser = ArgumentParser()
  parser.add_argument("--model", choices=["t5-small", "t5-base", "t5-large"], default=None)
  parser.add_argument("--init_from")
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight_decay", type=float, default=1e-4)

  add_train_args(
    parser, tasks=list(Task), epochs=8,
    clip_grad_norm=None, num_workers=4, batch_size=60)
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  if args.model is None:
    if args.debug in ["tiny", "small"] and args.init_from is None:
      # Hack the default to be t5-small if not manually specified
      args.model = "t5-small"
    else:
      args.model = "t5-base"

  conf = AutoConfig.from_pretrained(args.model)
  t5_dim = conf.d_model

  localization_loss = DetrLocalizationLoss(1, 5, 2, 1, 0.5, 1, 5, 2, ['labels'])

  # Current best
  model = T5GpvPerBox(
    args.model,
    loss=BasicGPVLoss(localization_loss),
    image_feature_extractor=Hdf5FeatureExtractor("vinvl", BasicBoxEmbedder()),
    image_joiner=Linear(2048+5, t5_dim),
    all_lower_case=True,
    initialize_from=args.init_from,
    contrast_query="other",
    convert_to_relevance="raw",
    combine_with_objectness="multiply",
    embed_objectness_score=False,
  )

  groups = [ParameterGroup(
    AllParameters(),
    group_name="all",
    overrides=dict(delay=0.0, warmup=0.1, lr=args.lr),
    allow_overlap=True
  )]

  scheduler = DelayedWarmupScheduleBuilder()
  optimizer = AdamWBuilder(
    lr=args.lr,
    weight_decay=args.weight_decay,
    parameter_groups=groups
  )

  print("Optimizer:")
  print(json.dumps(to_params(optimizer, OptimizerBuilder), indent=2))

  trainer: Trainer = get_trainer_from_args(
    args, logging_ema=0.995,
    optimizer=optimizer, scheduler=scheduler
  )

  run_trainer_from_args(trainer, model, args)


if __name__ == '__main__':
  main()
