from typing import Optional

import torch
import numpy as np
from torch.utils.data import Sampler

from gpv2.utils.py_utils import get_batch_bounds


class SubsetSampler(Sampler):

  def __init__(self, n: int, n_examples: int):
    if not isinstance(n, int) or not isinstance(n_examples, int):
      raise ValueError("args should be integers")
    if n_examples > n:
      raise ValueError(f"Requested {n_examples} samples, but only {n} examples.")
    self.n = n
    self.n_examples = n_examples

  def __iter__(self):
    ixs = np.random.choice(self.n, self.n_examples, replace=False)
    return iter(ixs)

  def __len__(self):
    return self.n_examples


class DistributedSubsetSampler(Sampler):
  def __init__(self, n: int, n_examples: Optional[int], rank: int,
               world_size: int, seed=0):
    if not isinstance(n, int):
      raise ValueError("args should be integers")
    if n_examples is not None and n_examples > n:
      raise ValueError(f"Requested {n_examples} examples, but only have {n} total")
    self.n = n
    self.rank = rank
    self.world_size = world_size
    self.n_examples = n_examples
    self.seed = seed
    self.bound = get_batch_bounds(n if n_examples is None else n_examples, self.world_size)

  def set_seed(self, seed):
    self.seed = seed

  def __iter__(self):
    g = torch.Generator()
    g.manual_seed(self.seed)
    indices = torch.randperm(self.n, generator=g)
    s, e = self.bound[self.rank]
    subset = indices[s:e].tolist()
    return iter(subset)

  def __len__(self):
    s, e = self.bound[self.rank]
    return e - s
