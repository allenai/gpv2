import logging
from typing import List, Union, Optional

from torch.utils.data import Dataset as TorchDataset, Sampler

import numpy as np

from gpv2.utils import py_utils
from gpv2.utils.py_utils import get_batch_bounds


class StratifiedSubsetSampler(Sampler):
  """Sampler that supports stratifying different groups, distributed training, and
  showing subsets of each group each epoch in cyclic manner, so that the overlap between
  the subsets shown during multiple epochs is minimal. Also supports being used as
  a batch samlpler through the `batch_size` parameter, which needs to be set in the
  distributed setting to ensure each distributed sampler is the same length

  Requires `set_epoch` to be called before each epoch, always returns shuffled ids.
  """

  def __init__(
      self,
      group_sizes: List[int],
      seed: Optional[int],
      stratify: bool,
      samples_per_epoch: List[Union[None, int, float]]=None,
      batch_size: Optional[int]=None,
      rank=None,
      world_size=None
  ):
    if world_size is not None and batch_size is None:
      logging.warning("Setting world size without batch size can results in"
                      " distributed samplers with different lengths")
    self.group_sizes = group_sizes
    if any(x == 0 for x in self.group_sizes):
      raise ValueError()
    if samples_per_epoch is None:
      samples_per_epoch = [None for _ in group_sizes]
    self.samples_per_epoch = samples_per_epoch
    self.seed = seed
    self.stratify = stratify
    self.rank = rank
    self.world_size = world_size

    self._epoch_data = []
    self._size_per_epoch = []
    self._group_offests = np.cumsum(group_sizes)
    for i, val in enumerate(samples_per_epoch):
      if val is None:
        self._size_per_epoch.append(group_sizes[i])
      elif isinstance(val, float):
        assert 0 < val <= 1.0
        n = int(round(group_sizes[i]*val))
        logging.info(f"Subsampling {n}/{group_sizes[i]} for group {i}")
        self._size_per_epoch.append(n)
      else:
        assert val <= group_sizes[i]
        logging.info(f"Subsampling {val}/{group_sizes[i]} for group {i}")
        self._size_per_epoch.append(val)

    proportions = np.array(self._size_per_epoch)
    proportions = proportions/sum(proportions)
    proportions_str = ", ".join(("%.3f" % x) for x in proportions)
    logging.info(f"Epoch proportions: {proportions_str}")
    total_examples = sum(self._size_per_epoch)
    if world_size is not None:
      self.bounds = get_batch_bounds(total_examples, self.world_size)
    else:
      self.bounds = None

    self.n = sum(self._size_per_epoch)

    self.batch_size = batch_size
    if self.batch_size is not None:
      if world_size is None:
        self.n_batches = (self.n + batch_size - 1) // batch_size
      else:
        # Ensure each sampler has the same number of batches
        max_n = max(e - s for s, e in self.bounds)
        self.n_batches = (max_n + batch_size - 1) // batch_size
    else:
      self.n_batches = None

  def _get_seed(self, group_cycle, ix):
    return self.seed + group_cycle*13 + 2039 + ix*117

  def set_epoch(self, epoch):
    # First decide which indices to include in this epoch
    all_data = []
    for i, group_sz in enumerate(self.group_sizes):
      offset = 0 if i == 0 else self._group_offests[i-1]
      sz = self._size_per_epoch[i]
      start = sz * epoch % group_sz
      end = start + sz
      group_cycle = ((epoch + 1)*sz - 1) // group_sz
      group_rng = np.random.RandomState(self._get_seed(group_cycle, i))
      indices = group_rng.permutation(group_sz)

      if end <= group_sz:
        # No wrap around
        all_data.append(indices[start:end] + offset)
      else:
        # Wraps around, get the remaining data from the previous cycle
        all_data.append(indices[:(end % group_sz)] + offset)
        group_rng = np.random.RandomState(self._get_seed(group_cycle-1, i))
        indices = group_rng.permutation(group_sz)
        all_data.append(indices[start:] + offset)

    # Then gather all the indices for this epoch into `_epoch_data`
    shuffle_rng = np.random.RandomState(self.seed + 5417)
    if self.stratify:
      # Merge groups in a stratified way
      # I am not sure if we really need to shuffle the groups again, but it might needed
      # be due to how the groups are cycled and more shuffling never hurt.
      for grp in all_data:
        shuffle_rng.shuffle(grp)
      self._epoch_data = py_utils.balanced_merge_multi(all_data)
    else:
      # Merge and shuffle
      self._epoch_data = py_utils.flatten_list(all_data)
      shuffle_rng.shuffle(self._epoch_data)

  def __iter__(self):
    # Data to show
    if self.world_size is not None:
      s, e = self.bounds[self.rank]
      data = self._epoch_data[s:e]
    else:
      data = self._epoch_data

    if self.batch_size is None:
      for x in data:
        yield x
    else:
      # Group into n_batches batches
      bounds = get_batch_bounds(len(data), self.n_batches)
      for s, e in bounds:
        yield data[s:e]

  def __len__(self):
    if self.n_batches is not None:
      return self.n_batches
    elif self.world_size is None:
      return self.n
    else:
      s, e = self.bounds[self.rank]
      return e - s

