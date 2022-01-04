import logging
from queue import Empty
from typing import List, Union

import torch
from torch.nn import functional as F
from torch.utils.data import IterableDataset


def stack_and_pad(tensors: List, max_len=None, pad=0.0, build_mask=False):
  tensors = [torch.as_tensor(x) for x in tensors]
  if max_len is None:
    max_len = max(x.size(0) for x in tensors)
  t0 = tensors[0]
  if len(t0.size()) == 2:
    out = torch.full((len(tensors), max_len, t0.size(1)), pad, dtype=t0.dtype, device=t0.device)
  else:
    out = torch.full((len(tensors), max_len), pad, dtype=t0.dtype, device=t0.device)
  for i, t in enumerate(tensors):
    out[i, :t.size(0)] = t

  if build_mask:
    mask = torch.zeros((len(tensors), max_len), dtype=torch.bool, device=t0.device)
    for i, t in enumerate(tensors):
      mask[i, :t.size(0)] = True
    return out, mask
  else:
    return out


def seq_len_to_binary_mask(seq_len, max_len=None):
  if max_len is None:
    max_len = seq_len.max()
  return seq_len.unsqueeze(1) > torch.arange(0, max_len, device=seq_len.device).unsqueeze(0)


def concat_masked_sequences(
    seq1, mask1,
    seq2, mask2
):
  batch = seq1.size(0)
  if mask1 is None and mask2 is None:
    return torch.cat([seq1, seq2], 1), None
  if mask1 is None:
    if len(mask2.size()) == 1:
      raise NotImplementedError("Sequence length masks2")
    out = torch.cat([seq1, seq2], 1)
    mask = torch.cat([
      torch.ones(batch, seq1.size(1), device=seq1.device, dtype=mask2.dtype),
      mask2
    ], 1)
    return out, mask
  elif mask2 is None:
    seq2_len = seq2.size(1)

    if len(mask1.size()) == 2:
      assert mask1.dtype == torch.bool or torch.all(torch.logical_or(mask1 == 0, mask1 == 1))
      seq_len1 = mask1.int().sum(1)
    else:
      assert mask1.dtype == torch.long and len(mask1.size()) == 1
      seq_len1 = mask1

    out = F.pad(seq1, [0, 0, 0, seq2_len, 0, 0])
    for i in range(batch):
      out[i, seq_len1[i]:seq_len1[i]+seq2_len] = seq2[i]
    return out, seq_len_to_binary_mask(seq_len1 + seq2_len)
  else:
    # both mask are not none
    if len(mask1.size()) != 1:
      raise NotImplementedError("Binary mask1")
    else:
      seq_len1 = mask1

    if len(mask2.size()) == 2:
      assert mask2.dtype == torch.bool or torch.all(torch.logical_or(mask2 == 0, mask2 == 1))
      seq_len2 = mask2.int().sum(1)
    else:
      seq_len2 = mask2

    out_len = (seq_len1 + seq_len2).max()
    to_pad = out_len - seq1.size(1)
    out = F.pad(seq1, [0, 0, 0, to_pad, 0, 0])
    for i in range(batch):
      out[i, seq_len1[i]:seq_len1[i]+seq_len2[i]] = seq2[i, :seq_len2[i]]
    return out, seq_len_to_binary_mask(seq_len1 + seq_len2)


def convert_logprob_to_sigmoid_logit(logprob, eps=-1e-6):
  assert torch.all(logprob <= 0.0)
  objectness = torch.minimum(
    logprob, torch.as_tensor([eps], device=logprob.device, dtype=logprob.dtype))
  object_lp = objectness
  non_object_lp = torch.log1p(-torch.exp(object_lp))
  return object_lp - non_object_lp


def log_prob_to_logits(logprob, eps=-1e-7):
  # Note eps needs to be large enough so torch.exp(eps) != 1.0, 1e-8 is too large
  assert torch.all(logprob <= 0.0)
  logprob = torch.minimum(
    logprob, torch.as_tensor([eps], device=logprob.device, dtype=logprob.dtype))
  inv_logprob = torch.log1p(-torch.exp(logprob))
  if not torch.all(torch.isfinite(inv_logprob)):
    raise ValueError()
  return torch.stack([logprob, inv_logprob], -1)


def stack_and_pad_blocks(tensors: List, max_len=None, pad=0.0):
  if max_len is None:
    max_len = max(x.size(1) for x in tensors)
  total_n = sum(x.size(0) for x in tensors)

  t0 = tensors[0]
  out = torch.full((total_n, max_len, t0.size(2)), pad, dtype=t0.dtype, device=t0.device)
  mask = torch.zeros((total_n, max_len,), dtype=torch.bool, device=t0.device)

  on = 0
  for i, t in enumerate(tensors):
    out[on:on+t.size(0), :t.size(1)] = t
    mask[on:on+t.size(0), :t.size(1)] = True
    on = on + t.size(0)
  assert on == total_n
  return out, mask


class QueueDataset(IterableDataset):
  def __init__(self, q):
    """
    q: Queue with all elements we want to yield already queue up
    """
    self.q = q

  def __iter__(self):
    while True:
      try:
        item = self.q.get(block=False)
        if item is None:
          return  # Allow None to also signal the end of the dataset
        yield item
      except Empty:
        # Even for non-blocking calls, it looks like empty can be raised even if the queue
        # had elements in it (due to locking issues?) double check here
        if self.q.empty():
          return


def get_device(device_name: Union[None, str, int]=None):
  if device_name is None:
    if torch.cuda.is_available():
      logging.info("cuda found, defaulting to cuda")
      return torch.device('cuda')
    else:
      logging.info("cuda not found, using cpu")
      return torch.device('cpu')
  else:
    try:
      device_name = int(device_name)
    except ValueError:
      pass
    return torch.device(device_name)


def get_devices(devices):
  if isinstance(devices, list) and len(devices) > 1:
    out = []
    for x in devices:
      try:
        out.append(int(x))
      except ValueError:
        out.append(x)
    return out

  if isinstance(devices, list):
    devices = devices[0]

  if devices is not None:
    try:
      return int(devices)
    except ValueError:
      return devices
  else:
    if torch.cuda.is_available():
      logging.info("cuda found, defaulting to cuda")
      return 'cuda'
    else:
      logging.info("cuda not found, using cpu")
      return 'cpu'


def to_device(batch, device):
  if batch is None:
    return None
  if isinstance(batch, (float, int, str)):
    return batch
  if isinstance(batch, dict):
    return {sub_k: to_device(sub_v, device) for sub_k, sub_v in batch.items()}
  if isinstance(batch, (tuple, list)):
    return [to_device(x, device) for x in batch]
  else:
    return batch.to(device)


def pin_memory_recursive(batch):
  if batch is None:
    return None
  if isinstance(batch, (float, int, str)):
    return batch
  if isinstance(batch, dict):
    return {sub_k: pin_memory_recursive(sub_v) for sub_k, sub_v in batch.items()}
  if isinstance(batch, (tuple, list)):
    return [pin_memory_recursive(x) for x in batch]
  else:
    return batch.pin_memory()


def get_model_device(module: torch.nn.Module):
  return next(module.parameters()).device
