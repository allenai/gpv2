"""Module to compute VinVL features for an image"""

import logging
from os.path import join, exists, relpath
from typing import Any, Union, Optional, Tuple, List

import h5py
import torch
import torchvision.ops
from dataclasses import dataclass
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from gpv2.image_featurizer.image_featurizer import ImageRegionFeatures
from gpv2.image_featurizer.vinvl_featurizer import VinvlImageFeaturizer
from gpv2.utils import pytorch_utils, image_utils
from gpv2.utils.pytorch_utils import QueueDataset


@dataclass
class WrapCollate:
  """Collater that also returns the batch, and reports errors"""
  collater: Any

  def __call__(self, batch: 'List[ExtractionTarget]'):
    try:
      features = self.collater.collate(batch)[0]
    except Exception as e:
      # Mulit-processing is sometimes flakey about showing us the right error
      # so print here just to ensure it is visible
      print(f"Error collating examples {[x.image_id for x in batch]}")
      print(e)
      raise e
    return batch, features


def _run(device, data, args):
  model = VinvlImageFeaturizer()
  model.to(device)

  loader = DataLoader(
    data, batch_size=args.batch_size, shuffle=False,
    collate_fn=WrapCollate(model.get_collate(False)), num_workers=args.num_workers)
  for examples, batch in loader:
    batch = pytorch_utils.to_device(batch, device)
    with torch.no_grad():
      out: ImageRegionFeatures = model(**batch)
    if args.no_features:
      out.features = None
    batch, n = out.boxes.size()[:2]
    boxes = torchvision.ops.box_convert(out.boxes.view(-1, 4), "cxcywh", "xyxy")
    out.boxes = boxes.view(batch, n, 4)
    yield examples, out.numpy()


def _run_worker(rank, devices, input_q, out_q, args):
  try:
    dataset = QueueDataset(input_q)
    for x in _run(devices[rank], dataset, args):
      out_q.put(x)
  except Exception as e:
    out_q.put(e)
    raise e


def _run_dist(targets, devices, args):
  # We need to use a spawn context to be compatible with `torch.multiprocessing.spawn`
  ctx = torch.multiprocessing.get_context("spawn")
  input_q = ctx.Queue()
  for x in targets:
    input_q.put(x)
  out_q = ctx.Queue()

  args = (devices, input_q, out_q, args)

  context = torch.multiprocessing.spawn(_run_worker, nprocs=len(devices), args=args, join=False)

  seen = 0
  while True:
    val = out_q.get()
    if isinstance(val, Exception):
      # A worker has failed, call .join immediately to get the exception and stacktrace
      # We need to use `cancel_join_thread` first so the child process terminates correctly
      # even though the queues still have enqueud data
      input_q.cancel_join_thread()
      out_q.cancel_join_thread()
      while not context.join():
        pass
      raise RuntimeError("context.join() should have raised an error!")

    k, v = val
    seen += len(k)

    if seen == len(targets):
      assert input_q.empty()
      yield k, v
      break
    else:
      yield k, v

  while not context.join():
    pass

  return


@dataclass(frozen=True)
class ExtractionTarget:
  """Target to extract features for, duck-types GPVExample"""
  image_id: Union[str, int]
  image_file: Optional[str]=None
  crop: Optional[Tuple[float, float, float, float]]=None
  query_boxes: Optional[np.ndarray]=None

  @property
  def target_boxes(self):
    return None


def add_args(parser):
  """Adds feature extractions parameters to `parser`"""
  parser.add_argument("--no_query", action="store_true",
                      help="Don't compute features for query boxes")
  parser.add_argument("--no_features", action="store_true",
                      help="Extract boxes and relevance score but not features")
  parser.add_argument("--append", action="store_true",
                      help="Append to the output hdf5 file")
  parser.add_argument("--devices", default=None, nargs="+", type=int,
                      help="GPU devices to run on")
  parser.add_argument("--batch_size", default=12, type=int)
  parser.add_argument("--num_workers", default=4, type=int)
  parser.add_argument("--output", required=True,
                      help="Output hdf5 file")


def run(targets: List[ExtractionTarget], args):
  """Extract VinVL features for `targets`

  `args` should contain the parameters set from `add_args`
  """

  if args.append:
    logging.info("Checking for existing entries")
    out = h5py.File(args.output, "a")
    all_keys = set(out.keys())
    filtered = []
    for target in targets:
      key = image_utils.get_hdf5_key_for_image(target.image_id, target.crop)
      if key not in all_keys:
        filtered.append(target)
    logging.info(f"Already have {len(targets)-len(filtered)}/{len(targets)} images")
    targets = filtered
    if len(targets) == 0:
      return
  else:
    if exists(args.output):
      raise ValueError(f"{args.output} already exists!")
    out = h5py.File(args.output, "w")

  devices = args.devices
  if devices is None:
    devices = [pytorch_utils.get_device()]

  if len(devices) == 1:
    it = _run(devices[0], targets, args)
  else:
    it = _run_dist(targets, devices, args)

  pbar = None  # Start pbar after the first iteration so we don't interrupt it with log output
  for examples, region_features in it:
    if pbar is None:
      pbar = tqdm(total=len(targets), desc="fe", ncols=100)
    pbar.update(len(examples))

    for ix, target in enumerate(examples):
      assert len(region_features.boxes[ix]) > 0
      assert region_features.boxes[ix].max() <= 1.0, target.image_id
      assert region_features.boxes[ix].min() >= 0.0, target.image_id
      if region_features.n_boxes is None:
        n_boxes = region_features.boxes[ix].shape[0]
      else:
        n_boxes = region_features.n_boxes[ix]

      if target.query_boxes is not None:
        e = n_boxes - len(target.query_boxes)
      else:
        e = n_boxes
      to_save = dict(
        bboxes=region_features.boxes[ix, :e],
        objectness=region_features.objectness[ix, :e],
      )
      if region_features.features is not None:
        to_save["features"] = region_features.features[ix, :e]

      if target.query_boxes is not None:
        to_save["query_bboxes"] = region_features.boxes[ix, e:n_boxes]
        to_save["query_features"] = region_features.features[ix, e:n_boxes]
        to_save["query_objectness"] = region_features.objectness[ix, e:n_boxes]

        # Sanity check the query boxes saved should match the input boxes
        expected_query_boxes = torchvision.ops.box_convert(torch.as_tensor(target.query_boxes), "xywh", "xyxy")
        if not torch.all(expected_query_boxes <= 1.0):  # if target.query_boxes are not normalized
          w, h = image_utils.get_image_size(target.image_id)
          expected_query_boxes /= torch.as_tensor([w, h, w, h]).unsqueeze(0)
        assert torch.abs(expected_query_boxes - to_save["query_bboxes"]).max() < 1e-6

      key = image_utils.get_hdf5_key_for_image(target.image_id, target.crop)
      grp = out.create_group(key)
      for k, data in to_save.items():
        grp.create_dataset(k, data=data)

  out.close()
