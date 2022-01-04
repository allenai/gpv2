"""Loader for GPV training data, the sae data used in GPV1"""

import logging
from collections import defaultdict, Counter
from os.path import join, dirname, exists
from typing import Dict, List
import numpy as np

from gpv2 import file_paths
from gpv2.data.dataset import LocalizationExample, VqaExample, CaptioningExample, Caption, \
  ClsExample, Dataset, Task
from gpv2.data.synonyms import SYNONYMS
from gpv2.model.model import PredictionArg
from gpv2.utils import image_utils, py_utils
from gpv2.utils.downloader import download_zip
from gpv2.utils.py_utils import load_json_object, int_to_str, cache_from_s3


def get_coco_categories():
  coco_file = join(dirname(__file__), "coco_categories.json")
  return load_json_object(coco_file)


GPV_BUCKET = "ai2-prior-gpv"
GPV_KEY = "public"



COCO_ID_TO_CATEGORY = {x["id"]: x["name"] for x in get_coco_categories()}
COCO_CATEGORIES = list(COCO_ID_TO_CATEGORY.values())
COCO_CATEGORIES_TO_ID = {k: i for i, k in enumerate(COCO_CATEGORIES)}


def load_instances(kind, split, gpv_split=True) -> List[Dict]:
  """Loads GPV data in list-of-dictionary format"""

  if kind == "cls":
    ds = "coco_classification"
  elif kind == "vqa":
    ds = "vqa"
  elif kind in {"det", "detection", "coco_detection"}:
    ds = "coco_detection"
  elif kind in {"cap", "captioning", "coco_captions"}:
    ds = "coco_captions"
  elif kind in {"web_80"}:
    ds = "web_80"
  else:
    raise NotImplementedError(kind)
  if ds == "web_80":
    split_txt = ""
  elif gpv_split:
    split_txt = "gpv_split"
  else:
    split_txt = "original_split"
  target_file = join(file_paths.GPV_DATA_DIR, ds, split_txt, f"{split}.json")
  logging.info(f"Loading instances from {target_file}")
  return load_json_object(target_file)


def load_gpv_loc(split, gpv_split) -> List[LocalizationExample]:
  """Load GPV-I detection data"""

  raw_instances = load_instances("detection", split, gpv_split)
  out = []
  for x in raw_instances:
    if "coco_categories" in x:
      cats = x["coco_categories"]
      meta = {
        "gpv1-seen": cats["seen"],
        "gpv1-unseen": cats["unseen"],
        "gpv1-query": x["query"],
        "gpv1-id": x["id"]
      }
    else:
      meta = {
        "gpv1-query": x["query"],
        "gpv1-id": x["id"]
      }
    image_id = get_image_id(x["image"])
    cat_id = x["category_id"]
    gpv_id = f"coco-boxes{image_id}-cat{cat_id}"
    bbox = LocalizationExample(
      gpv_id, image_id, np.array(x["boxes"]),
      COCO_ID_TO_CATEGORY[cat_id], meta)
    out.append(bbox)
  return out


def get_image_id(image_dict):
  """
  Turns image_id dictionary found in GPV data into a single image_id
  """
  return f'coco/{image_dict["subset"]}/COCO_{image_dict["subset"]}_{str(image_dict["image_id"]).zfill(12)}.jpg'


def load_gpv_vqa(split, gpv_split) -> List[VqaExample]:
  """Load GPV-I VQA data"""

  raw_instances = load_instances("vqa", split, gpv_split)
  out = []
  for x in raw_instances:
    cats = x.get("coco_categories")
    if "answer" in x:
      meta = {"gpv1-answer": x["answer"]}
    else:
      meta = {}
    if cats is not None:
      meta.update({"gpv1-seen": cats["seen"], "gpv1-unseen": cats["unseen"], })
    if "all_answers" in x:
      answers = Counter(x["all_answers"])
    else:
      answers = None
    q = VqaExample(
      f"vqa{x['question_id']}", get_image_id(x["image"]), x["query"],
      answers, meta=meta)
    out.append(q)
  return out


def load_gpv_captioning(split, gpv_split) -> List[CaptioningExample]:
  """Load GPV-I captioning data"""

  raw_instances = load_instances("cap", split, gpv_split)
  grouped_by_image = defaultdict(list)
  for i, x in enumerate(raw_instances):
    meta = {}
    if "coco_categories" in x:
      cats = x["coco_categories"]
      meta.update({
        "gpv1-unseen": cats["unseen"],
        "gpv1-seen": cats["seen"],
      })
    if "answer" in x:
      meta["gpv1-answer"] = x["answer"]
    meta["gpv1-query"] = x["query"]
    if "cap_id" not in x:
      assert not gpv_split
      assert split == "test"
      cap_id = f"coco-cap-test{i}"
    else:
      cap_id = x["cap_id"]
    q = Caption(f"coco-cap{cap_id}", x.get("answer"), meta)
    grouped_by_image[get_image_id(x["image"])].append(q)

  out = []
  for image_id, captions in grouped_by_image.items():
    gpv_id = f"coco-image-cap{image_utils.get_coco_int_id(image_id)}"
    out.append(CaptioningExample(gpv_id, image_id, captions))
  return out


def load_gpv_cls(split, gpv_split) -> List[ClsExample]:
  return _load_gpv_cls(split, gpv_split, False)


def load_gpv_cic(split, gpv_split) -> List[ClsExample]:
  return _load_gpv_cls(split, gpv_split, True)


def _load_gpv_cls(split, gpv_split, in_context=False) -> List:
  """Load GPV-I CLS data"""
  if in_context:
    def fn(i, image_id, category_id, box, meta):
      # TODO should change to `ident` to `cic` if we can avoid breaking already saved
      # prediction files
      return ClsExample(
        f"coco-ident{i}", image_id,
        COCO_ID_TO_CATEGORY[category_id], query_box=box, meta=meta)
  else:
    def fn(i, image_id, category_id, box, meta):
      return ClsExample(
        f"coco-box{i}", image_id,
        COCO_ID_TO_CATEGORY[category_id], crop=box, meta=meta)

  raw_instances = load_instances("cls", split, gpv_split)
  out = []
  for x in raw_instances:
    cats = x.get("coco_categories")
    meta = {"gpv1-query": x["query"]}
    if cats is not None:
      meta.update({"gpv1-seen": cats["seen"], "gpv1-unseen": cats["unseen"]})
    assert x["answer"] == COCO_ID_TO_CATEGORY[x["category_id"]]
    q = fn(
      x["id"], get_image_id(x["image"]), x["category_id"],
      x["boxes"], meta=meta)
    out.append(q)
  return out


GPV_KINDS = {
  'vqa': load_gpv_vqa,
  'cls': load_gpv_cls,
  'cap': load_gpv_captioning,
  'det': load_gpv_loc,
}


def load_gpv_instances(kind, split, gpv_split):
  return GPV_KINDS[kind](split, gpv_split)


def split_seen_unseen(instances):
  unseen_instances = []
  seen_instances = []
  for instance in instances:
    if isinstance(instance, CaptioningExample):
      unseen = sum(len(x.meta["gpv1-unseen"]) > 0 for x in instance.captions)
      unseen = unseen > 1
    else:
      unseen = instance.meta["gpv1-unseen"]
    if unseen:
      unseen_instances.append(instance)
    else:
      seen_instances.append(instance)
  return unseen_instances, seen_instances


@PredictionArg.register("coco-categories")
class CocoCategories(PredictionArg, list):
  def __init__(self, synonyms=False):
    self.synonyms = synonyms
    if self.synonyms:
      super().__init__(py_utils.flatten_list(SYNONYMS[x] for x in COCO_CATEGORIES))
    else:
      super().__init__(COCO_CATEGORIES)


@Dataset.register("gpv")
class GpvDataset(Dataset):
  """
  Loads data used in GPV1

  Note we load the data in its raw format (without the default GPV1 prompts) to allow the model
  to decide what prompts to use. Supports the default and SCE splits.
  """

  KINDS = {
    Task.VQA: load_gpv_vqa,
    Task.CLS: load_gpv_cls,
    Task.CLS_IN_CONTEXT: load_gpv_cic,
    Task.CAPTIONING: load_gpv_captioning,
    Task.LOCALIZATION: load_gpv_loc,
  }

  UNSEEN1 = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
             'sheep', 'suitcase', 'surfboard', 'wine glass']
  UNSEEN2 = ['banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']

  UNSEEN_GROUPS = {
    Task.VQA: UNSEEN1,
    Task.CLS: UNSEEN2,
    Task.CLS_IN_CONTEXT: UNSEEN2,
    Task.CAPTIONING: UNSEEN1,
    Task.LOCALIZATION: UNSEEN2,
  }

  def __init__(self, task: Task, split: str, gpv_split=True, sample=None):
    if split not in {"test", "val", "train"}:
      raise ValueError(split)
    self.sample = sample
    self.task = task
    self.split = split
    self.gpv_split = gpv_split

  def get_name(self):
    kind = "gpvsce" if self.gpv_split else "gpv"
    name = f"{kind}-{self.task}-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def get_source_name(self) -> str:
    if self.gpv_split:
      return "coco-sce-" + str(self.task)
    else:
      return "coco-" + str(self.task)

  def get_task(self) -> Task:
    return self.task

  def get_answer_options(self, synonyms=False):
    if self.task not in {Task.CLS, Task.CLS_IN_CONTEXT}:
      raise ValueError()
    return CocoCategories(synonyms)

  def load(self):
    instances = self.KINDS[self.task](self.split, self.gpv_split)
    if self.sample:
      instances.sort(key=lambda x: x.gpv_id)
      np.random.RandomState(613423).shuffle(instances)
      return instances[:self.sample]
    else:
      return instances
