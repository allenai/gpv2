import enum
from typing import List, Union, Optional, Tuple, Dict, Counter, Any

from allennlp.common import Registrable, FromParams
from dataclasses import dataclass

import numpy as np


class Dataset(Registrable):
  """Dataset we can train/evaluate on"""

  def get_source_name(self) -> str:
    """Return the source name (e.g., 'vqa 2.0')

    Used for logging purposes
    """
    raise NotImplementedError()

  def get_name(self) -> str:
    """Get the name of the dataset

    The name should by uniquely identified with the set of examples `load` returns since we might
    use it for caching.
    """
    raise NotImplementedError()

  def load(self) -> List:
    """Loads the examples"""
    raise NotImplementedError()


class Task(FromParams, enum.Enum):
  """Defines the tasks we have for DCE and our COCO training data"""

  CLS = "cls"
  VQA = "vqa"
  LOCALIZATION = "loc"
  CAPTIONING = "cap"
  CLS_IN_CONTEXT = "cic"

  @classmethod
  def from_params(
      cls,
      params,
      constructor_to_call=None,
      constructor_to_inspect=None,
      **extras,
  ):
    # Need a custom method here due to some interactions between
    # FromParams/Enum
    if params["type"] == "det":
      # Backward compatibility for when this was call DETECTION
      return Task.LOCALIZATION
    return Task(params["type"])

  def to_params(self):
    # Params objects can represent classes with no args as just strings
    # We do that here, which is easier to read and makes sure we can use
    # `Task` objects as keys in maps
    return self._value_

  def __str__(self):
    return self._value_

  def __reduce_ex__(self, protocol):
    # Adding `FromParam` makes enum.Enum think the default unpickle implementation will
    # fail, so it helpfully breaks pickling so we fail fast when saving with pickle.
    # In fact, the default unpickle implementation is fine because `FromParams` does not
    # add any state, so we do this redundant override of __reduce_ex__ so `enum.Enum` trusts
    # that the object can be pickled
    return enum.Enum.__reduce_ex__(self, protocol)


# Below are the different kinds of training/test examples GPV supports
# These represent "raw" examples before we have decided what prompts or
# target output text we want our models to use for those examples


@dataclass
class WebQaExample:
  """
  WebQaExample, there are 6 kinds of questions/answer pairs this might be:

  qtypes:
  q: What is the full query?
  1n: What is the noun?
  1v: What is the verb?
  1a: What is the adj?
  2a: What is the adj given the noun?
  2v: What is the verb given the noun?

  We allow the model to determine how to build natural language questions
  for these questions types
  """

  gpv_id: str
  image_id: Union[int, str]
  answer: str
  qtype: str
  noun: Optional[str]
  verb: Optional[str]
  adj: Optional[str]
  meta: Optional[Dict[str, Any]] = None

  def get_gpv_id(self):
    return self.gpv_id


@dataclass(frozen=True)
class ClsExample:
  """
  CLS (which will have crop set) or CiC (which will have query_box set) example
  """
  gpv_id: str
  image_id: str
  category: str
  crop: Optional[Tuple[float, float, float, float]] = None
  query_box: Optional[Tuple[float, float, float, float]] = None
  meta: Optional[Dict] = None

  def __post_init__(self):
    if self.crop is not None and self.query_box is not None:
      raise ValueError("Both crop and query not supported")

  def get_gpv_id(self):
    return self.gpv_id


@dataclass(frozen=True)
class VqaExample:
  gpv_id: str
  image_id: str
  question: str
  answers: Union[str, Counter]
  meta: Optional[Dict] = None

  def get_gpv_id(self):
    return self.gpv_id


@dataclass(frozen=True)
class LocalizationExample:
  gpv_id: str
  image_id: str

  """[x, y, w, h] un-normalized format"""
  bboxes: np.ndarray

  """Object name these boxes are for"""
  category: str
  meta: Optional[Dict] = None

  @property
  def crop(self):
    return None
  
  def get_gpv_id(self):
    return self.gpv_id


@dataclass(frozen=True)
class Caption:
  gpv_id: str
  caption: Optional[str]
  meta: Optional[Dict[str, Any]] = None

  def get_gpv_id(self):
    return self.gpv_id


@dataclass(frozen=True)
class CaptioningExample:
  gpv_id: str
  image_id: str
  captions: List[Caption]
  meta: Optional[Dict[str, Any]] = None

  @property
  def crop(self):
    return None

  def get_gpv_id(self):
    return self.gpv_id
