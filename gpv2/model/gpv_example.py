from typing import Union, List, Any, Optional, Tuple

import numpy as np

from dataclasses import dataclass


@dataclass
class GPVExample:
  """Data representation that can be passed to GPV `collate` functions

  This representation puts the "raw" input examples for various tasks into a universal format
  so examples from different task can be jointly processed, and may encompass some pre-processing,
  like deciding what queries to use for an example, or tokenizing the text
  """

  """ID for this example that is unique among all datasets"""
  id: str

  """Image this is for"""
  image_id: Union[str, int]

  """Query (or list of queries) that can be used for this example, possibly tokenized"""
  query: Union[str, List[str], List[np.ndarray]]

  """Query for deciding which boxes are relevant, used by some models to compute box ranking"""
  relevance_query: Optional[str] = None

  """Crop of the image to use, in [x, y, h, w] form"""
  crop: Optional[Tuple[float, float, float, float]] = None

  """Optional array of boxes that are part of the query in [x, y, w, h] form"""
  query_boxes: np.ndarray = None

  """Boxes to predict for this query, if there are any, in [x, y, h, w] form"""
  target_boxes: Optional[np.ndarray] = None

  """Text to learn to generate for this example, if there is any"""
  target_text: Optional[Any] = None

  """Meta-data about this example"""
  meta: Any = None

  def get_gpv_id(self):
    return self.id

