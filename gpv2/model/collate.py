from collections import Callable
from typing import Any, List

from dataclasses import dataclass
from transformers import PreTrainedTokenizer

from gpv2.image_featurizer.image_featurizer import ImageCollater
from gpv2.model.gpv_example import GPVExample
import numpy as np


@dataclass
class CollateWithTokenizer(Callable):
  """Collate GPVExamples into tensor features"""

  tokenizer: PreTrainedTokenizer

  """How to collate the images"""
  image_collater: ImageCollater
  q_len: int
  ans_len: int

  """Extra collation to perform"""
  other_collate: Any = None

  def __call__(self, batch: List[GPVExample]):
    queries = []
    answers = []

    for ex in batch:
      if isinstance(ex.query, str):
        q = ex.query
      else:
        q = ex.query[np.random.randint(0, len(ex.query))]

      if ex.target_text is None or len(ex.target_text) == 0:
        # This is a bit messy since it conflates no output text requested (therefore, a
        # detection examples) with an unlabelled example (predicting a caption with no known label),
        # although there is no harm done since we ignore the labels when predicting anyway
        a = self.tokenizer.pad_token
      elif isinstance(ex.target_text, list):
        a = ex.target_text[np.random.randint(0, len(ex.target_text))]
      else:
        a = ex.target_text

      queries.append(q)
      answers.append(a)

    image_data = self.image_collater.collate(batch)
    image_inputs, box_targets = image_data

    queries = self.tokenizer(
      queries, return_tensors='pt', padding=True, max_length=self.q_len, truncation=True)
    answers = self.tokenizer(
      answers, return_tensors='pt', padding=True, max_length=self.ans_len)

    labels = dict(
      text_labels=answers["input_ids"],
      box_targets=box_targets,
      # Noted for logging purposes
      loss_logging=[None if x.meta is None else x.meta.get("loss-logging") for x in batch]
    )

    out = dict(
      input_ids=queries["input_ids"],
      input_mask=queries["attention_mask"],
      labels=labels,
      image_inputs=image_inputs
    )

    if self.other_collate:
      out.update(self.other_collate.collate(batch, out))
    return out
