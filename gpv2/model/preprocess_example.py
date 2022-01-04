from collections import Counter
from typing import Optional

from allennlp.common import FromParams

from gpv2.data.dataset import CaptioningExample, LocalizationExample, VqaExample, ClsExample
from gpv2.data.webqa_templates import WebQaQueryGenerator
from gpv2.model.gpv_example import GPVExample
import numpy as np


BBOX_QUERIES = [
  'Locate {}.',
  'Locate {} in the image.',
  'Locate {} in this image.',
  'Locate instances of {}.',
  'Locate instances of {} in the image.',
  'Locate instances of {} in this image.',
  'Locate all instances of {}.',
  'Locate all instances of {} in the image.',
  'Locate all instances of {} in this image.',
  'Find {}.',
  'Find {} in the image.',
  'Find {} in this image.',
  'Find instances of {}.',
  'Find instances of {} in the image.',
  'Find instances of {} in this image.',
  'Find all instances of {}.',
  'Find all instances of {} in the image.',
  'Find all instances of {} in this image.',
]

SEGMENTATION_QUERIES = [
  'Segment {}.',
  'Segment {} in the image.',
  'Segment {} in this image.',
  'Segment instances of {}.',
  'Segment instances of {} in the image.',
  'Segment instances of {} in this image.',
  'Segment all instances of {}.',
  'Segment all instances of {} in the image.',
  'Segment all instances of {} in this image.',
  'Find the masks of {}.',
  'Find the masks of {} in the image.',
  'Find the masks of {} in this image.',
]

CLS_QUERIES = [
  'What is this?',
  'What is this object?',
  'What object is this?',
  'What is this thing?'
]


class ExamplePreprocessor(FromParams):
  """
  This class converts `raw` examples into GPVExamples

  This generally requires deciding what
  prompts to use and what the target output text should be. Training examples
  can have a one-to-many mapping.
  """

  def example_type(self):
    raise NotImplementedError()

  def preprocess(self, example, is_train=False, include_meta=False):
    raise NotImplementedError()


class ClassificationPreprocessor(ExamplePreprocessor):
  CLS_QUERIES = [
    'What is this?',
    'What is this object?',
    'What object is this?',
    'What is this thing?'
  ]

  def example_type(self):
    return ClsExample

  def preprocess(self, example, is_train=False, include_meta=False):
    ex = GPVExample(
      example.gpv_id,
      example.image_id,
      self.CLS_QUERIES,
      # Convert to a [1, 4] array
      query_boxes=None if example.query_box is None else np.array(example.query_box)[None, :],
      crop=example.crop,
      target_text=example.category,
      meta=example.meta if include_meta else None
    )
    return [ex]


class CaptioningPreprocessor(ExamplePreprocessor):

  CAPTION_QUERIES = [
    'Generate a caption.',
    'Generate a description.',
    'Describe this image.',
    'Describe the image.',
    'Caption this image.',
    'Caption the image.',
    'What is happening in this image?',
    'What is happening in the image?',
    'What is going on in this image?',
    'What is going on in the image?',
    'Generate a caption for this image.',
    'Generate a caption for the image.',
    'Generate a description for this image.',
    'Generate a description for the image.',
  ]

  def example_type(self):
    return CaptioningExample

  def preprocess(self, example, is_train=False, include_meta=False):
    if is_train:
      out = []
      for cap in example.captions:
        out.append(GPVExample(
          cap.gpv_id,
          example.image_id,
          query=self.CAPTION_QUERIES,
          target_text=cap.caption,
          meta=cap.meta if include_meta else None
        ))
    else:
      out = [GPVExample(
        example.gpv_id,
        example.image_id,
        query=self.CAPTION_QUERIES,
        target_text=None,
        meta=example.meta if include_meta else None
      )]
    return out


class VqaPreprocessor(ExamplePreprocessor):

  def example_type(self):
    return VqaExample

  def preprocess(self, example, is_train=False, include_meta=False):
    if isinstance(example.answers, Counter):
      answer = max(example.answers.items(), key=lambda x: (x[1], len(x[0])))[0]
    elif isinstance(example.answers, list):
      answer = example.answers[0]
    else:
      answer = example.answers
    return [GPVExample(
      example.gpv_id,
      example.image_id,
      [example.question],
      target_text=answer,
      meta=example.meta if include_meta else None
    )]


class LocalizationPreprocessor(ExamplePreprocessor):

  def example_type(self):
    return LocalizationExample

  def preprocess(self, example, is_train=False, include_meta=False):
    out = [GPVExample(
      example.gpv_id,
      example.image_id,
      [x.format(example.category) for x in BBOX_QUERIES],
      relevance_query=example.category,
      target_boxes=example.bboxes,
      target_text=None,
      meta=example.meta if include_meta else None
    )]
    return out
