import re
from collections import defaultdict, Counter
from numbers import Number
from typing import Optional, List, Dict
import numpy as np


import torch
from allennlp.common import FromParams, Registrable
from dataclasses import dataclass, replace
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
import third_party.detection_metrics.lib.Evaluator as det_evaluator

from gpv2.data.dataset import VqaExample, ClsExample, WebQaExample, LocalizationExample, \
  CaptioningExample
from gpv2.data.gpv_datasets import COCO_CATEGORIES
from gpv2.data.synonyms import SYNONYMS
from gpv2.model.model import GPVExampleOutput
from gpv2.train.vqa2_eval_data import punct, periodStrip, commaStrip, manualMap, articles, \
  contractions
from gpv2.utils import py_utils
from gpv2.utils.image_utils import get_image_size
from gpv2.utils.quiet_ptbtokenizer import QuitePTBTokenizer


def vqa_score(answer, ground_truth_answer_counts):
  gt_answers = {k.lower(): v for k, v in ground_truth_answer_counts.items()}
  return min(gt_answers.get(answer, 0) / 3, 1)


@dataclass(frozen=True)
class ResultKey(FromParams):
  """Key for a result from a model"""

  metric_name: str
  subset_name: Optional[str] = None
  dataset_name: Optional[str] = None

  def __str__(self):
    out = [self.dataset_name, self.subset_name, self.metric_name]
    return "/".join(x for x in out if x is not None)

  def __repr__(self):
    return str(self)


class Evaluator(Registrable):
  """Computes evaluations metrics"""

  def evaluate(
      self, examples: List, predictions: Dict[str, GPVExampleOutput],
      allow_partial=False, subset_mapping=None
  ) -> Dict[ResultKey, Number]:
    """Computes corpus wide metrics

    :param examples: List of source examples
    :param predictions: example key -> model output
    :param allow_partial: Allow the predictions to only cover a subset of `examples`,
                          in which only those predictions should be evaluated
    :param subset_mapping: Function that maps example -> list of strings, names of the subsets that
                           example is part of
    """
    raise NotImplementedError()


class PerExampleEvaluator(Evaluator):
  """Computes per-examples evaluations metrics"""

  def evaluate_examples(self, examples: List, predictions: Dict[str, GPVExampleOutput])-> List[Dict[str, Number]]:
    raise NotImplementedError()

  def evaluate(
      self,
      examples: List,
      predictions: Dict[str, GPVExampleOutput],
      allow_partial=False,
      mean=True,
      subset_mapping=None
  ) -> Dict[ResultKey, Number]:
    examples_with_predictions = [x for x in examples if x.get_gpv_id() in predictions]
    if not allow_partial and (len(examples) != len(examples_with_predictions)):
      raise ValueError(f"Only {len(examples_with_predictions)}/{len(examples)} "
                       f"of examples have predictions")
    examples = examples_with_predictions
    per_example_scores = self.evaluate_examples(examples, predictions)
    per_metric_scores = py_utils.transpose_list_of_dicts(per_example_scores)

    subsets = defaultdict(list)
    all_ids = [x.get_gpv_id() for x in examples]

    id_to_ix = {k: i for i, k in enumerate(all_ids)}
    subsets[None] = list(range(len(all_ids)))

    if subset_mapping is not None:
      for example in examples:
        example_id = id_to_ix[example.get_gpv_id()]
        for subset in subset_mapping(example):
          subsets[subset].append(example_id)

    out = {}
    for metric_name, score in per_metric_scores.items():
      score = np.array(score)
      for subset_name, ixs in subsets.items():
        if mean:
          out[ResultKey(metric_name, subset_name)] = float(np.mean(score[ixs]))
        else:
          out[ResultKey(metric_name, subset_name)] = (float(np.sum(score[ixs])), len(ixs))

    return out


@Evaluator.register("vqa-evaluator")
class VqaEvaluator(PerExampleEvaluator):

  def evaluate_examples(self, examples: List[VqaExample],
                        predictions: Dict[str, GPVExampleOutput], add_scores=False):
    out = []
    for example in examples:
      answer = predictions[example.gpv_id].text[0]
      score = vqa_score(answer.lower(), example.answers)
      out.append(dict(score=score))
    return out


@Evaluator.register("cls-evaluator")
class ClsEvaluator(PerExampleEvaluator):

  def evaluate_examples(self, examples: List[ClsExample], predictions: Dict[str, GPVExampleOutput]):
    out = []
    for example in examples:
      answer = predictions[example.gpv_id].text[0].lower()
      gt_answer = SYNONYMS[example.category]
      out.append(dict(accuracy=answer in gt_answer))
    return out


@Evaluator.register("webqa-evaluator")
class WebQaEvaluator(PerExampleEvaluator):
  def evaluate_examples(self, examples: List[WebQaExample], predictions: Dict[str, GPVExampleOutput]):
    out = []
    for example in examples:
      answer = predictions[example.get_gpv_id()].text[0].lower()
      gt_answer = SYNONYMS[example.answer] if example.answer in SYNONYMS else [example.answer]
      out.append(dict(accuracy=answer in gt_answer))
    return out


@Evaluator.register("dce-cls")
class DceClsEvaluator(PerExampleEvaluator):

  def __init__(self, top_k: Optional[List[int]]=(5,)):
    self.top_k = top_k

  def evaluate_examples(self, examples: List[ClsExample], predictions: Dict[str, GPVExampleOutput]):
    out = []
    for example in examples:
      answers = [x.lower() for x in predictions[example.get_gpv_id()].text]
      gt = [example.category]
      vals = {"accuracy": answers[0] in gt}
      if self.top_k is not None:
        for k in self.top_k:
          assert len(answers) >= k
          vals[f"top{k}-acc"] = any(a in gt for a in answers[:k])
      out.append(vals)
    return out


def compute_vqa_accuracy(
    gt_answers: List[str],
    pred_answers: List[str]) -> List[float]:
  ngt_answers = [preprocess_answer(ans) for ans in gt_answers]
  topk_npred_answers = [preprocess_answer(ans) for ans in pred_answers]
  gt_consensus = Counter(ngt_answers)
  return [vqa_accuracy(ans, gt_consensus) for ans in topk_npred_answers]


def vqa_accuracy(npred_answer: str, gt_consensus: Counter):
  return min(gt_consensus[npred_answer]/3,1)


def processPunctuation(inText):
  outText = inText
  for p in punct:
    if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
      outText = outText.replace(p, '')
    else:
      outText = outText.replace(p, ' ')
  outText = periodStrip.sub("",outText,re.UNICODE)
  return outText


def processDigitArticle(inText):
  outText = []
  tempText = inText.lower().split()
  for word in tempText:
    word = manualMap.setdefault(word, word)
    if word not in articles:
      outText.append(word)
    else:
      pass
  for wordId, word in enumerate(outText):
    if word in contractions:
      outText[wordId] = contractions[word]
  outText = ' '.join(outText)
  return outText


def preprocess_answer(ans):
  ans = ans.replace('\n', ' ')
  ans = ans.replace('\t',' ')
  ans = ans.lower().strip()
  return processDigitArticle(processPunctuation(ans))


@Evaluator.register("opensce-vqa")
class DceVqaEvaluator(PerExampleEvaluator):

  def __init__(self, top_k: Optional[List[int]]=(5,)):
    self.top_k = top_k
    if top_k is not None:
      assert all(x > 0 for x in top_k)

  def evaluate_examples(self, examples: List[VqaExample], predictions: Dict[str, GPVExampleOutput]):
    out = []
    for example in examples:
      max_k = 1 if self.top_k is None else max(self.top_k)
      answers = predictions[example.get_gpv_id()].text[:max_k]
      gt = example.answers
      scores = compute_vqa_accuracy(gt, answers)
      vals = dict(acc=scores[0])
      if self.top_k:
        for k in self.top_k:
          if k > len(scores):
            raise ValueError(f"Cannot evaluate top-{k}, but only have top-{len(scores)} predictions")
          vals[f"top{k}-acc"] = max(scores[:k])
      out.append(vals)
    return out


@Evaluator.register("localization-evaluator")
@Evaluator.register("detect-evaluator")
class LocalizationEvaluator(PerExampleEvaluator):

  def __init__(self, iou_thresh=0.5):
    self.iou_thresh = iou_thresh

  def evaluate_examples(self, examples: List[LocalizationExample], predictions: Dict[str, GPVExampleOutput],
                        return_pr=False):
    eval_engine = det_evaluator.Evaluator()
    out = []

    for i, ex in enumerate(examples):
      pred = predictions[ex.gpv_id]
      scores = pred.relevance
      pred_boxes = pred.boxes.copy()
      gt_boxes = np.array(ex.bboxes)

      # Convert cx cy, w, h -> x1, y1, w, h
      pred_boxes[:, 0] = pred_boxes[:, 0] - 0.5 * pred_boxes[:, 2]
      pred_boxes[:, 1] = pred_boxes[:, 1] - 0.5 * pred_boxes[:, 3]

      B = pred_boxes.shape[0]
      all_boxes = det_evaluator.BoundingBoxes()
      W, H = get_image_size(ex.image_id)

      for b in range(B):
        x, y, w, h = pred_boxes[b]
        all_boxes.addBoundingBox(det_evaluator.BoundingBox(
          imageName=ex.image_id,
          classId=ex.category,
          x=x,
          y=y,
          w=w,
          h=h,
          typeCoordinates=det_evaluator.CoordinatesType.Relative,
          imgSize=(W, H),
          bbType=det_evaluator.BBType.Detected,
          classConfidence=scores[b],
          format=det_evaluator.BBFormat.XYWH))

      normalized_gt = all(all(val <= 1.0 for val in b) for b in gt_boxes)
      if not normalized_gt:
        # convert to relative coordinates
        # TODO its a bit of hack to check this by looking coordinates > 1.0
        # but we need this check atm since DCE stores relative scaling
        # coco uses absolute
        W, H = get_image_size(ex.image_id)
        gt_boxes[:, 0] = gt_boxes[:, 0] / W
        gt_boxes[:, 1] = gt_boxes[:, 1] / H
        gt_boxes[:, 2] = gt_boxes[:, 2] / W
        gt_boxes[:, 3] = gt_boxes[:, 3] / H

      B = gt_boxes.shape[0]
      for b in range(B):
        x, y, w, h = gt_boxes[b]

        all_boxes.addBoundingBox(det_evaluator.BoundingBox(
          imageName=ex.image_id,
          classId=ex.category,
          x=x,
          y=y,
          w=w,
          h=h,
          typeCoordinates=det_evaluator.CoordinatesType.Relative,
          imgSize=(W, H),
          bbType=det_evaluator.BBType.GroundTruth,
          format=det_evaluator.BBFormat.XYWH))

      det_metrics = eval_engine.GetPascalVOCMetrics(all_boxes, self.iou_thresh)
      if return_pr:
        out.append(det_metrics[0])
      else:
        out.append({"AP": det_metrics[0]['AP']})

    return out


def get_per_caption_data(examples: List[CaptioningExample], predictions):
  # In per-caption evaluation the model makes one prediction for each ground truth
  # caption, each of which it still evaluated against all the captions,
  caption_examples = []
  caption_predictions = {}
  for ex in examples:
    pred = predictions[ex.gpv_id]
    for cap in ex.captions:
      caption_examples.append(CaptioningExample(cap.gpv_id, ex.image_id, ex.captions))
      caption_predictions[cap.gpv_id] = pred
  return caption_examples, caption_predictions


@Evaluator.register("cap-evaluator")
class CaptionEvaluator(Evaluator):
  def __init__(self, cider=True, bleu=4, per_caption=False):
    self.cider = cider
    self.bleu = bleu
    self.per_caption = per_caption
    scorers = {}
    if cider:
      # from exp.ours.eval.fast_cider import FastCider
      scorers["cider"] = Cider()
    if bleu:
      scorers["bleu"] = Bleu(bleu)
    self.scorers = scorers
    self.tokenizer = QuitePTBTokenizer()

  def evaluate(
      self,
      examples: List,
      predictions: Dict[str, GPVExampleOutput],
      allow_partial=False,
      subset_mapping=None,
  ):
    examples_with_predictions = [x for x in examples if x.get_gpv_id() in predictions]
    if not allow_partial and (len(examples) != len(examples_with_predictions)):
      raise ValueError(f"Only {len(examples_with_predictions)}/{len(examples)} "
                       f"of examples have predictions")
    examples = examples_with_predictions

    if self.per_caption:
      examples, predictions = get_per_caption_data(examples, predictions)

    subsets = defaultdict(list)
    subsets[None] = examples
    if subset_mapping is not None:
      for example in examples:
        example_subsets = subset_mapping(example)
        for subset in example_subsets:
          subsets[subset].append(example)

    out = {}
    for subset_name, examples in subsets.items():
      all_scores = self._get_scores(examples, predictions)

      results = {}
      for name, scorer in self.scorers.items():
        corpus_scores, _ = all_scores[name]
        if isinstance(scorer, Cider):
          results["cider"] = corpus_scores
        elif isinstance(scorer, Bleu):
          scores, _ = all_scores[name]
          for i, score in enumerate(corpus_scores):
            results[f"bleu{i+1}"] = score
      if subset_name is not None:
        results["n"] = len(examples)
      out.update({ResultKey(metric_name=k, subset_name=subset_name): v for k, v in results.items()})
    return out

  def evaluate_examples(self, examples: List[CaptioningExample], predictions: Dict[str, GPVExampleOutput]):
    all_scores = self._get_scores(examples, predictions)

    per_examples_scores = [{} for _ in examples]
    for name, scorer in self.scorers.items():
      score, scores = all_scores[name]
      if isinstance(scorer, Cider):
        for score, ex_scores in zip(scores, per_examples_scores):
          ex_scores["cider"] = score
      elif isinstance(scorer, Bleu):
        scores = py_utils.transpose_lists(scores)
        for score, ex_scores in zip(scores, per_examples_scores):
          for i, s in enumerate(score):
            ex_scores[f"bleu{i+1}"] = s

    return per_examples_scores

  def _get_scores(self,  examples: List[CaptioningExample], predictions: Dict[str, GPVExampleOutput]):
    gts = {}
    res = {}
    for ix, instance in enumerate(examples):
      key = instance.get_gpv_id()
      assert key not in res
      res[key] = [predictions[instance.get_gpv_id()].text[0]]
      gts[key] = [x.caption.lower() for x in instance.captions]

    res = self.tokenizer.tokenize(res)
    gts = self.tokenizer.tokenize(gts)

    scores = {}
    for name, scorer in self.scorers.items():
      if isinstance(scorer, Bleu):
        scores[name] = scorer.compute_score(gts, res, verbose=0)
      else:
        scores[name] = scorer.compute_score(gts, res)
    return scores

