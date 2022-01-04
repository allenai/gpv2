import logging
from typing import Union, Tuple, Dict, Optional, Callable, List

import torch
import torchvision.ops
from allennlp.common import Params
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer

from torch import nn
from torch.nn import functional as F

import numpy as np
from dataclasses import replace
from gpv2.data.gpv_datasets import COCO_CATEGORIES
from gpv2.image_featurizer.image_featurizer import ImageFeatureExtractor, ImageRegionFeatures
from gpv2.model.allennlp_beamsearch import t5_initialize_decoding
from gpv2.model.collate import CollateWithTokenizer
from gpv2.model.gpv_example import GPVExample
from gpv2.model.layers import Layer, Linear
from gpv2.model.loss import BoxPredictions, BasicGPVLoss
from gpv2.model.model import GPVModel, build_per_example_output, BeamSearchSpec
from gpv2.model.preprocess_example import ExamplePreprocessor, CaptioningPreprocessor, \
  ClassificationPreprocessor, VqaPreprocessor, LocalizationPreprocessor
from gpv2.model.t5_custom import OurT5ForConditionalGeneration
from gpv2.utils import pytorch_utils


@dataclass
class CollateLocalizationLabels:
  tokenizer: PreTrainedTokenizer

  def collate(self, batch: List[GPVExample], out):
    # Get the tokenized labels for each detection example
    if not any(x.relevance_query is not None for x in batch):
      return {}

    per_box_labels = []
    for ex in batch:
      if ex.relevance_query is not None:
        per_box_labels.append(ex.relevance_query)
      else:
        per_box_labels.append(self.tokenizer.pad_token)

    labels = self.tokenizer(
      per_box_labels, return_tensors='pt', padding=True, truncation=True)
    return dict(relevance_queries=labels["input_ids"].view(len(batch), -1))


@GPVModel.register("gpv2")
@GPVModel.register("t5-gpv-per-box")
class T5GpvPerBox(GPVModel):
  """The GPV2 model"""

  @classmethod
  def from_params(
    cls,
    params: Params,
    **kwargs,
  ):
    # Backwards compatibility fixes
    if "nms" in params:
      assert params["nms"] is None
      del params["nms"]
    if "loc_relevance_query" in params:
      assert params["loc_relevance_query"] == 'category'
      del params["loc_relevance_query"]
    if "use_image_sep" in params:
      assert not params["use_image_sep"]
      del params["use_image_sep"]
    if "image_seperator" in params:
      assert not params["image_seperator"]
      del params["image_seperator"]
    if "query_box" in params:
      assert params["query_box"] == "always"
      del params["query_box"]
    if "webqa_templates" in params:
      assert params["webqa_templates"] ==  {'type': 'templated-v1', 'oversample_questions': 3, 'oversample_test': 3, 'use_commands': True}
      del params["webqa_templates"]
    if "cls_from_query_w" in params:
      assert params["cls_from_query_w"] == 0
      del params["cls_from_query_w"]
    if "pre_tokenize" in params:
      del params["pre_tokenize"]
    return super().from_params(params, **kwargs)

  def __init__(
      self,
      t5_model_name: str,
      loss: BasicGPVLoss,
      image_feature_extractor: ImageFeatureExtractor,
      image_joiner: Layer,
      preprocessors: Optional[List[ExamplePreprocessor]]=None,
      embed_objectness_score=False,
      initialize_t5=True,
      predict_trailing_pad_tokens=False,
      image_positional_bias="zero",
      initialize_joiner="coco",
      all_lower_case=False,
      initialize_from=None,
      convert_to_relevance="sigmoid-logits",
      combine_with_objectness="multiply",
      contrast_query="other",
      box_context="none",
  ):
    """
    :param t5_model_name: t5 name
    :param loss: Loss function to use during training
    :param image_feature_extractor: Extract the region-baesd image features
    :param image_joiner: Converts those features to the t5_embedding space
    :param preprocessors: Determines how to convert examples to GPVExamples
    :param embed_objectness_score: Integrate the objectness score from `image_feature_extractor`
                                   into the image features, currently not used
    :param initialize_t5: Use the pre-trained t5 weights
    :param predict_trailing_pad_tokens: Evaluate loss on padding tokens
    :param image_positional_bias: How to handle the positional bias for image features
    :param initialize_joiner: How to initialize the image_joiner, an experimental feature that
                              I don't think made much difference, but it still used in some of
                              out models.
    :param all_lower_case: Always lower case input/output text
    :param initialize_from: Initialize from the input checkpoint
    :param convert_to_relevance: How to convert log-probability box scores into a relevance score
    :param combine_with_objectness: How to combine the box scores with the objectness scores
                                    from the image_feature_extractor.
    :param contrast_query: Text to use to get a not-relevant score
    :param box_context: Context to use when computer per-box scores
    """
    super().__init__()
    self.preprocessors = preprocessors
    self.box_context = box_context
    self.all_lower_case = all_lower_case
    self.t5_model_name = t5_model_name
    self.loss = loss
    self.image_feature_extractor = image_feature_extractor
    self.initialize_t5 = initialize_t5
    self.predict_trailing_pad_tokens = predict_trailing_pad_tokens
    self.image_positional_bias = image_positional_bias
    self.initialize_joiner = initialize_joiner
    self.initialize_t5 = initialize_t5
    self.initialize_from = initialize_from
    self.embed_objectness_score = embed_objectness_score
    self.combine_with_objectness = combine_with_objectness
    self.contrast_query = contrast_query
    self.from_query_box = False
    self.convert_to_relevance = convert_to_relevance

    if self.preprocessors is None:
      _process = [
        CaptioningPreprocessor(),
        ClassificationPreprocessor(),
        VqaPreprocessor(),
        LocalizationPreprocessor()
      ]
    else:
      _process = self.preprocessors

    self._example_preprocess = {x.example_type(): x for x in _process}

    self.tokenizer = AutoTokenizer.from_pretrained(self.t5_model_name)

    if self.contrast_query is not None:
      self.register_buffer("contrast_query_tok", self.tokenizer(
        self.contrast_query, return_tensors='pt', padding=True, truncation=True)["input_ids"][0])

    # Speed up tokenization by caching per-token tokenization
    self.tokenizer_cache = {}

    self.image_joiner = image_joiner

    self.model = None

    # Prediction arguements
    self.rerank_answer_options = None
    self.beam_search_spec = None
    self.register_buffer("mask", None)

  def tokenize(self, x):
    return self.tokenizer.encode(x, add_special_tokens=False)

  def _preprocess_text(self, text: str) -> Union[str, np.ndarray]:
    if self.all_lower_case:
      text = text.lower()
    return text

  def initialize(self, load_params=True):
    if self.initialize_from is not None:
      logging.info(f"Initializing model from {self.initialize_from}")
      state_dict = torch.load(self.initialize_from)
      if state_dict["image_joiner.weight"].size() != self.image_joiner.weight.size():
        state_dict["image_joiner.weight"] = F.pad(state_dict["image_joiner.weight"], [0, 5, 0, 0],)
      missing_key, unexpected_key = self.load_state_dict(state_dict, strict=False)
      logging.info(f"Missing keys {missing_key}")
      return

    if self.initialize_t5:
      logging.info(f"Loading pre-trained LM {self.t5_model_name}")
      self.model: OurT5ForConditionalGeneration = OurT5ForConditionalGeneration.from_pretrained(self.t5_model_name)
    else:
      config = AutoConfig.from_pretrained(self.t5_model_name)
      self.model = OurT5ForConditionalGeneration(config)

    self._init_non_pretrained()

    if self.initialize_joiner:
      if self.initialize_joiner == "coco":
        words = COCO_CATEGORIES
      else:
        raise NotImplementedError()
      logging.info("Initializing joiner bias with mean embeddings")
      if isinstance(self.image_joiner, Linear):
        all_tokens = set()
        for cat in words:
          all_tokens.update(self.tokenize(cat))
        all_tokens = torch.as_tensor(list(all_tokens), dtype=torch.long)
        self.image_joiner.bias.data[:] = self.model.shared(all_tokens).mean(0)
      else:
        raise NotImplementedError()

  def _init_non_pretrained(self):
    t5_dim = self.model.config.d_model
    n_heads = self.model.config.num_heads

    self.relevance_rescale = nn.Linear(2, 2)
    if self.combine_with_objectness == "multiply-rescale":
      self.objectness_factor = nn.Linear(t5_dim, 1)

    # Initialize to directly use log_probability as relevance
    self.relevance_rescale.bias.data[:] = 0
    self.relevance_rescale.weight.data[:] = 0
    self.relevance_rescale.weight.data[0, 0] = 1.0

    if self.embed_objectness_score:
      self.objectness_embed = nn.Parameter(torch.zeros(t5_dim))
    else:
      self.objectness_embed = None

    self.query_embedding = nn.Parameter(torch.zeros(t5_dim).uniform_(-0.05, 0.05))

    if self.image_positional_bias == "learned":
      self.learned_image_text_bias = nn.Parameter(torch.zeros(n_heads,))
      self.learned_image_image_bias = nn.Parameter(torch.zeros(n_heads,))
      self.learned_text_image_bias = nn.Parameter(torch.zeros(n_heads,))
    else:
      self.learned_image_text_bias = None
      self.learned_image_image_bias = None
      self.learned_text_image_bias = None

  def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
    if "mask" in state_dict:
      # In case it was built during `set_prediction_args` and accidentally saved
      del state_dict["mask"]

    state_dict = dict(state_dict)
    for k in list(state_dict):
      if k.startswith("image_relevance."):
        del state_dict[k]

    if self.model is None:
      config = AutoConfig.from_pretrained(self.t5_model_name)
      self.model = OurT5ForConditionalGeneration(config)

    self._init_non_pretrained()

    super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

  def preprocess_example_train(self, example):
    return self.preprocess_example(example, is_train=True)

  def preprocess_example(self, example, is_train=False):
    if isinstance(example, GPVExample):
      if isinstance(example.query, str):
        example = replace(example, query=[example.query])
      examples = [example]
    else:
      examples = self._example_preprocess[example.__class__].preprocess(example, is_train)

    # Set query_box to the all-image box
    default = np.array([[0.0, 0.0, 1.0, 1.0]])
    for ex in examples:
        if ex.query_boxes is None:
          ex.query_boxes = default

    # Text pre-processing
    for ex in examples:
      ex.query = [self._preprocess_text(q) for q in ex.query]
      if ex.target_text is not None:
        ex.target_text = self._preprocess_text(ex.target_text)

    if not is_train:
      assert len(examples) == 1
      return examples[0]
    else:
      return examples

  def get_collate(self, is_train=False):
    n_pos = self.model.config.n_positions
    return CollateWithTokenizer(
      self.tokenizer, self.image_feature_extractor.get_collate(is_train),
      n_pos, n_pos,
      CollateLocalizationLabels(self.tokenizer))

  def _get_encoder_pos_bias(self, seq_len, n_image):
    input_pos_bias = None
    if self.image_positional_bias in {"zero", "learned"}:
      first_self_atten = self.model.encoder.block[0].layer[0].SelfAttention
      if self.image_positional_bias == "zero":
        input_pos_bias = first_self_atten.compute_bias(seq_len, seq_len)
        input_pos_bias[:, :, :n_image, :] = 0
        input_pos_bias[:, :, :n_image] = 0
      elif self.image_positional_bias.startswith("learned"):
        n_heads = self.model.config.num_heads
        n_text = seq_len - n_image
        i_t_bias = self.learned_image_text_bias.view(1, n_heads, 1, 1).repeat(1, 1, n_image, n_text)
        t_i_bias = self.learned_text_image_bias.view(1, n_heads, 1, 1).repeat(1, 1, n_text, n_image)
        i_i_bias = self.learned_image_image_bias.view(1, n_heads, 1, 1).repeat(1, 1, n_image, n_image)
        t_t_bias = first_self_atten.compute_bias(n_text, n_text)
        input_pos_bias = torch.cat([
          torch.cat([i_i_bias, i_t_bias], 3),
          torch.cat([t_i_bias, t_t_bias], 3)
        ], 2)
      else:
        raise NotImplementedError()
    elif self.image_positional_bias is not None:
      raise NotImplementedError(self.image_positional_bias)
    return input_pos_bias

  def _encode(self, image_inputs, input_ids, input_mask):
    """Builds the encoder hidden state"""

    # Get image features
    image: ImageRegionFeatures = self.image_feature_extractor(**image_inputs)
    device = image.features.device

    # Map to t5 embedding dim
    if isinstance(self.image_joiner, Linear):
      detr_embed = self.image_joiner(image.features)
    else:
      detr_embed = self.image_joiner(image.features, self.model.shared)

    if self.embed_objectness_score:
      detr_embed += torch.exp(image.objectness).unsqueeze(2) * self.objectness_embed.unsqueeze(0).unsqueeze(0)

    # Adds a query embeddings to the query boxes
    if image.n_boxes is not None:
      batch_ixs = torch.arange(len(detr_embed), device=device, dtype=torch.long)
      end_ixs = image.n_boxes - 1
      detr_embed[batch_ixs, end_ixs] += self.query_embedding
    else:
      detr_embed[:, -1] += self.query_embedding

    query_embed = self.model.shared(input_ids)

    query_embed, input_mask = pytorch_utils.concat_masked_sequences(
      detr_embed, image.n_boxes,
      query_embed, input_mask
    )

    input_pos_bias = self._get_encoder_pos_bias(input_ids.size(1), image.features.size(1))

    encoder_outputs = self.model.encoder(
      inputs_embeds=query_embed,
      attention_mask=input_mask,
      encoder_positional_bias=input_pos_bias,
      return_dict=True
    )
    return encoder_outputs, input_mask, image

  def _get_rel_logprob(self, objectness):
    if len(objectness.size()) == 2:
      objectness = pytorch_utils.log_prob_to_logits(objectness)
    else:
      if objectness.size(2) > 2:
        non_object_lp = F.log_softmax(objectness, -1)[:, :, -1]
        object_lp = torch.log1p(-torch.exp(non_object_lp))
        objectness = torch.stack([object_lp, non_object_lp], -1)
      else:
        assert objectness.size(2) == 2
        objectness = F.log_softmax(objectness, -1)
    return objectness

  def _image_rel(self, image_features: ImageRegionFeatures, box_rel, contextual_embeds):
    """Converts box-scores and image_features to relevance scores"""
    if self.convert_to_relevance == "raw":
      box_scores = self.relevance_rescale(box_rel)

    elif self.convert_to_relevance == "sigmoid-logits":
      if len(box_rel.size()) == 2:
        assert torch.all(torch.isfinite(box_rel))
        box_rel = pytorch_utils.log_prob_to_logits(box_rel)
        assert torch.all(torch.isfinite(box_rel))
      else:
        box_rel = torch.log_softmax(box_rel, -1)

      # Re-calibrate now [batch, n_boxes, 2] logits
      box_scores = self.relevance_rescale(box_rel)
    else:
      raise NotImplementedError(self.convert_to_relevance)

    objectness = self._get_rel_logprob(image_features.objectness)

    if self.combine_with_objectness == "none":
      pass
    elif self.combine_with_objectness == "multiply":
      box_scores = F.log_softmax(box_scores, -1)
      box_scores = objectness + box_scores
    elif self.combine_with_objectness == "multiply-rescale":
      box_scores = F.log_softmax(box_scores, -1)
      factor = self.objectness_factor(contextual_embeds[:, :box_rel.size(1)])
      box_scores = objectness*factor + box_scores
    else:
      raise ValueError()

    return box_scores

  def compute_per_box_score(
      self, contextual_emb, n_boxes, relevance_query, input_mask,
      include_query=False
  ):
    """
    @returns [batch, n_boxes], or [batch, n_boxes, 2] if there is a contrastive query,
    of per-box generation log-probabilities
    """
    device = contextual_emb.device

    if not self.predict_trailing_pad_tokens:
      # -100 marks a label as not a target
      relevance_query = relevance_query.masked_fill(
        relevance_query == self.tokenizer.pad_token_id, -100)

    if not include_query:
      n_boxes = n_boxes - 1

    per_box_inputs_lst = []
    per_box_outputs_lst = []
    ixs = []
    for i, query in enumerate(relevance_query):
      if query is None:
        continue
      ixs.append(i)
      if self.box_context == "none" or self.box_context is None:
        context = None
      elif self.box_context == "query":
        query_start = n_boxes[i]
        context = contextual_emb[i, query_start:input_mask[i].sum()]
      else:
        if self.box_context == "query_end":
          ix = input_mask[i].sum() - 1
        else:
          raise ValueError()
        context = contextual_emb[i, ix].unsqueeze(0)

      box_emb = contextual_emb[i, :n_boxes[i]].unsqueeze(1)

      if context is not None:
        context = context.unsqueeze(0).repeat(box_emb.size(0), 1, 1)
        box_emb = torch.cat([box_emb, context], 1)

      per_box_inputs_lst.append(box_emb)
      per_box_outputs_lst.append(query.unsqueeze(0).repeat(box_emb.size(0), 1))

    # Build a tensor for the [batch, n_boxes] sparse representation we will fill out
    if self.contrast_query is not None:
      batched_rel_scores = torch.full(
        (n_boxes.size(0), n_boxes.max(), 2),
        -10000,
        device=device, dtype=torch.float
      )
    else:
      batched_rel_scores = torch.full(
        (n_boxes.size(0), n_boxes.max()),
        -10000,
        device=device, dtype=torch.float
      )

    if len(ixs) == 0:
      return batched_rel_scores

    per_box_inputs, per_box_mask = pytorch_utils.stack_and_pad_blocks(per_box_inputs_lst)
    per_box_outputs = torch.cat(per_box_outputs_lst, 0)
    assert per_box_inputs.size(0) == per_box_outputs.size(0)
    total_boxes = per_box_inputs.size(0)

    t5_out = self.model(
      encoder_outputs=(per_box_inputs,),
      attention_mask=per_box_mask,
      labels=per_box_outputs,
      return_dict=True,
    )
    dim = t5_out.logits.size(-1)
    per_label_score = F.cross_entropy(
      t5_out.logits.view(-1, dim), per_box_outputs.view(-1), reduction="none")
    per_label_score = -per_label_score.view(total_boxes, -1).sum(1)

    if self.contrast_query is not None:
      c_labels = self.contrast_query_tok.unsqueeze(0).repeat(total_boxes, 1)
      contrast_query_out = self.model(
        encoder_outputs=(per_box_inputs,),
        attention_mask=per_box_mask,
        labels=c_labels,
        return_dict=True,
      )
      c_per_label_score = F.cross_entropy(
        contrast_query_out.logits.view(-1, dim), c_labels.view(-1), reduction="none")
      c_per_label_score = -c_per_label_score.view(total_boxes, -1).sum(1)
      per_label_score = torch.stack([per_label_score, c_per_label_score], -1)

    on = 0
    for i in ixs:
      n = n_boxes[i]
      batched_rel_scores[i, :n] = per_label_score[on:on+n]
      on = on + n
    assert on == len(per_label_score)
    return batched_rel_scores

  def forward(self, image_inputs, input_ids, input_mask, labels,
              relevance_queries=None) -> Tuple[torch.Tensor, Dict[str, float]]:
    encoder_outputs, input_mask, image_features = self._encode(image_inputs, input_ids, input_mask)

    if relevance_queries is not None:
      per_box_scores = self.compute_per_box_score(
        encoder_outputs.last_hidden_state, image_features.get_n_boxes(),
        relevance_queries, input_mask, True)

      rel = self._image_rel(image_features, per_box_scores, encoder_outputs.last_hidden_state)
    else:
      rel = None

    boxes = image_features.boxes
    n_boxes = image_features.n_boxes

    t5_out = self.model(
      encoder_outputs=encoder_outputs,
      attention_mask=input_mask,
      labels=labels["text_labels"],
      return_dict=True,
    )

    if not self.predict_trailing_pad_tokens:
      # -100 marks a label as not a target
      labels["text_labels"] = labels["text_labels"].masked_fill(
        labels["text_labels"] == self.tokenizer.pad_token_id, -100)
    return self.loss(t5_out.logits, BoxPredictions(boxes, rel, n_boxes), labels)

  def set_prediction_args(
      self,
      beam_search_spec: BeamSearchSpec=None,
      answer_options=None, mask=None, nms=None,
      rerank_answer_options=False,
  ):
    if rerank_answer_options:
      if answer_options is None:
        raise ValueError("No answer options to re-rank!")
      self.rerank_answer_options = answer_options
    else:
      self.rerank_answer_options = None

    self.beam_search_spec = beam_search_spec

    voc_len = self.model.config.vocab_size
    device = pytorch_utils.get_model_device(self)

    if mask is not None:
      words = mask.get_target_words()
      tensor_mask = np.zeros([voc_len], dtype=np.bool)
      for word in words:
        tensor_mask[self.tokenize(word)] = True
      for word in COCO_CATEGORIES:
        if word not in words:
          tensor_mask[self.tokenize(word)] = False
      tensor_mask[self.tokenizer.eos_token_id] = mask.target_eos()
      tensor_mask = torch.as_tensor(tensor_mask, device=device)
      self.register_buffer("mask", tensor_mask.float() * mask.val, persistent=False)
    else:
      self.register_buffer("mask", None, persistent=False)

    self.register_buffer("answer_ids", None)

    if answer_options is not None:
      if rerank_answer_options:
        if beam_search_spec:
          raise ValueError("No beam search if we just doing re-ranking")
        tokenized_answers = self.tokenizer(
          answer_options, return_tensors='pt', padding=True, max_length=self.model.config.n_positions)
        labels = tokenized_answers["input_ids"].to(device)
        if not self.predict_trailing_pad_tokens:
          # -100 marks a label as not a target
          labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        self.register_buffer("answer_ids", labels, persistent=False)
      else:
        eos = self.tokenizer.eos_token_id
        tokenized = [self.tokenize(x) + [eos]
                     for x in answer_options]
        answer_mask = np.zeros((max(len(x) for x in tokenized), voc_len), dtype=np.bool)
        for tok in tokenized:
          answer_mask[np.arange(len(tok)), tok] = True
        answer_mask = torch.as_tensor(answer_mask, device=device).float()
        # Word pieces that can never be part of an answer option get a large negative weight
        answer_mask = (1 - answer_mask) * -1e9
        if self.mask is not None:
          answer_mask = answer_mask + self.mask.unsqueeze(0)
        self.register_buffer("mask", answer_mask, persistent=False)

  def predict(
      self, image_inputs, input_ids, input_mask, labels=None, relevance_queries=None,
      no_box_sort=False
  ):
    # Use no_grad just so clients don't have to remember to
    with torch.no_grad():
      return self._predict(image_inputs, input_ids, input_mask, relevance_queries, no_box_sort)

  def _predict(
      self, image_inputs, input_ids, input_mask, relevance_queries=None, no_box_sort=False):
    encoder_outputs, input_mask, image_features = self._encode(image_inputs, input_ids, input_mask)
    if relevance_queries is not None:
      per_box_scores = self.compute_per_box_score(
        encoder_outputs.last_hidden_state,
        image_features.get_n_boxes(), relevance_queries, input_mask, True)
      rel = self._image_rel(image_features, per_box_scores, encoder_outputs.last_hidden_state)
      rel = rel.softmax(-1)[:, :, 0]
    else:
      if len(image_features.objectness.size()) == 3:
        objectness = self._get_rel_logprob(image_features.objectness)
        rel = F.softmax(objectness, -1)[:, :, 0]
      else:
        rel = torch.exp(image_features.objectness)

    if self.beam_search_spec is not None:
      if self.mask is None:
        post_process = None
      else:
        def post_process(logits, _, time_step):
          if len(self.mask.size()) == 1:
            return F.log_softmax(logits + self.mask, -1)
          else:
            return F.log_softmax(logits + self.mask[time_step], -1)
      bs = self.beam_search_spec.build(self.tokenizer.eos_token_id)
      decode_init = t5_initialize_decoding(
        self.tokenizer, self.model, encoder_outputs[0], input_mask, post_process)
      input_ids, logprobs = bs.search(*decode_init)
      input_ids = input_ids.detach().cpu()

      out_text = []
      for batch in range(len(input_ids)):
        text = [self.post_process_generation(x) for x in input_ids[batch]]
        out_text.append(text)

    elif self.rerank_answer_options:
      n_answers = len(self.answer_ids)
      n_queries, enc_len, dim = encoder_outputs.last_hidden_state.size()
      labels = self.answer_ids.unsqueeze(0).repeat(n_queries, 1, 1).view(n_queries*n_answers, -1)
      input_mask = input_mask.unsqueeze(1).repeat(1, n_answers, 1).view(n_queries*n_answers, -1)
      encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(1)\
        .repeat(1, n_answers, 1, 1).view(n_queries*n_answers, enc_len, dim)
      t5_out = self.model(
        encoder_outputs=encoder_outputs,
        attention_mask=input_mask,
        labels=labels,
        return_dict=True,
      )
      dec_len = labels.size(1)
      if self.mask is not None:
        t5_out.logits = F.log_softmax(t5_out.logits, -1) + self.mask
      per_answer_loss = F.cross_entropy(
        t5_out.logits.view(n_queries*n_answers*dec_len, -1),
        labels.view(n_queries*n_answers*dec_len), reduction="none"
      ).view(n_queries, n_answers, dec_len)
      per_answer_loss = per_answer_loss.sum(-1).cpu().numpy()
      answer_ranks = np.argsort(per_answer_loss, axis=1)
      out_text = []
      logprobs = []
      for batch in range(n_queries):
        out_text.append([self.rerank_answer_options[r] for r in answer_ranks[batch]])
        # Negative convert NLL to LL
        logprobs.append(-per_answer_loss[batch][answer_ranks[batch]])
    else:
      out_text, logprobs = None, None

    return build_per_example_output(
      out_text, logprobs, image_features.boxes, rel, image_features.n_boxes, no_box_sort=no_box_sort)

  def post_process_generation(self, generated_ids):
    return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
