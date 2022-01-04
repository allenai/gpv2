"""
Expanded templates for WebQA
"""
import logging
from os.path import dirname, join

from allennlp.common import FromParams, Params, Registrable

from gpv2.utils.py_utils import load_json_object

TEMPLATES = {}

TEMPLATES['adj'] = [
  "WH ADJ_TYPE is DT_OBJ?",
  "What is the ADJ_TYPE of DT_OBJ?",
  "CMD the ADJ_TYPE of DT_OBJ.",
]

TEMPLATES['verb'] = [
  'What is being done?',
  "WH action is being done?",
  "WH activity is being done?",
  "WH activity is this?",
  "WH action is being taken?",
  "CMD the activity being done.",
  "CMD the action being done.",
  "CMD the action being taken.",
]


TEMPLATES['verb_object'] = [
  "What is DT_OBJ doing?",
  "What action is DT_OBJ taking?",
  "What action is DT_OBJ performing?",
  "What action is DT_OBJ carrying out?",
  "What action is DT_OBJ doing?",
  "What activity is DT_OBJ doing?",
  "CMD the action being taken by DT_OBJ.",
  "CMD the activity DT_OBJ is doing.",
  "CMD what DT_OBJ is doing.",
]

TEMPLATES['query'] = [
  "What is this?",
  "What is that?",
]


TEMPLATES['noun'] = [
  "What is DT_OBJ?",
  "What OBJ is this?",
  "What OBJ is that?",
  "NAME DT_OBJ.",
]

SUBSTITUIONS = {
  "DT_OBJ": [
    "this object", "this entity", "this thing",
    "the object", "the entity",
    "that object", "that entity",  "that thing"
  ],
  "DT": ["the", "this", "that"],
  "OBJ": ['object', 'entity'],
  "CMD": ["Describe", "State", "Specify", "Name"],
  "NAME": ["Describe", "Specify", "Name", "Classify"],
  "WH": ["What", "Which"]
}


def _expand_templates(templates):
  for (prefix, subs) in SUBSTITUIONS.items():
    out = []
    for template in templates:
      if prefix in template:
        for sub in subs:
          out.append(template.replace(prefix, sub))
      else:
        out.append(template)
    templates = out
  return templates


def _substitute_noun(templates, noun):
  out = []
  for x in templates:
    if "DT_OBJ" in x:
      out.append(x.replace("DT_OBJ", f"DT {noun}"))
    elif "OBJ" in x:
      out.append(x.replace("OBJ", noun))
    else:
      raise ValueError()
  return out


def get_noun_templates():
  return _expand_templates(TEMPLATES['noun'])


def get_query_tempates():
  return _expand_templates(TEMPLATES['query'])


def get_adj_templates(adj_type, noun=None):
  templates = [x.replace("ADJ_TYPE", adj_type) for x in TEMPLATES['adj']]
  if noun is not None:
    templates = _substitute_noun(templates, noun)
  return templates


def get_verb_templates(noun=None):
  templates = TEMPLATES['verb_object']
  if noun is not None:
    templates = _substitute_noun(templates, noun)
  else:
    # Add generic no-noun templates
    templates = TEMPLATES["verb"] + templates
  return templates


ADJ_TYPES = load_json_object(join(dirname(__file__), "webqa_adj_types.json"))


class WebQaQueryGenerator(Registrable):
  def get_prompts(self, x, is_train=True):
    raise ValueError()


@WebQaQueryGenerator.register("default")
class DefaultWebQueryGenerator(WebQaQueryGenerator):
  """Generates simple default webqa prompts"""

  DEFAULT_PROMPTS = {
    "q": 'What is this?',
    "1n": 'What object is this?',
    "1v": 'What is this entity doing?',
    "1a": 'What ADJ_TYPE is this entity?',
    "2v": 'What is this NOUN doing?',
    "2a": 'What ADJ_TYPE is this NOUN?',
  }

  def get_prompts(self, x, is_train=True):
    prompt = self.DEFAULT_PROMPTS[x.qtype]
    if "ADJ_TYPE" in prompt:
      assert x.adj is not None
      prompt = prompt.replace("ADJ_TYPE", ADJ_TYPES[x.adj])
    if "NOUN" in prompt:
      assert x.noun is not None
      prompt = prompt.replace("NOUN", x.noun)
    return [prompt]


@WebQaQueryGenerator.register("templated-v1")
class TemplateWebQueryGenerator(WebQaQueryGenerator):
  """Generates webqa prompts for the different webqa question types"""

  @classmethod
  def from_params(
      cls,
      params: Params,
      constructor_to_call=None,
      constructor_to_inspect=None,
      **extras,
    ):
      if "type" in params or "version" in params:
        params = Params({})
        logging.warning("Loading with older version, templates have changed slighly")
      return super().from_params(params, constructor_to_call, constructor_to_inspect, **extras)

  def __init__(self, oversample_questions=3, oversample_test=3, use_commands=True):
    # Precompute these
    self.use_commands = use_commands
    self.oversample_questions = oversample_questions
    self.oversample_test = oversample_test
    self.noun_qa = self._expand_templates(get_noun_templates())
    self.query_qa = self._expand_templates(get_query_tempates())
    self.verb_qa = self._expand_templates(get_verb_templates())

  def _expand_templates(self, templates):
    if not self.use_commands:
      templates = [x for x in templates if x.endswith("?")]
    return _expand_templates(templates)

  def get_query_prompts(self, qtype, noun=None, adj=None, verb=None, is_train=True):
    if qtype == "q":
      out = self.query_qa
    elif qtype == "1n":
      out = self.noun_qa
    elif qtype == "1v":
      out = self.verb_qa
    elif qtype == "1a":
      out = self._expand_templates(get_adj_templates(adj_type=ADJ_TYPES[adj]))
    elif qtype == "2a":
      out = self._expand_templates(get_adj_templates(adj_type=ADJ_TYPES[adj], noun=noun))
    elif qtype == "2v":
      out = self._expand_templates(get_verb_templates(noun=noun))
    else:
      raise NotImplementedError(qtype)
    if not is_train:
      # For testing always using the first template
      return out[:1]
    if self.oversample_questions is not None and any(not x.endswith("?") for x in out):
      # Commands are often more common just because they have more templated, compensate by
      out = out + [x for x in out if x.endswith("?")]
    if self.oversample_test is not None:
      out = out + out[:1] * self.oversample_test
    return out

  def get_prompts(self, x, is_train=True):
    return self.get_query_prompts(x.qtype, x.noun, x.adj, x.verb, is_train)
