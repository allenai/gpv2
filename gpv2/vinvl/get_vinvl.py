import logging
from os.path import join, dirname, exists

from . import defaults
from . import sg_defaults
from .attr_rcnn import AttrRCNN
from .structures.image_list import to_image_list
from .transforms import build_transforms
from .utils.checkpoint import DetectronCheckpointer
from gpv2 import file_paths
from gpv2.utils.py_utils import DisableLogging, download_to_file


def get_vinvl(model="release"):
  if model == "release":
    model_cfg = join(dirname(__file__), "vinvl_x152c4.yaml")
    if not exists(file_paths.VINVL_STATE):
      raise ValueError("VinVL state not downladoed")
  else:
    raise NotImplementedError()

  logging.info(f"Initializing VinVL model")
  cfg = defaults._C.clone()
  cfg.set_new_allowed(True)
  cfg.merge_from_other_cfg(sg_defaults._C)

  cfg.merge_from_file(model_cfg)
  cfg.set_new_allowed(False)

  # Parameter recommended for generating features
  cfg.TEST.OUTPUT_FEATURE = True
  cfg.MODEL.ATTRIBUTE_ON = False
  # TODO should we be using the regression predictions for Localization?
  cfg.TEST.IGNORE_BOX_REGRESSION = True
  cfg.MODEL.ROI_HEADS.SCORE_THRESH = 0.2
  cfg.MODEL.ROI_HEADS.NMS_FILTER = 1
  cfg.MODEL.ROI_BOX_HEAD.COMPUTE_BOX_LOSS = True

  cfg.freeze()
  model = AttrRCNN(cfg)

  eval_transform = build_transforms(cfg, is_train=False)

  logging.info(f"Loading vinvl state from {file_paths.VINVL_STATE}")
  checkpointer = DetectronCheckpointer(cfg, model)
  with DisableLogging():
    checkpointer.load(file_paths.VINVL_STATE)
  model.device = None
  model.eval()
  return model, eval_transform
