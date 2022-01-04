from os import mkdir
from os.path import join, dirname, expanduser, exists

DATA_DIR = expanduser("~/data")

# Where the webqa data is located
WEBQA_DIR = join(DATA_DIR, "webqa")


# Where the GPV data is located, compatible with the GPV1 data
GPV_DIR = join(DATA_DIR, "gpv-dbg")
GPV_DATA_DIR = join(GPV_DIR, "learning_phase_data")
COCO_IMAGES = join(GPV_DATA_DIR, "coco/images")

# Stores cached data specific to gpv2
CACHE_DIR = join(dirname(dirname(__file__)), "data-cache")

# Stores image features
PRECOMPUTED_FEATURES_DIR = join(CACHE_DIR, "precomputed-features")

IMSITU_IMAGE_DIR = None
WEB_IMAGES_DIR = None
DCE = join(DATA_DIR, "dce")
DCE_IMAGES = join(DCE, "images")

VINVL_STATE = join(DATA_DIR, "vinvl", "R50C4_4setsvg_005000_model.roi_heads.score_thresh_0.2.pth")
NOCAPS_TEST_IMAGE_INFO = join(DCE, "nocaps_test_image_info.json")
NOCAPS_VAL_IMAGE_INFO = join(DCE, "nocaps_val_image_info.json")

if not exists(CACHE_DIR):
  mkdir(CACHE_DIR)
