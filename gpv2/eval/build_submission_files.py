"""Script that converts predictions in our format to the test-server format for VQA and captioning"""
import argparse
import json
import logging
from os import listdir
from os.path import join, exists

from gpv2 import file_paths
from gpv2.data.dataset import Task
from gpv2.data.dce_dataset import DceDataset
from gpv2.data.gpv_datasets import GpvDataset
from gpv2.train.runner import load_gpv_predictions
from gpv2.utils import py_utils
from gpv2.utils.py_utils import load_json_object


class SubmissionFileBuilder:

  def build(self, prediction_dir):
    raise NotImplementedError()


class NoCapsSubmissionBuilder(SubmissionFileBuilder):
  def __init__(self, split):
    self.dataset = DceDataset(Task.CAPTIONING, split)
    self.image_id_to_id = None
    self.instances = None

  def build(self, prediction_dir):
    if self.image_id_to_id is None:
      src = file_paths.NOCAPS_TEST_IMAGE_INFO if self.dataset.part == "test" else file_paths.NOCAPS_VAL_IMAGE_INFO
      image_info = load_json_object(src)['images']
      image_id_to_id = {info['open_images_id']:info['id'] for info in image_info}
      self.image_id_to_id = image_id_to_id
      self.instances = self.dataset.load()

    out = []
    predictions = load_gpv_predictions(prediction_dir)
    for instance in self.instances:
      out.append(dict(
        image_id=self.image_id_to_id[instance.image_id.split("/")[-1].split(".")[0]],
        caption=predictions[instance.get_gpv_id()].text[0]
      ))
    return out


class CocoCaptionSubmissionBuilder(SubmissionFileBuilder):
  def __init__(self, part):
    self.dataset = GpvDataset(Task.CAPTIONING, part, False)
    self._id_map = None

  def build(self, prediction_dir):
    predictions = load_gpv_predictions(prediction_dir)

    out = []
    for gpv_id, pred in predictions.items():
      assert gpv_id.startswith("coco-image-cap")
      image_id = int(gpv_id[len("coco-image-cap"):])
      out.append(dict(
        image_id=image_id,
        caption=pred.text[0]
      ))
    return out


class VqaSubmissionBuilder(SubmissionFileBuilder):
  def __init__(self):
    self.dataset = GpvDataset(Task.VQA, "test", False)
    self.image_id_to_id = None
    self.instances = None

  def build(self, prediction_dir):
    predictions = load_gpv_predictions(prediction_dir)
    out = []
    for gpv_id, pred in predictions.items():
      assert gpv_id.startswith("vqa")
      out.append(dict(
        question_id=int(gpv_id[3:]),
        answer=pred.text[0]
      ))
    return out


def main():
  parser = argparse.ArgumentParser(description="Build test-server submission files")
  parser.add_argument("model", help="Model or run directory to use")
  parser.add_argument("--eval_name", default=None, help="Eval name to use")
  args = parser.parse_args()

  models = py_utils.find_models(args.model)
  if len(models) == 0:
    logging.info("No models found")
    return

  builders = [
    NoCapsSubmissionBuilder("test"),
    NoCapsSubmissionBuilder("val"),
    VqaSubmissionBuilder(),
    CocoCaptionSubmissionBuilder("test"),
    CocoCaptionSubmissionBuilder("val")
  ]

  for model_name, (model_dir, runs) in models.items():
    for run in runs:
      for builder in builders:
        if args.eval_name is None:
          candidates = [x for x in listdir(join(run, "eval"))
                        if x.startswith(builder.dataset.get_name())]
          if len(candidates) == 0:
            continue
          elif len(candidates) == 1:
            target_dir = join(run, "eval", candidates[0])
          else:
            raise ValueError()
        else:
          eval_name = builder.dataset.get_name() + "--" + args.eval_name
          target_dir = join(run, "eval", eval_name)
          if not exists(target_dir):
            continue

        target_file = join(target_dir, "submission.json")
        if exists(target_file):
          logging.info(f"Skip model {model_name}/{eval_name}, already has submission file {target_file}")
          continue

        logging.info(f"Building submission file {target_file}")
        out = builder.build(target_dir)
        with open(target_file, "w") as f:
          json.dump(out, f, indent=2)


if __name__ == '__main__':
  py_utils.add_stdout_logger()
  main()