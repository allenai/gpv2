import logging
import os
import tempfile
import zipfile
from multiprocessing import Pool
from os import listdir, makedirs
from os.path import join, exists, dirname, relpath
from typing import List, Tuple

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

from gpv2.utils.py_utils import ensure_dir_exists


def download_to_file(url, output_file, pbar=False):
  """Download `url` to `output_file`"""
  logging.info(f"Downloading file from {url} to {output_file}")
  ensure_dir_exists(output_file)

  if not pbar:
    with requests.get(url) as r:
      r.raise_for_status()
      with open(output_file, 'wb') as f:
        f.write(r.content)
  else:
    with requests.get(url, stream=True) as r:
      r.raise_for_status()
      with open(output_file, 'wb') as f:
        _write_to_stream(r, f, True)


def download_zip(url, source, progress_bar=True):
  """Download zip file at `url` and extract to `source`"""
  # Download to a temp file to ensure we
  # don't eat a lot of RAM with downloading a large file
  with tempfile.TemporaryFile() as tmp_f:
    with requests.get(url, stream=True) as r:
      _write_to_stream(r, tmp_f, progress_bar)

    logging.info("Extracting to %s...." % source)
    makedirs(source, exist_ok=True)
    with zipfile.ZipFile(tmp_f) as f:
      f.extractall(source)


def _download(x):
  url, output_file = x
  os.makedirs(dirname(output_file), exist_ok=True)
  r = requests.get(url, allow_redirects=True)

  with tempfile.NamedTemporaryFile(delete=False) as tmp_f:
    # Hacky sanity check to help make sure we actually got an image
    try:
      header = r.content[:20].decode("utf-8").strip()
      if header.startswith("<html") or header.startswith("<?xml"):
        raise ValueError(f"Unexpected non-image response from {url}: {header}")
    except UnicodeError:
      pass
    tmp_f.write(r.content)
    tmp_f.close()
    os.rename(tmp_f.name, output_file)


def download_images(images: List[Tuple[str, str]], n_procs=10):
  with Pool(n_procs) as p:
    list(tqdm(p.imap(_download, images), total=len(images), ncols=100))


def download_s3_folder(bucket_name, s3_folder, local_dir):
  """
  From https://stackoverflow.com/questions/49772151/download-a-folder-from-s3-using-boto3

  Download the contents of a folder directory
  Args:
      bucket_name: the name of the s3 bucket
      s3_folder: the folder path in the s3 bucket
      local_dir: a relative or absolute directory path in the local file system
  """
  s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
  for obj in s3.list_objects(Bucket=bucket_name, Prefix=s3_folder)['Contents']:
    key = obj['Key']
    target = join(local_dir, relpath(key, s3_folder))
    if not exists(dirname(target)):
      makedirs(dirname(target), exist_ok=True)
    if key[-1] == '/':
      # Folder, skip
      continue
    s3.download_file(bucket_name, key, target)


def download_from_s3(bucket, key, out, progress_bar=True):
  """Download s3 file at `bucket`/`key` to `out`"""
  logging.info(f"Downloading {bucket}/{key} to {out}")
  s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
  if not exists(dirname(out)):
    makedirs(dirname(out))

  if progress_bar:
    object_size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    with tqdm(total=object_size, unit="b", unit_scale=True, desc="download", ncols=100) as pbar:
      s3.download_file(
        Bucket=bucket,
        Key=key,
        Filename=out,
        Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
      )
  else:
    s3.download_file(bucket, key, out)


def _write_to_stream(response, output_fh, progress_bar=True, chunk_size=32768):
  """Write streaming `response` to `output_fs` in chunks"""
  response.raise_for_status()
  if progress_bar:
    # tqdm does not format decimal numbers. We could in theory add decimal formatting
    # using the `bar_format` arg, but in practice doing so is finicky, in particular it
    # seems impossible to properly format the `rate` parameter. Instead we just manually
    # ensure the 'total' and 'n' values of the bar are rounded to the 10th decimal place
    content_len = response.headers.get("Content-Length")
    if content_len is not None:
      total = int(content_len)
    else:
      total = None
    pbar = tqdm(desc="downloading", total=total, ncols=100, unit="b", unit_scale=True)
  else:
    pbar = None

  cur_total = 0
  for chunk in response.iter_content(chunk_size=chunk_size):
    if chunk:  # filter out keep-alive new chunks
      if pbar is not None:
        cur_total += len(chunk)
        next_value = cur_total
        pbar.update(next_value - pbar.n)
      output_fh.write(chunk)

  if pbar is not None:
    if pbar.total is not None:
      pbar.update(pbar.total - pbar.n)
    pbar.close()