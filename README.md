# Webly Supervised Concept Expansion for General Purpose Vision Models
This is the codebase for [GPV 2](https://prior.allenai.org/projects/gpv2) from our paper [Webly Supervised Concept Expansion for General Purpose Vision Models](http://arxiv.org/abs/2202.02317).
Code for the web data is in a [separate repo](https://prior.allenai.org/projects/gpv2). 

# Installation
## Code
Clone the repo with --recurse-submodules
```
git clone git@github.com:allenai/gpv2.git --recurse-submodules
```

Create conda environment
```
conda create -n gpv2 python=3.6 -y
conda activate gpv2
```

Install [pytorch](https://pytorch.org/), I have been using pytorch 1.8.1, 
other versions might work but are not tested. For example on linux:

```
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.2 -c pytorch -c conda-forge
```

but you might need to change that command depending on your operating system/gpu setup.

Finally, install libraries:
```bash
conda install -c cyclus java-jdk=8.45.14 -y  
pip3 install -r requirements.txt
```
 
## Data
Download data for COCO, DCE, and Web as well as the pre-computed VinVL features for these datasets
(note you cannot use the VinVL features provided by the VinVL authors since we need features 
for the cropped images and for the all-image boxes):
```
python gpv2/download_data.py 
```

The data is saved in the locations found in `file_paths.py`, by default source data is saved 
into ~/data/gpv while the features are stored in ./data-cache/precomputed-features/vinvl.
The command lines args for the script can download particular subsets if you don't need
everything.

## Models
We have currently released GPV-2 trained with and without Web data:


- With web: s3://ai2-prior-gpv/public/gpv2-models/gpv2
- Without web: s3://ai2-prior-gpv/public/gpv2-models/gpv2-noweb

To download, use aws s3 cp with --recursive:

```
mkdir -p models
aws s3 cp --recursive s3://ai2-prior-gpv/public/gpv2-models/gpv2 models/gpv2
```

# Training
The repo is currently setup to train the basic model on COCO data, training with web data will be 
added we complete the release process.

To train on devices 0 and 1 of your machine without web data:

```
python exp/ours/experiments/train_t5.py --device 0 1 --num_workers 3 --task gpv2 --output_dir /path/to/output/dir
```

For debugging purposes I recommend using the --debug flag and reducing the number of devices and 
workers to 0 which will get you much faster startup times and better error messages:

```
python exp/ours/experiments/train_t5.py --device 0 --num_workers 0 --task gpv2 --output_dir /path/to/output/dir --debug small
```

which will run the model on a small sample of the data and without complicated distributed training.

# Eval

## Single Image
Run on a single image using `run_on_image_id` 

```
python gpv2/eval/run_on_image_id.py model/gpv2 dce/test/nocaps/0003d84e0165d630.jpg "What is this?"
```

Here "What is this?" is the prompt and `dce/test/nocaps/0003d84e0165d630.jpg` is an image_id, _not a filepath_, that 
can be used to look up the needed VinVL features in the HDF5 feature files.
Look at `GpvDataset` or `DceDataset` to see the format of the image_ids.

## For a Dataset 
To compute predictions for a dataset, use:

```
python gpv2/eval/compute_topn_predictions.py models/gpv2 --datasets dce-vqa --part val --eval --output_name default
```

The predictions for VQA will saved to models/gpv2/r0/eval/{dataest-name}--default, an 
evaluation file with the results will be saved there as eval.json. 

See the command line flags on compute_topn_predictions to run on other datasets, or 
use multiple GPUs.


## Test server evaluations
The script `gpv2/eval/build_sumbmission_files.py` will construct the submissions files needs to evaluate on the VQA test, COCO test 
and nocaps val/test server assuming the needed predictions for those models have already been saved
using `compute_topn_predictions.py`.


# Precomputing features for new images
GPV-2 uses VinVL pre-computed image features.
If you want to run the model on a new dataset, you will need to pre-computed the image features
for that dataset. We provide our script for doing this and getting results in a HDF5 file we
can use, the results are compatibile with the ones produced by [here](https://github.com/microsoft/scene_graph_benchmark#vinvl-feature-extraction).
There are three steps to doing this:

1. Gather your images into one directory, it may include subdirectories, but it should not contain any
   files other than images.
2. Run:

    ```
    python gpv2/build_image_features/precompute_image_features.py /path/to/image_directory your_dataset_name --output features.hdf5
    ```
   where `/path/to/image_directory` should point to your image directory and `your_dataset_name` should
   be a name for the set of images you are adding. The script has parameters to control the batch size and run across multiple devices
   which can be used to tune the process. This will
   produce the hdf5 file vinvl.hdf5.

3. Move the hdf5 file to `file_paths.PRECOMPUTED_FEATURES_DIR` under the vinvl directory, for example:

    ```
    mkdir -p data-cache/precomputed-features/vinvl
    mv features.hdf5 data-cache/precomputed-features/your_dataset_name/your_dataset_name.hdf5
    ```

Now the model will support image_ids with the format of `your_dataset_name/path/to/image_file/in/your/directory`.
For example, if your directory contained the
image val/dog/001.jpg and your dataset_name was "pets", the image_id "pets/val/001.jpg" will
now be recognized by the model and load the pre-computed features for that image. Image ids of that format
can now be passed to`run_on_image_id.py` or used in `GPVExample` objects with VinVL models.

Features for the web/coco/dce datasets can be re-computed using gpv2/build_image_features/precompute_dataset_features.py,
but by default download_data will download them automatically.
