# Neural Entity Typing with Knowledge Attention

This repo contains the source code and dataset for the following paper:

*   Ji Xin, Yankai Lin, Zhiyuan Liu, Maosong Sun. Improving Neural Fine-Grained Entity Typing with Knowledge Attention. *The 32nd AAAI Conference on Artificial Intelligence (AAAI 2018)* [pdf](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16321/16167).



## How to use our code for KNET

### Prerequisite

*   python 2.7.6
*   numpy >=1.13.3
*   tensorflow 0.12.1
    *   can be done by `pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl`

All the codes are tested under Ubuntu 16.04.

### Data

Data files should be put in the `data/` folder.

*   `disamb_file`, containing information for disambiguation, is already in `data/`. Please unzip it.
*   Train, valid and test set data are also in `data/`. Please unzip them.
*   For the word vector file, we recommend using Glove from http://nlp.stanford.edu/data/glove.840B.300d.zip . Please download, unzip, and put it in `data/`.
*   `types` records all they types in the taxonomy (only for recording; not used in the code).

### Parameters

*   Parameters saved from training is in the `parameter/` folder, but you can also choose a new location.
*   We provide parameters for the model shown in our paper in the `paper_parameter/` folder.

### Usage

Detailed usage can be found by running `python src/run.py --help`.

Quick start: simply run `./run.sh`.

For training and testing, follow the example of line 5 and 6 in `run.sh`.



## How to direclty use the code for typing

1. Organize input data in `.npy` format. See https://github.com/thunlp/KNET/issues/1 for instructions.

   Another example is in the `direct/` folder.

   * every sentence occupies three lines in `raw`. The first line is the entity mention, the second is left context, the third is right context. Words are separated with spaces.
   * run `raw2npy.py`. It's better to use the same python version with step 2 to avoid encoding issues.

2. Follow the example of line 7 in `run.sh`.
