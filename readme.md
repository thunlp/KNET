# Neural Entity Typing with Knowledge Attention

This repo contains the source code and dataset for the following paper:

*   Ji Xin, Yankai Lin, Zhiyuan Liu, Maosong Sun. Improving Neural Fine-Grained Entity Typing with Knowledge Attention. *The 32nd AAAI Conference on Artificial Intelligence (AAAI 2018)*.

## How to use our code for KNET

### Prerequisite

*   python 2.7.6
*   numpy >=1.13.3
*   tensorflow 0.12.1

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
