# R2D2


## Overview
This repository contains the code and scripts used for the experiments in our paper, "Descend or Rewind? Stochastic Gradient Descent Unlearning."


### Prerequisites
Package versions are included in `packages.txt`

## Datasets
Download the MAAD-Face annotations here: https://github.com/pterhoer/MAAD-Face

MAAD-Face preprocessing procedure and Lacuna-100 generation from VGGFace2 are in `lacuna_maad_preprocessing_binary.ipynb`

To access the eICU dataset, follow the instructions here: https://eicu-crd.mit.edu/gettingstarted/access/

## Training the Model
To train the model from scratch, run:
```sh
python3 main.py --dataset eicu --model mlp --dataroot ../data/eicu/ --epochs 50 --lr 0.001 --batch-size 64 --model-selection --plot --save-checkpoints --proj-radius 10 --estimate-constants
```

## R2D Experiments
To rerun the R2D checkpointing experiments, run the bash scripts `checkpointbash_eicu.sh` and `checkpointbash_lacuna.sh`.

To rerun the D2D checkpointing experiments, run the bash scripts `eicu_d2d.sh` and `lacuna_d2d.sh`.

