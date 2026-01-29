#!/bin/bash

#checkpointarray=( 10 20 30 40 )
checkpointarray=( 5 10 15 20 25 30 35 40 )
bs=64


for checkpoint in ${checkpointarray[@]}
do
    E=$((43 - checkpoint))
    python3 -u main.py  --dataset lacuna100binary128 --model resnetsmooth --dataroot ../data/lacuna100binary128/ --epochs ${E} --lr 0.01 --batch-size ${bs} --num-ids-forget 2 --seed 1 --proj-radius 50 --resume checkpoints/lacuna100binary128_resnetsmooth_1_0_forget_None_lr_0_01_bs_${bs}_seed_1_projR_50_0_selected.pt --device 2
done
