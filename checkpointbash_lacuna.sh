#!/bin/bash

#checkpointarray=( 5 15 25 35 )
checkpointarray=( 35 )
bs=64


#python3 main.py  --dataset lacuna100binary128 --model resnetsmooth --dataroot ../data/lacuna100binary128/ --epochs 44 --lr 0.01 --batch-size ${bs} --num-ids-forget 2 --seed 1 --proj-radius 50 --resume checkpoints/lacuna100binary128_resnetsmooth_1_0_forget_None_lr_0_01_bs_${bs}_seed_1_projR_50_0_init.pt 

for checkpoint in ${checkpointarray[@]}
do
    E=$((43 - checkpoint))
    python3 -u main.py  --dataset lacuna100binary128 --model resnetsmooth --dataroot ../data/lacuna100binary128/ --epochs ${E} --lr 0.01 --batch-size ${bs} --num-ids-forget 2 --seed 1 --proj-radius 50 --resume checkpoints/lacuna100binary128_resnetsmooth_1_0_forget_None_lr_0_01_bs_${bs}_seed_1_projR_50_0_${checkpoint}.pt 
done

#sudo apt install bc
#python3 main.py  --dataset lacuna100binary128 --model resnetsmooth --dataroot ../data/lacuna100binary128/ --epochs 50 --lr 0.01 --batch-size 64 --model-selection --plot --save-checkpoints --seed 1 --proj-radius 50 --estimate-constants
#selected epoch 43
