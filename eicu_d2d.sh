#!/bin/bash
#checkpointarray=( 10 20 30 40 )
checkpointarray=( 15 25 35 45 )
batchsizearray=( 64 )

device=3

name=eicu_mlp_1_0_forget_None_lr_0_001_bs_64_seed_1_projR_10_0



for bs in ${batchsizearray[@]}
do


for checkpoint in ${checkpointarray[@]}
do
    E=$((48 - 1 - checkpoint))
    python3 -u main.py  --dataset eicu --model mlp --dataroot ../data/eicu/ --epochs ${E} --lr 0.001 --batch-size ${bs} --num-ids-forget 940 --seed 1 --proj-radius 10 --device ${device} --resume checkpoints/${name}_selected.pt 
done

done

#sudo apt install bc
