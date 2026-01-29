#!/bin/bash
#checkpointarray=( 10 20 30 40 )
checkpointarray=( 5 15 25 35 45 )
batchsizearray=( 64 )

device=1

name=eicu_mlp_1_0_forget_None_lr_0_001_bs_64_seed_1_projR_10_0

#initial training

#python3 main.py --dataset eicu --model mlp --dataroot ../data/eicu/ --epochs 50 --lr 0.001 --batch-size 64 --model-selection --plot --save-checkpoints --proj-radius 10 --estimate-constants

#selected epoch: 47

for bs in ${batchsizearray[@]}
do

#retrain from scratch
#python3 -u main.py  --dataset eicu --model mlp --dataroot ../data/eicu/ --epochs 48 --lr 0.001 --batch-size ${bs} --num-ids-forget 940 --seed 1 --proj-radius 10 --device ${device} --resume checkpoints/${name}_init.pt 

for checkpoint in ${checkpointarray[@]}
do
    E=$((48 - 1 - checkpoint))
    python3 -u main.py  --dataset eicu --model mlp --dataroot ../data/eicu/ --epochs ${E} --lr 0.001 --batch-size ${bs} --num-ids-forget 940 --seed 1 --proj-radius 10 --device ${device} --resume checkpoints/${name}_${checkpoint}.pt 
done

done

#sudo apt install bc
