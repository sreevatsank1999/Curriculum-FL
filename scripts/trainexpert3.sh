#!/bin/sh

python3 ../main_trainmodel.py \
    --seed=202207 \
    --batch_size=64 \
    --num_epoch=50 \
    --num_users=100 \
    --lr0=0.01 \
    --lr_sched_a=0.001 \
    --lr_sched_b=0.75 \
    --w_decay=5e-4 \
    --momentum=0.9 \
    --model='resnet-9' \
    --dataset='food101' \
    --exp_label='TrainExpert' \
    --datadir='../data/' \
    --logdir='../save_results/' \
    --ptdir='../pretrain/' \
    --log_filename='expert_training' \
    --alg='train_expert' \
    --gpu=3
