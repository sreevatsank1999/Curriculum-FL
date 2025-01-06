#!/bin/sh

dir='../save_results/fedavg_curr_lg_loss/cifar10'
if [ ! -e $dir ]; then
mkdir -p $dir
fi

python ../main.py \
--ntrials=3 \
--rounds=100 \
--num_users=100 \
--frac=0.1 \
--local_ep=10 \
--local_bs=10 \
--lr=0.01 \
--momentum=0.9 \
--model=resnet9 \
--dataset=cifar100 \
--partition='noniid-labeldir' \
--ordering='curr' \
--pacing_f='linear' \
--pacing_a=0.8 \
--pacing_b=0.2 \
--datadir='/datasets/' \
--logdir='../save_results/' \
--log_filename='curr_linear_0.8a_0.2b_0.9beta' \
--alg='fedavg_curr_lg_loss' \
--beta=0.9 \
--local_view \
--noise=0 \
--gpu=1 \
--print_freq=50