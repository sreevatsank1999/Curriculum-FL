#!/bin/sh

dir='../save_results/fedavg/cifar10'
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
--model=simple-cnn \
--dataset=cifar10 \
--partition='noniid-#label2' \
--datadir='../../data/' \
--logdir='../save_results/' \
--log_filename='3' \
--alg='fedavg' \
--beta=0.5 \
--local_view \
--noise=0 \
--gpu=0 \
--print_freq=10