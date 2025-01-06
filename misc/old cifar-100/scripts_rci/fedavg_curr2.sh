#!/bin/sh
#SBATCH --time=70:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --job-name=fedavg_curr_cifar10
#SBATCH --err=results/fedavg_curr_cifar10.err
#SBATCH --out=results/fedavg_curr_cifar10.out

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
ml PyTorch/1.8.0-fosscuda-2019b-Python-3.7.4
ml torchvision/0.9.1-fosscuda-2019b-PyTorch-1.8.0
ml scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4

dir='../save_results/fedavg_curr/G-Loss/cifar10'
if [ ! -e $dir ]; then
mkdir -p $dir
fi

for a in 0.01 0.2 0.8 1.6
do
    for b in 0.0025 0.2 0.8
    do
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
        --ordering='curr' \
        --pacing_f='log' \
        --pacing_a=$a \
        --pacing_b=$b \
        --datadir='../../data/' \
        --logdir='../save_results/' \
        --log_filename='curr_log_'$a'a_'$b'b' \
        --alg='fedavg_curr' \
        --beta=0.5 \
        --local_view \
        --noise=0 \
        --gpu=0 \
        --print_freq=10
    done
done
