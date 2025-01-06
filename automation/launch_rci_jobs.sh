#!/bin/sh

#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=table_1
#SBATCH --err=jobstream/table_1.err
#SBATCH --out=jobstream/table_1.out
#SBATCH --time=12:00:00               # time limits: 12 hour

cd /home/vahidsae/Curriculum-FL
ulimit -n 2048

srun singularity run --nv ../nvidia-torch-cuda-cfl.sif \
        python3 main.py \
            --ntrials=3 \
            --seed=202207 \
            --rounds=100 \
            --num_users=100 \
            --frac=0.1 \
            --local_ep=10 \
            --local_bs=10 \
            --lr=0.001 \
            --momentum=0.9 \
            --glob_momentum=0.0 \
            --model='simple-cnn' \
            --dataset='cifar10' \
            --partition='homo' \
            --partition_difficulty_dist='rand' \
            --num_partitions=100 \
            --ordering='rand' \
            --pacing_f='step' \
            --pacing_a=1.0 \
            --pacing_b=1.0 \
            --client_ordering='rand' \
            --client_pacing_f='step' \
            --client_pacing_a=1.0 \
            --client_pacing_b=1.0 \
            --client_bs=10 \
            --exp_label='T2n' \
            --datadir='data/' \
            --logdir='save_results/' \
            --ptdir='pretrain/' \
            --train_expert='False' \
            --log_clientnet='False' \
            --data_score_sample_p=0.01 \
            --client_score_sample_n=1 \
            --log_filename='client100_rand_step_1.0a_1.0b_data_rand_step_1.0a_1.0b_gloss_p100' \
            --alg='fedavg_curr_lg_loss' \
            --beta=0.1 \
            --local_view=True  \
            --lg_scoring='G' \
            --noise=0 \
            --gpu=0 \
            --print_freq=10


# singularity run --nv ../nvidia-torch-cuda-cfl.sif bash -c pwd

#srun nvidia-smi
#srun pwd
#srun cuda_device_query