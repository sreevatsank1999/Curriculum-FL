#!/bin/bash
# Automation script to deploy model training on a set of hyperparameters varying on a 2D grid 
# TODO: Port to python for much more robust scripting 

echo "Launching Jobs"
echo "---------------------------------------------------------------------------------------------"

print_help(){
    echo -e "Usage: ./$0 [-t <tableID>] [-d]"
    echo -e "\t\t -t \t tableID: \tShort phrase to identify table name, value is prepend to the job_name"
    echo -e "\t\t -d \t dryrun: \tRun Helm in dryrun mode"
    echo
}

tableID="default"
dryrun=""
nshift=0;
while getopts t:hd flag
do
    case "${flag}" in
        t) tableID="${OPTARG}"
            ((nshift=nshift+1));;
        d) dryrun="--dry-run";;
        *)
            print_help;
            exit -1;
        ;;
    esac
    ((nshift=nshift+1));
done
shift $nshift

echo "tableID: $tableID"
echo "dryrun: $dryrun"
echo 

# Hyperparameters listed below, 
# Any set of two array parameters can be used as the 2D grid axes for the hyperparameter search space 

data_ordering=("curr" "anti" "rand")
data_pacing_f="linear"
data_pacing_a=0.8
data_pacing_b=0.2

client_ordering=("curr" "anti" "rand")
client_pacing_f="linear"
client_pacing_a=0.8
client_pacing_b=0.2

lg_scoring="E"

num_round=100
num_users=100
nbpartition=100

# Use the array variables for the range of these for loops. 
# Note: Needs invasive modifications to commands below, to ensure correct use of the range variable (╥﹏╥)
for exp_id in $(seq 0 $((${#client_ordering[@]}-1)))
do
    exp_name="C${num_users}R${num_round}-_L(${client_pacing_a},${client_pacing_b})-_L(${data_pacing_a},${data_pacing_b})-${lg_scoring}loss"

    echo "Launching Experiment $exp_name "
    
    for _i in $(seq 0 $((${#data_ordering[@]}-1)))
    do
        echo "Start Exp ${data_ordering[$_i]} ..."
        job_name="${tableID}.$exp_id.$_i"
        
        helm install $job_name chart/Curriculum-FL/ $dryrun \
            --set jobParam.name="$job_name" \
            --set jobParam.description="experimentName: $(printf '%q' "$exp_name")" \
            --set credentialSecret="credential-skadaveru" \
            --set code.directory="~/Projects/Curriculum-FL" \
            \
            --set cfl.ntrials=3 \
            --set cfl.seed=202207 \
            --set cfl.rounds=${num_round} \
            --set cfl.num_users=${num_users} \
            --set cfl.frac=0.1 \
            --set cfl.local_ep=10 \
            --set cfl.local_bs=10 \
            --set cfl.lr=0.001 \
            --set cfl.momentum=0.9 \
            --set cfl.glob_momentum=0.0 \
            --set cfl.model="resnet-9" \
            --set cfl.dataset="food101" \
            --set cfl.partition="noniid-labeldir" \
            --set cfl.partition_difficulty_dist="rand" \
            --set cfl.num_partitions=$nbpartition \
            --set cfl.ordering="${data_ordering[$_i]}" \
            --set cfl.pacing_f="$data_pacing_f" \
            --set cfl.pacing_a=$data_pacing_a \
            --set cfl.pacing_b=${data_pacing_b} \
            --set cfl.client_ordering="${client_ordering[$exp_id]}" \
            --set cfl.client_pacing_f="$client_pacing_f" \
            --set cfl.client_pacing_a=${client_pacing_a} \
            --set cfl.client_pacing_b=${client_pacing_b} \
            --set cfl.client_bs=10 \
            --set cfl.exp_label="$job_name" \
            --set cfl.datadir="data/" \
            --set cfl.logdir="save_results/" \
            --set cfl.ptdir="pretrain/" \
            --set cfl.train_expert="True" \
            --set cfl.log_clientnet="False" \
            --set cfl.data_score_sample_p=0.005 \
            --set cfl.client_score_sample_n=2 \
            --set cfl.log_filename="client${num_users}round${num_round}_${client_ordering[$exp_id]}_${client_pacing_f}_${client_pacing_a}a_${client_pacing_b}b_data_${data_ordering[$_i]}_${data_pacing_f}_${data_pacing_a}a_${data_pacing_b}b_${lg_scoring}loss_p${nbpartition}" \
            --set cfl.alg="fedavg_curr_lg_loss" \
            --set cfl.beta=0.05 \
            --set cfl.local_view=True  \
            --set cfl.lg_scoring="${lg_scoring}" \
            --set cfl.noise=0 \
            --set cfl.gpu=0 \
            --set cfl.print_freq=10 \
            \
            --set "requiredGPU={NVIDIA-GeForce-RTX-3090,NVIDIA-TITAN-RTX,NVIDIA-RTX-A5000,Quadro-RTX-6000,Tesla-V100-SXM2-32GB,NVIDIA-A40,NVIDIA-RTX-A6000,Quadro-RTX-8000}"

    done
done 