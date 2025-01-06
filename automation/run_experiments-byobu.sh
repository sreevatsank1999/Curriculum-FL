# Automation script to deploy model training on a set of hyperparameters varying on a 2D grid 
# TODO: Port to python for much more robust scripting 

session_name="CFL"
cfl_container_name="nvidia-torch-cuda-cfl_$(id -un)"
append=0
vanilla=0

print_help() {
            echo "Usage: ./run_experiments-byobu.sh [-s <session_name>] [-c <cfl_container_name>] [-a][-v]  "
            echo "         -a \tAppend new experiments if session already exists"
            echo "         -v \tInclude Vanilla run"
            echo "Default: "
            echo "        session_name='CFL'"
            echo "        cfl_container_name='nvidia-torch-cuda-cfl_$(id -un)'"
}

while getopts hs:c:av flag
do
    case "${flag}" in
        s) session_name=${OPTARG};;
        c) cfl_container_name=${OPTARG};;
        a) append=1;;
        v) vanilla=1;;
        h) 
            print_help; 
            exit 0;
        ;; 
        *)
            print_help;
            exit -1;
        ;;
    esac
done

echo ""
echo "*********************************************************************************************"
echo "session_name: $session_name";
echo "cfl_container_name: $cfl_container_name";
echo "vanilla: $vanilla";
echo "*********************************************************************************************"
echo ""

byobu has-session -t "$session_name"
if [ $? == 0 ]; then
    echo "Byobu Session: $session_name, Already exists"    
    if [ $append != 0 ]; then
        echo "Append: $append"
        echo "Appending to active session ..."
    else
        echo "use the -a flag to append to exisitng session"
        exit 2;
    fi
else 
    echo "Start new Byobu Session"
    byobu new-session -d -s "$session_name"

    echo "Start Byobu window 'ControlPanel' Tab"
    byobu new-window -t "$session_name" -n "ControlPanel"
    byobu-tmux select-pane -t 0
    byobu-tmux split-window -h

    byobu-tmux select-pane -t 0
    byobu-tmux send-keys "nvtop" Enter
    byobu-tmux select-pane -t 1
    byobu-tmux send-keys "htop" Enter
fi

# Hyperparameters listed below, 
# Any set of two array parameters can be used as the 2D grid axes for the hyperparameter search space 

data_ordering=('curr' 'anti' 'rand')
# data_ordering=('curr')
data_pacing_f='linear'
data_pacing_a=0.8
data_pacing_b=0.2

# data_ordering=('rand')
# data_pacing_f='step'
# data_pacing_a=1.0
# data_pacing_b=1.0


# client_ordering=('curr' 'anti' 'rand')
# client_ordering=('curr')
# client_pacing_f='linear'
# client_pacing_a=0.8
# client_pacing_b=0.2

client_ordering=('rand')
client_pacing_f='step'
client_pacing_a=1.0
client_pacing_b=1.0

# dir_beta=(0.05 0.2 0.9)
dir_beta=(0.05 0.2 0.9)

pdist='rand'
# porder_frac=(0.0 0.5 0.9 1.0)
porder_frac=(0.0)

lg_scoring='G'

num_round=100    
num_users=100
nbpartition=100

dataset='pathmnist'
model='simple-cnn'
gpu=0

# Use the array variables for the range of these for loops. 
# Note: Needs invasive modifications to commands below, to ensure correct use of the range variable (╥﹏╥)
for exp_id in $(seq 0 $((${#dir_beta[@]}-1)))
do
    # exp_name="C${num_users}-_L(${client_pacing_a},${client_pacing_b[$exp_id]})-RS(${data_pacing_a},${data_pacing_b})-${lg_scoring}loss"
    exp_name="Dir(${dir_beta[$exp_id]})"

    echo "Start Byobu window \'$exp_name\' Tab"
    byobu new-window -t "$session_name" -n "$exp_name"
    byobu-tmux select-window -t "$session_name"
    byobu-tmux select-pane -t 0
    byobu-tmux split-window -h
    byobu-tmux select-pane -t 0
    byobu-tmux split-window -v
    byobu-tmux select-pane -t 2
    byobu-tmux split-window -v

    for _i in $(seq 0 $((${#data_ordering[@]}-1)))
    do
        echo "Start Exp \'${data_ordering[$_i]}\' ..."
        byobu-tmux select-pane -t $_i
        byobu-tmux send-keys "docker exec -it $cfl_container_name /bin/bash" Enter
        sleep 0.1
        byobu-tmux send-keys \
            "cd ../ && python main.py \
                --ntrials=3 \
                --seed=202207 \
                --rounds=${num_round} \
                --num_users=${num_users} \
                --frac=0.1 \
                --local_ep=10 \
                --local_bs=10 \
                --lr0=0.01 \
                --lr_sched_a=0.001 \
                --lr_sched_b=0.75 \
                --w_decay=5e-4 \
                --momentum=0.9 \
                --glob_momentum=0.0 \
                --model="$model" \
                --dataset="${dataset}" \
                --partition='noniid-labeldir' \
                --partition_difficulty_dist="$pdist" \
                --partition_ordering_f=${porder_frac[0]} \
                --num_partitions=$nbpartition \
                --ordering="${data_ordering[$_i]}" \
                --pacing_f="$data_pacing_f" \
                --pacing_a=$data_pacing_a \
                --pacing_b=${data_pacing_b} \
                --client_ordering="${client_ordering[0]}" \
                --client_pacing_f="$client_pacing_f" \
                --client_pacing_a=${client_pacing_a} \
                --client_pacing_b=${client_pacing_b} \
                --client_bs=10 \
                --exp_label="$session_name" \
                --datadir='data/' \
                --logdir='save_results/' \
                --ptdir='pretrain/' \
                --train_expert='False' \
                --log_clientnet='False' \
                --data_score_sample_p=0.00001 \
                --client_score_sample_n=0 \
                --log_filename="client${num_users}round${num_round}_${client_ordering[0]}_${client_pacing_f}_${client_pacing_a}a_${client_pacing_b}b_data_${data_ordering[$_i]}_${data_pacing_f}_${data_pacing_a}a_${data_pacing_b}b_${lg_scoring}loss_p${porder_frac[0]}${pdist}${nbpartition}_beta${dir_beta[$exp_id]}" \
                --alg='fedavg_curr_lg_loss' \
                --beta="${dir_beta[$exp_id]}" \
                --local_view=True  \
                --lg_scoring='${lg_scoring}' \
                --noise=0 \
                --gpu=${gpu} \
                --print_freq=10 \
            "\
            Enter
    done

    if [[ $vanilla ]]; then
        echo "Start Exp \'vanilla\' ..."
        byobu-tmux select-pane -t 3
        byobu-tmux send-keys "docker exec -it $cfl_container_name /bin/bash" Enter
        sleep 0.1
        byobu-tmux send-keys \
            "cd ../ && python main.py \
                --ntrials=3 \
                --seed=202207 \
                --rounds=${num_round} \
                --num_users=${num_users} \
                --frac=0.1 \
                --local_ep=10 \
                --local_bs=10 \
                --lr0=0.01 \
                --lr_sched_a=0.001 \
                --lr_sched_b=0.75 \
                --w_decay=5e-4 \
                --momentum=0.9 \
                --glob_momentum=0.0 \
                --model="$model" \
                --dataset="${dataset}" \
                --partition='noniid-labeldir' \
                --partition_difficulty_dist="$pdist" \
                --partition_ordering_f=${porder_frac[0]} \
                --num_partitions=$nbpartition \
                --ordering="rand" \
                --pacing_f="step" \
                --pacing_a=1.0 \
                --pacing_b=1.0 \
                --client_ordering="rand" \
                --client_pacing_f="step" \
                --client_pacing_a=1.0 \
                --client_pacing_b=1.0 \
                --client_bs=10 \
                --exp_label="$session_name" \
                --datadir='data/' \
                --logdir='save_results/' \
                --ptdir='pretrain/' \
                --train_expert='False' \
                --log_clientnet='False' \
                --data_score_sample_p=0.00001 \
                --client_score_sample_n=0 \
                --log_filename="client${num_users}round${num_round}_rand_step_1.0a_1.0b_data_rand_step_1.0a_1.0b_${lg_scoring}loss_p${porder_frac[0]}${pdist}${nbpartition}_beta${dir_beta[$exp_id]}" \
                --alg='fedavg_curr_lg_loss' \
                --beta="${dir_beta[$exp_id]}" \
                --local_view=True  \
                --lg_scoring='${lg_scoring}' \
                --noise=0 \
                --gpu=${gpu} \
                --print_freq=10 \
            "\
            Enter
    fi
done 