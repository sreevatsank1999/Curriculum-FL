import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()

from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

import os
from os.path import isfile, join, isdir

import numpy as np
from scipy.signal import savgol_filter


def get_trace(path,smooth=0):
    
    trace=[[]];
    # trace_sm=[[]];
    
    for trial in [0,1,2]:
        path_t=f'{path}/Trial{trial}_Global_test_accuracy_Accuracy';
        try: 
            files = [f for f in os.listdir(path_t) if isfile(join(path_t, f))];
        except: 
            continue;
        
        event_file = files[0];
        events = LegacyEventFileLoader(f'{path_t}/{event_file}').Load();
        trace_name=f'Trial{trial}/Global/test_accuracy';
        for e in events:
            for v in e.summary.value:
                if v.tag == trace_name:
                    # print(e.step, v.simple_value)
                    trace[trial].append(v.simple_value)
        
        # trace_sm[trial] = savgol_filter(trace[trial], smooth, 3) # window size, polynomial order 3    
        trace.append([]);
        # trace_sm.append([]);
    
    
    trace_mean,trace_std = calc_mean_std(trace);

    trace_mean = savgol_filter(trace_mean, smooth,5);
    trace_std = savgol_filter(trace_std, smooth,5);
    
    return trace_mean,trace_std;

def calc_mean_std(trace):
    trace_mean=[];trace_std=[];
    if len(trace) ==3:
        trace[2] = [];
    min_rounds = np.min([len(trace[0]), len(trace[1]), len(trace[2])]);
    argmin_rounds = np.argmin([len(trace[0]), len(trace[1]), len(trace[2])]);
    
    if min_rounds < 80:
        # indx = [True]*3;
        # indx[argmin_rounds] = False;
        # trace_mean[argmin_rounds] = [np.mean([trace_mean[i] for i in np.where(indx)]) for j in range];
        # trace_std[argmin_rounds] = [np.mean([trace_std[i] for i in np.where(indx)]) for j in range];
        _trace_mean=[];
        min_rounds = np.min([len(trace[0]), len(trace[1])]);
        for r in range(min_rounds):
            point=[trace[0][r],trace[1][r]];
            _trace_mean.append(np.mean(point));
        trace[argmin_rounds] = _trace_mean;
    
    min_rounds = np.min([len(trace[0]), len(trace[1]), len(trace[2])]);
    
    for r in range(min_rounds):
        point=[trace[0][r],trace[1][r],trace[2][r]];
        trace_mean.append(np.mean(point));
        trace_std.append(np.std(point));
        
    return np.array(trace_mean),np.array(trace_std);


# plt.tick_params(labelsize=14);
plt.figure(figsize=(4,3))

base_path='save_results'

# for dist in ["IID", "Dir02", "Dir005"]:
for dist in ["Dir02"]:
    dist_dir='homo' if dist == "IID" else 'noniid-labeldir';
    for cc in ["curr", "rand"]:
    # for cc in ["anti"]:
        for dc in ["curr", "rand"]:
        # for dc in ["anti"]:
            # for partdiff in ["inc", "rand"]:
            for partdiff in ["inc"]:
                # for loss in ["E","G"]:
                for loss in ["E"]:
                    # tbpath = f"{base_path}/CurrClient_{partdiff}_{loss}loss_{dist}/fedavg_curr_lg_loss/cifar10/{dist_dir}/simple-cnn";
                    # tblogdir = f"tblog_client100round100_{cc}_linear_0.8a_0.2b_data_{dc}_linear_0.8a_0.2b_{loss}loss_p100"
                    tbpath = f"{base_path}/Ablation/fedprox_curr/cifar10/{dist_dir}/simple-cnn";
                    if dc == "rand":
                        dc_s = "rand_step_1.0a_1.0b";
                    else:
                        dc_s = f'{dc}_linear_0.8a_0.2b';
                    if cc == "rand":
                        cc_s = "rand_step_1.0a_1.0b";
                    else:
                        cc_s = f'{cc}_linear_0.8a_0.2b';
                        
                    tblogdir = f"tblog_client100round100_{cc_s}_data_{dc_s}_{loss}loss_p0.9{partdiff}100_beta0.2"
                        
                    for d in os.listdir(tbpath):
                        if isdir(join(tbpath, d)):
                            if tblogdir in d:
                                tblogdir=d;
                                break;
                    
                    t_m,t_v = get_trace(f"{tbpath}/{tblogdir}",smooth=25);
                    
                    r=[i for i in range(len(t_m))];
                    
                    if dist == "Dir02":
                        _dist = "Dir(0.2)";
                    elif dist == "Dir005":
                        _dist = "Dir(0.05)";
                    else:
                        _dist= "IID";
                    
                    plt.plot(r, t_m, linewidth=3.0 ,label=f'{partdiff}-{loss}-{cc}-{dc}');
                    plt.xlim([0,100]);
                    plt.xlabel("rounds");
                    plt.ylabel("Accuracy");
                    plt.fill_between(r, t_m - t_v/2, t_m + t_v/2,alpha=0.2);
                    
                    plt.show();
plt.legend();
plt.savefig('test.png',dpi=300, bbox_inches="tight")
                    
                    
                    
                    