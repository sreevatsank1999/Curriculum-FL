import numpy as np

import copy
import os 
import gc 
import pickle
import time 
import sys
import datetime
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data import *
from src.models import *
from src.client import * 
from src.clustering import *
from src.utils import * 
from src.benchmarks import *

if __name__ == '__main__':
    print('-'*40)
    
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu) ## Setting cuda on GPU 
    path = args.logdir + args.exp_label + '/' + args.alg +'/' + args.dataset + '/' + args.partition + '/' + args.model + '/'
    mkdirs(path)
    
    if args.log_filename is None: 
        filename='logs_%s.txt' % datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    else:
        filename='logs_'+args.log_filename+'.txt'  

    sys.stdout = Logger(fname=path+filename)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    fname=path+filename
    fname=fname[0:-4]
    if args.alg == 'solo':
        alg_name = 'SOLO'
        run_solo(args, fname=fname)
    elif args.alg == 'fedavg':
        alg_name = 'FedAvg'
        run_fedavg(args, fname=fname)
    elif args.alg == 'fedprox_curr':
        alg_name = 'FedProx_Curr'
        run_fedprox_curr(args, fname=fname)
    elif args.alg == 'pfedme_curr':
        alg_name = 'pFedMe_Curr'
        run_pfedme_curr(args, fname=fname)
    elif args.alg == 'fednova_curr':
        alg_name = 'FedNova_Curr'
        run_fednova_curr(args, fname=fname)
    elif args.alg == 'scaffold_curr':
        alg_name = 'Scaffold_Curr'
        run_scaffold_curr(args, fname=fname)
    elif args.alg == 'lg':
        alg_name = 'LG'
        run_lg(args, fname=fname)
    elif args.alg == 'per_fedavg':
        alg_name = 'Per-FedAvg'
        run_per_fedavg(args, fname=fname)
    elif args.alg == 'ifca':
        alg_name = 'IFCA'
        run_ifca(args, fname=fname)
    elif args.alg == 'cfl':
        alg_name = 'CFL'
        run_cfl(args, fname=fname)
    elif args.alg == 'fedavg_curr':
        alg_name = 'FedAvg_Curr'
        run_fedavg_curr(args, fname=fname)
    elif args.alg == 'fedavg_curr_lg_loss':
        alg_name = 'FedAvg_Curr_LG_Loss'
        run_fedavg_curr_lg_loss(args, fname=fname)
    elif args.alg == 'fedavg_curr_lg_pred':
        alg_name = 'FedAvg_Curr_LG_Pred'
        run_fedavg_curr_lg_pred(args, fname=fname)
    elif args.alg == 'partition_distribution':
        alg_name = 'PartitionDistribution'
        run_partition_distribution(args, fname=fname)
    elif args.alg == 'noise_experiment':
        alg_name = 'NoiseExp'
        run_noise_experiment(args, fname=fname)
    else: 
        print('Algorithm Does Not Exist')
        sys.exit()
        