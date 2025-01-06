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
    
    args = train_model_args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu) ## Setting cuda on GPU 
    path = args.logdir + args.exp_label + '/' + args.alg +'/' + args.dataset + '/' + args.model + '/'
    mkdirs(path)
    ptpath = args.ptdir + args.dataset + '/' + args.model + '/'
    mkdirs(ptpath)
    
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
    if args.alg == 'train_expert':
        alg_name = 'TrainExpert'
        run_trainexpert(args, fname=fname)
    else: 
        print('Algorithm Does Not Exist')
        sys.exit()
        