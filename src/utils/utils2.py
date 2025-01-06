import sys
import os
#sys.path.append("..")
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from src.data import *
from src.models import *
from src.utils import * 

import numpy as np

import copy
import gc 

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class Logger(object):
    def __init__(self, fname):
        self.terminal = sys.stdout
        self.log = open(fname, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass 

def interpol_weights(w1,w2,alpha):        
    w_avg = {}
    for k in w1.keys():
        if k in w2.keys():        
            w_avg[k] = w1[k].cuda()*alpha + w2[k].cuda()*(1-alpha)
    return w_avg

def FedAvg(w, weight_avg=None):
    """
    Federated averaging
    :param w: list of client model parameters
    :return: updated server model parameters
    """
    if weight_avg == None:
        weight_avg = [1/len(w) for i in range(len(w))]
        
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].cuda() * weight_avg[0]
        
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k].cuda() + w[i][k].cuda() * weight_avg[i]
        #w_avg[k] = torch.div(w_avg[k].cuda(), len(w)) 
    return w_avg

def init_nets(args, dropout_p=0.5):

    users_model = []
    net=None;
    for net_i in range(-1, args.num_users):
        if args.dataset == "generated":
            net = PerceptronModel().to(args.device)
        elif args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16,8]
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p).to(args.device)
        elif args.model == "vgg":
            net = vgg11().to(args.device)
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(in_chan=3,input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNN(in_chan=1,input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset in ('organamnist','organcmnist','organsmnist'):
                net = SimpleCNN(in_chan=1,input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=11).to(args.device)
            elif args.dataset in ('tissuemnist'):
                net = SimpleCNN(in_chan=1,input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=8).to(args.device)
            elif args.dataset in ('bloodmnist'):
                net = SimpleCNN(in_chan=3,input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=8).to(args.device)
            elif args.dataset in ('pathmnist'):
                net = SimpleCNN(in_chan=3,input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=9).to(args.device)
            elif args.dataset in ('dermamnist'):
                net = SimpleCNN(in_chan=3,input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=7).to(args.device)
                
            elif args.dataset == 'stl10':
                net = SimpleCNN(in_chan=3,input_dim=(16 * 21 * 21), hidden_dims=[120*4, 84*4], output_dim=10).to(args.device)
            elif args.dataset == 'celeba':
                net = SimpleCNN(in_chan=3,input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2).to(args.device)
        elif args.model =="simple-cnn-3":
            if args.dataset == 'cifar100': 
                net = SimpleCNN_3(input_dim=(16 * 3 * 5 * 5), hidden_dims=[120*3, 84*3], output_dim=100).to(args.device)
            if args.dataset == 'tinyimagenet':
                net = SimpleCNNTinyImagenet_3(input_dim=(16 * 3 * 13 * 13), hidden_dims=[120*3, 84*3], 
                                              output_dim=200).to(args.device)
        elif args.model == "lenet-5":
            if args.dataset in ("cifar10"):
                net = LeNetBN5Cifar().to(args.device);
        elif args.model == "resnet-50":
            if args.dataset in ("food101","food101n"):
                net = ResNet50(101).to(args.device);
            elif args.dataset == "miniimagenet":
                net = ResNet50(100).to(args.device);
                
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST().to(args.device)
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN().to(args.device)
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2).to(args.device)
        elif args.model == 'resnet-9': 
            if args.dataset in ['cifar100','cifar100-lvln']: 
                net = ResNet9(num_classes=100).to(args.device)
            elif args.dataset == 'stl10':
                net = ResNet9(num_classes=10).to(args.device)
            elif args.dataset == 'tinyimagenet': 
                net = ResNet9(num_classes=200).to(args.device)
            elif args.dataset == 'isic2019': 
                net = ResNet9(num_classes=8).to(args.device)
            elif args.dataset == 'flowers102':
                net = ResNet9(num_classes=102).to(args.device)
            elif args.dataset in ("food101","food101n"):
                net = ResNet9(num_classes=101).to(args.device)
            elif args.dataset == "miniimagenet":
                net = ResNet9(num_classes=100).to(args.device)
        elif args.model == '_resnet9': 
            if args.dataset in ['cifar100','cifar100-lvln']: 
                net = ResNet9_(in_channels=3, num_classes=100).to(args.device);
            elif args.dataset == 'stl10':
                net = ResNet9_(in_channels=3, num_classes=10, dim=4608).to(args.device);
            elif args.dataset == 'tinyimagenet': 
                net = ResNet9_(in_channels=3, num_classes=200, dim=512*2*2).to(args.device);
        elif args.model == "resnet":
            net = ResNet50_cifar10().to(args.device)
        elif args.model == "vgg16":
            net = vgg16().to(args.device)
        if net == None:
            print(f"Model: {args.model} and dataset {args.dataset} combination is not supported yet")
            exit(1)
            
        if net_i == -1: 
            net_glob = copy.deepcopy(net)
            initial_state_dict = copy.deepcopy(net_glob.state_dict())
            server_state_dict = copy.deepcopy(net_glob.state_dict())
            if args.load_initial:
                initial_state_dict = torch.load(args.load_initial)
                server_state_dict = torch.load(args.load_initial)
                net_glob.load_state_dict(initial_state_dict)            
        else:
            users_model.append(copy.deepcopy(net))
            users_model[net_i].load_state_dict(initial_state_dict)

    return users_model, net_glob, initial_state_dict, server_state_dict