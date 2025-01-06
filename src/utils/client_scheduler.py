from math import inf
import numpy as np
import copy 
import sys

import torch 
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

import src.pacing_fn as pacing_fn

class ClientScheduler(object):
    score_client_indices=[]
    def __init__(self, client_set, total_steps,
                ordering='random', pacing_f='step', scoring='G', pacing_a=inf, pacing_b=None):
        
        if pacing_b == None:
            pacing_b = (1/total_steps);
        
        self.total_steps = total_steps;
                                
        self.client_set = client_set;
        self.ordering = ordering;
        self.pacing_f = pacing_f;
        self.pacing_a = pacing_a;
        self.pacing_b = pacing_b;
        self.lg_scoring = scoring;
        
        self.pacing_function = pacing_fn.get_pacing_function(pacing_f,pacing_a,pacing_b,total_steps,len(client_set));
        
    def scoring_func(self,w_glob):
        
        order = [i for i in range(len(self.client_set))]
        ind_loss  = {};

        for i in range(len(self.client_set)):            
            _,stat = self.client_set[i].scoring_func(w_glob=w_glob);
            indloss = [np.mean([v[j] for k,v in stat.items()]) for j in range(len(list(stat.values())[0]))];
            
            ind_loss[i]=indloss;


        stat = {k:v for k, v in sorted(ind_loss.items(), key=lambda item:sum(item[1]))};
        myorder = list(stat.keys())
        return myorder,stat
        
    def next_sched(self,w_glob,curr_step,logger=None):
        
        order,stat = self.scoring_func(w_glob=w_glob);
        if self.ordering == "curr":
            order = order
        elif self.ordering == "rand":
            np.random.shuffle(order)
        elif  self.ordering == "anti":
            order = [x for x in reversed(order)]
        else:
            print('Ordering Does Not Exist')
            sys.exit();
            
        if logger:
            logger.add_histogram(torch.tensor(list(stat.values())),f"score/ClientSet")
            for idx in ClientScheduler.score_client_indices:
                trace = {}
                if self.lg_scoring == 'L':
                    trace['L'] = stat[idx][0]
                elif self.lg_scoring == 'G':
                    trace['G'] = stat[idx][0]
                elif self.lg_scoring == 'E':
                    trace['E'] = stat[idx][0]
                elif self.lg_scoring == 'LG':
                    trace['G'] = stat[idx][0]
                    if len(stat[idx])==2:
                        trace['L'] = stat[idx][1]
                    trace['LG'] = sum(stat[idx])
                rank = order.index(idx)
                logger.add_scalars(trace,f"score/Client/{idx}")
                logger.add_scalars({"Rank":rank},f"rank/Client/{idx}")       
            
        pool_size = self.pacing_function(curr_step);
        select_clients = Subset(self.client_set, list(order[0:pool_size]));
                        
        return select_clients;