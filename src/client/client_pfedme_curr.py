import numpy as np
import copy 
import sys

import torch 
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

import src.pacing_fn as pacing_fn
from src.utils import eval_on_dataloader,exp_lr_decay
import collections

class Client_pFedMe_curr(object):
    score_data_indices = {};
    train_count = {};
    
    def __init__(self, name, model, local_bs, local_ep, lr0, lr_a, lr_b, momentum, w_decay, device, 
                 lg_scoring, 
                 lambd,K,eta,
                 train_ds_local = None, test_dl_local = None, 
                 ordering='curr', pacing_f='linear', pacing_a=0.2, pacing_b=0.4, w_expert=None,
                 log_net='False'):
        
        self.name = name 
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr0 = lr0
        self.lr_a = lr_a 
        self.lr_b = lr_b 
        self.momentum = momentum
        self.w_decay = w_decay
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_test = test_dl_local
        self.acc_best = 0
        Client_pFedMe_curr.train_count[self.name]=0
        self.save_best = True
        
        self.lambd = lambd
        self.K = K
        self.eta = eta
        
        self.lds_train = train_ds_local
        self.ordering = ordering
        self.pacing_f=pacing_f
        self.pacing_a=pacing_a
        self.pacing_b=pacing_b
        self.w_expert=w_expert
        self.lg_scoring=lg_scoring
        self.log_net=log_net
        self.w_local = None
        
    def train(self, w_glob, logger = None):
        Client_pFedMe_curr.train_count[self.name]+=1
        
        order,stat = self.scoring_func(w_glob=w_glob)
        if self.ordering == "curr":
            order = order
        elif self.ordering == "rand":
            np.random.shuffle(order)
        elif  self.ordering == "anti":
            order = [x for x in reversed(order)]
        else:
            print('Ordering Does Not Exist')
            sys.exit()
        
        if logger:
            logger.add_histogram(torch.tensor(list(stat.values())),f"score/Client{self.name}/Data")
            for idx in Client_pFedMe_curr.score_data_indices[self.name]:
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
                _,y = self.lds_train[idx];
                rank = order.index(idx)
                logger.add_scalars(trace,f"score/Data/Class{y}/{idx}")
                logger.add_scalars({"Rank":rank},f"rank/Data/Class{y}/{idx}")
                
        
        self.net.load_state_dict(w_glob,strict=False);
        self.net.to(self.device)
        self.net.train()
        
        net_param = [];
        for name,param in self.net.named_parameters():
            if name[:6] == 'linear':
                continue;
            net_param.append(param);
        
        optimizer = torch.optim.SGD([{'params': net_param},
                                    {'params': self.net.linear.parameters()}], lr=self.lr0, momentum=self.momentum, weight_decay=self.w_decay);
        lr_scheduler=exp_lr_decay(self.lr0,self.lr_a,self.lr_b);
        
        lr=self.lr0;
        bs = self.local_bs
        N = len(order)
        myiterations = (N//bs+1)*self.local_ep

        iter_per_epoch = N//bs         
        pre_iterations = 0
        startIter = 0
        step = 0
        
        pacing_function = pacing_fn.get_pacing_function(self.pacing_f,self.pacing_a,self.pacing_b,
                                                        myiterations, N);
        startIter_next = pacing_function(step)
        trainsets = Subset(self.lds_train, list(order[startIter:max(startIter_next,32)]))
        ds_loader = DataLoader(trainsets, batch_size=self.local_bs, shuffle=True,pin_memory=False) 
        
        buffer_list = [k for k,_ in self.net.named_buffers()];
        w_r={};
        for k,v in w_glob.items():
            if k not in buffer_list:
                w_r[k]=v.detach();
        
        epoch_loss = []
        while step < myiterations:
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(ds_loader):
                images, labels = images.to(self.device), labels.type(torch.LongTensor).to(self.device)
                
                self.net.zero_grad()
                #optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels);
                regloss = self.lambd*self.calc_w_diff(self.get_param_dict(),w_r)/2;
                totloss = loss + regloss;
                totloss.backward()
                
                optimizer.step()
                batch_loss.append(loss.item())
                if step%self.K==0:                              # Update w_r every K rounds
                    for (n1,p),(n2,wr_i) in zip(self.net.named_parameters(),w_r.items()):
                        wr_i.data = wr_i.data - self.eta * (p.grad.data + self.lambd * (p.data - wr_i.data));   # Updating w_r model
                
                if logger:
                    logger.add_scalars({"dataloss":loss.item(),"regloss":regloss.item(),"totloss":totloss.item()},f"train_loss/Client{self.name}",step=step);
                    logger.add_scalars({"lr":lr},f"lr/Client{self.name}",step=step);
                
                if str(self.log_net).lower()=='true':                
                    if step+1 in [int(0.10*myiterations), int(0.5*myiterations), int(0.8*myiterations), int(0.99*myiterations)]:
                        if logger:
                            logger.add_weight_dist(self.get_state_dict(),f"Client{self.name}",step=step)
                
                lr = lr_scheduler(step);
                optimizer.param_groups[0]['lr'] = lr;
                # optimizer.param_groups[1]['lr'] = 10*lr;
                optimizer.param_groups[1]['lr'] = lr;
                step+=1
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            if startIter_next <= N:
                startIter_next = pacing_function(step)
                trainsets = Subset(self.lds_train, list(order[startIter:max(startIter_next, 32)]))
                ds_loader = DataLoader(trainsets, batch_size=self.local_bs, shuffle=True,pin_memory=False)
        
        self.w_local = copy.deepcopy(self.get_param_dict())
        return sum(epoch_loss) / len(epoch_loss), step
    
    def scoring_func(self, w_glob=None):
        ind_loss  = collections.defaultdict(list)
        ds_loader = DataLoader(self.lds_train, batch_size=self.local_bs, shuffle=False,pin_memory=False)
        
        # backup
        w_curr_net = copy.deepcopy(self.get_state_dict())
        
        with torch.no_grad():
            if (self.lg_scoring in ("E")):
                if (self.w_expert != None):
                    ind_loss= self.evaluate(ind_loss,ds_loader,w_net=self.w_expert)
                else:
                    print('Expert scoring selected but expert moddel not loaded ... exiting')
                    sys.exit()
                
            if (self.lg_scoring in ("LG","G")) or (self.lg_scoring=="L" and self.w_local == None):
                ind_loss= self.evaluate(ind_loss,ds_loader,w_net=w_glob)
            
            if (self.w_local != None) and (self.lg_scoring in ("LG","L")):
                ind_loss= self.evaluate(ind_loss,ds_loader,w_net=self.w_local)

        stat = {k:v for k, v in sorted(ind_loss.items(), key=lambda item:sum(item[1]))}
        myorder = list(stat.keys())
        
        # restore
        self.net.load_state_dict(w_curr_net)
        self.net.train()
        return myorder,stat

    def evaluate(self,ind_loss,ds_loader,w_net):
        order = [i for i in range(len(ds_loader.dataset))]
        
        criterion = nn.CrossEntropyLoss(reduction="none").to(self.device)
        
        self.net.load_state_dict(w_net,strict=False)
        self.net.to(self.device);
        # switch to evaluate mode
        self.net.eval()
        start = 0
        for i, (images, labels) in enumerate(ds_loader):
            images, labels = images.to(self.device), labels.type(torch.LongTensor).to(self.device)
            output = self.net(images)
            indloss = criterion(output, labels)
            list(map(lambda a, b : ind_loss[a].append(b.item()), order[start:start+len(labels)], indloss))
            start += len(labels)
        
        return ind_loss;
    
    def get_param_dict(self):
        state_dict = self.net.state_dict();
        buffer_list = [k for k,_ in self.net.named_buffers()];
        [state_dict.pop(key) for key in buffer_list];
        return state_dict;
    def get_state_dict(self):
        return self.net.state_dict()
    def get_best_acc(self):
        return self.acc_best
    def get_count(self):
        return self.count
    def get_net(self):
        return self.net
    def set_state_dict(self, state_dict,strict=True):
        self.net.load_state_dict(state_dict,strict=strict);

    def eval_test(self):
        return eval_on_dataloader(self.net,self.ldr_test,device=self.device)
    
    def eval_test_glob(self, glob_dl):
        return eval_on_dataloader(self.net,glob_dl,device=self.device);
    
    def eval_train(self):
        return eval_on_dataloader(self.net,self.ldr_train,device=self.device);
    
    def calc_w_diff(self,w1,w2,p=2):
        return sum(torch.linalg.vector_norm((x.float() - y.float()),ord=p) for x, y in zip(w2.values(), w1.values()));