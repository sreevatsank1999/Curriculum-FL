import numpy as np
import copy 

import torch 
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

import collections

class Client_FedAvg_Curr(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device, 
                 train_ds_local = None, test_dl_local = None, 
                 ordering='curr', pacing_f='linear', pacing_a=0.2, pacing_b=0.4):
        
        self.name = name 
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr 
        self.momentum = momentum 
        self.device = device 
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_test = test_dl_local
        self.acc_best = 0 
        self.count = 0 
        self.save_best = True 
        
        self.lds_train = train_ds_local
        self.ordering = ordering
        self.pacing_f='linear'
        self.pacing_a=0.2
        self.pacing_b=0.2
        
    def train(self, is_print = False):
        self.net.to(self.device)
        self.net.train()
        
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)
        
        order = self.scoring_func()
        if self.ordering == "curr":
            order = order
        elif self.ordering == "random":
            np.random.shuffle(order)
        elif  self.ordering == "anti_curr":
            order = [x for x in reversed(order)]
        else:
            print('Ordering Does Not Exist')
            sys.exit()
        
        bs = self.local_bs
        N = len(order)
        myiterations = (N//bs+1)*self.local_ep

        iter_per_epoch = N//bs         
        pre_iterations = 0
        startIter = 0
        step=0
        pacing_function = self.get_pacing_function(myiterations, N)
        
        startIter_next = pacing_function(step)
        trainsets = Subset(self.lds_train, list(order[startIter:max(startIter_next,32)]))
        ds_loader = DataLoader(trainsets, batch_size=self.local_bs, shuffle=True) 

        epoch_loss = []
        while step < myiterations:
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(ds_loader):
                step+=1
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)
                
                self.net.zero_grad()
                #optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward() 
                        
                optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            pre_iterations = step          
            if startIter_next <= N:            
                startIter_next = pacing_function(step)
                trainsets = Subset(self.lds_train, list(order[startIter:max(startIter_next,32)]))
                ds_loader = DataLoader(trainsets, batch_size=self.local_bs, shuffle=True)
            
#         if self.save_best: 
#             _, acc = self.eval_test()
#             if acc > self.acc_best:
#                 self.acc_best = acc 
        
        return sum(epoch_loss) / len(epoch_loss)
    
    def scoring_func(self):
        order = [i for i in range(len(self.lds_train))]
        ind_loss  = collections.defaultdict(list)

        ds_loader = DataLoader(self.lds_train, batch_size=self.local_bs, shuffle=False) 
        # switch to evaluate mode
        self.net.eval()
        criterion = nn.CrossEntropyLoss(reduction="none").to(self.device)
        start = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(ds_loader):
                images, labels = images.to(self.device), labels.type(torch.LongTensor).to(self.device)
                output = self.net(images)
                indloss = criterion(output, labels)
                list(map(lambda a, b : ind_loss[a].append(b.item()), order[start:start+len(labels)], indloss))
                start += len(labels)

        stat = {k:v[0] for k, v in sorted(ind_loss.items(), key=lambda item:sum(item[1]))}
        myorder = list(stat.keys())
        self.net.train()
        return myorder
    
    def get_pacing_function(self, total_step, total_data):
        """Return a  pacing function  w.r.t. step.
        input:
        a:[0,large-value] percentage of total step when reaching to the full data. This is an ending point (a*total_step,
        total_data)) 
        b:[0,1]  percentatge of total data at the begining of the training. Thia is a starting point (0,b*total_data))
        """
        a = self.pacing_a
        b = self.pacing_b 
        index_start = b*total_data
        if self.pacing_f == 'linear':
            rate = (total_data - index_start)/(a*total_step)
            def _linear_function(step):
                return int(rate *step + index_start)
            return _linear_function

        elif self.pacing_f == 'quad':
            rate = (total_data-index_start)/(a*total_step)**2  
            def _quad_function(step):
                return int(rate*step**2 + index_start)
            return _quad_function

        elif self.pacing_f == 'root':
            rate = (total_data-index_start)/(a*total_step)**0.5
            def _root_function(step):
                return int(rate *step**0.5 + index_start)
            return _root_function

        elif self.pacing_f == 'step':
            threshold = a*total_step
            def _step_function(step):
                return int( total_data*(step//threshold) +index_start)
            return _step_function      

        elif self.pacing_f == 'exp':
            c = 10
            tilde_b  = index_start
            tilde_a  = a*total_step
            rate =  (total_data-tilde_b)/(np.exp(c)-1)
            constant = c/tilde_a
            def _exp_function(step):
                if not np.isinf(np.exp(step *constant)):
                    return int(rate*(np.exp(step*constant)-1) + tilde_b )
                else:
                    return total_data
            return _exp_function

        elif self.pacing_f == 'log':
            c = 10
            tilde_b  = index_start
            tilde_a  = a*total_step
            ec = np.exp(-c)
            N_b = (total_data-tilde_b)
            def _log_function(step):
                return int(N_b*(1+(1./c)*np.log(step/tilde_a+ ec)) + tilde_b )
            return _log_function
        
    def get_state_dict(self):
        return self.net.state_dict()
    def get_best_acc(self):
        return self.acc_best
    def get_count(self):
        return self.count
    def get_net(self):
        return self.net
    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
                
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy
    
    def eval_test_glob(self, glob_dl):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in glob_dl:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
                
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(glob_dl.dataset)
        accuracy = 100. * correct / len(glob_dl.dataset)
        return test_loss, accuracy
    
    def eval_train(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
                
                output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy
