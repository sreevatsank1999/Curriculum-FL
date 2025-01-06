from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import sys

class TBLogger:    
    def __init__(self,logpath) -> None:
        self.writer = SummaryWriter(logpath);
        self.trial_id=0;
        self.global_step=0;

    def get_tag_prefix(self):
        return f"Trial{self.trial_id}";
        
    def add_histogram(self,values,name,step=0):
        self.writer.add_histogram(self.get_tag_prefix() + f"/{name}",values,self.global_step+step);
    
    def add_scalars(self,traces,name,step=0):
        self.writer.add_scalars(self.get_tag_prefix() + f"/{name}",traces,self.global_step+step);
        
    def add_weight_dist(self,w,name,step=0):
        for layer in w.keys():
            self.writer.add_histogram(self.get_tag_prefix() + f"/{name}/{layer}",w[layer],self.global_step+step);
            
    def add_graph(self,net,input):
        self.writer.add_graph(net,input,verbose=True);
        
        
def eval_checkpoint(clients, test_dl_global, clients_local_acc, clients_global_acc, logger=None):    
    temp_acc = []; temp_glob_acc = [];
    temp_best_acc = []; temp_best_glob_acc = [];
    for k in range(len(clients)):
        sys.stdout.flush()
        loss, acc,_ = clients[k].eval_test() 
        loss_globdl, acc_globdl,_ = clients[k].eval_test_glob(glob_dl=test_dl_global) 
        clients_local_acc[k].append(acc)
        clients_global_acc[k].append(acc_globdl)
        temp_acc.append(clients_local_acc[k][-1])
        temp_best_acc.append(np.max(clients_local_acc[k]))
        temp_glob_acc.append(clients_global_acc[k][-1])
        temp_best_glob_acc.append(np.max(clients_global_acc[k]))

        template = ("Client {:3d} Local, current_acc {:3.2f}, best_acc {:3.2f}")
        print(template.format(k, clients_local_acc[k][-1], np.max(clients_local_acc[k])))
        template = ("Client {:3d} Global, current_acc {:3.2f}, best_acc {:3.2f}")
        print(template.format(k, clients_global_acc[k][-1], np.max(clients_global_acc[k])))
        
        if logger:
            logger.add_scalars({"Local":acc,"Global":acc_globdl},f"Client{clients[k].name}/test_accuracy")
            logger.add_scalars({"Local":loss,"Global":loss_globdl},f"Client{clients[k].name}/test_loss")

    #print('*'*25)
    template = ("-- Avg Client Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_acc)))
    template = ("-- Avg Best Client Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_best_acc)))
    template = ("-- Avg Client Global Acc: {:3.2f}")
    print(template.format(np.mean(temp_glob_acc)))
    template = ("-- Avg Best Client Global Acc: {:3.2f}")
    print(template.format(np.mean(temp_best_glob_acc)))
    print('*'*25)
    
    if logger:
        logger.add_histogram(torch.tensor(np.reshape(temp_acc,(len(temp_acc),))),f"ClientLocal/test_accuracy")
        logger.add_histogram(torch.tensor(np.reshape(temp_glob_acc,(len(temp_glob_acc),))),f"ClientGlobal/test_accuracy")
        logger.add_scalars({"Local":np.mean(temp_acc),"Global":np.mean(temp_glob_acc)},f"ClientAvg/test_accuracy")
        logger.add_scalars({"Local":np.mean(temp_best_acc),"Global":np.mean(temp_best_glob_acc)},f"ClientBest/test_accuracy")
        
    return clients_local_acc,clients_global_acc,temp_acc,temp_best_acc,temp_glob_acc,temp_best_glob_acc;