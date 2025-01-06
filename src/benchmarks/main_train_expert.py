import torch
import copy

from src.data import *
from src.models import *
from src.client import * 
from src.clustering import *
from src.utils import * 

def main_train_expert(args,log=None):
    print(' ')
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    #print(str(args))
    ################################### build model
    print('-'*40)
    print('Building models for clients')
    print(f'MODEL: {args.model}, Dataset: {args.dataset}')
    _, netE, _, _ = init_nets(args, dropout_p=0.5)
    print('-'*40)
    print(netE)
    print('')
    
    total = 0 
    for name, param in netE.named_parameters():
        print(name, param.size())
        total += np.prod(param.size())
        #print(np.array(param.data.cpu().numpy().reshape([-1])))
        #print(isinstance(param.data.cpu().numpy(), np.array))
    print(f'total params {total}')
    print('-'*40)
    ################################# Getting Global Datasets
    train_dl, test_dl, train_ds, test_ds = get_dataloader(args.dataset,args.datadir,
                                                                       args.batch_size,
                                                                       args.batch_size,num_workers=8)
    ################################# Training expert model
    step=0;
    print('-'*40)
    print('Training Expert Model')
    net_expert = copy.deepcopy(netE)
    w_expert,train_loss,train_acc,test_loss,test_acc,step= train_model(netE,train_dl,test_dl,'expert',
                                                                    lr0=args.lr0,a=args.lr_sched_a,b=args.lr_sched_b,momentum=args.momentum,weight_decay=args.w_decay,
                                                                    num_epoch=args.num_epoch,batch_size=args.batch_size,
                                                                    logger=log,device=args.device,step0=step);
    net_expert.load_state_dict(w_expert);
    print( 'Expert Training Done')
    print(f'       train_loss:{train_loss}')
    print(f'       test_loss:{test_loss}')
    print(f'       test_acc:{test_acc}')
    print('Saving Trained Expert Model')
    expert_pt_path = args.ptdir + '/' + args.dataset + '/' + args.model + '/' + 'expert.pt';
    torch.save(w_expert,expert_pt_path)
    
    return train_loss,train_acc,test_loss,test_acc

def run_trainexpert(args, fname):
    alg_name = 'TrainExpert'
            
    path = args.logdir + args.exp_label + '/' + args.alg +'/' + args.dataset + '/' + args.model + '/'    
    mkdirs(path)    
    log = TBLogger(path + '/' + "tblog_"+ args.log_filename);        
        
    log.trial_id = 0;
    log.global_step = 0;
    
    train_loss,train_acc,test_loss,test_acc = main_train_expert(args,log=log);        
        
    print('*'*40)
    print(' '*20, alg_name)
    
    template = "-- Final Train Loss: {:.3f} " 
    print(template.format(train_loss))
    template = "-- Final Train Acc: {:.3f} " 
    print(template.format(train_acc))
    template = "-- Final Test Loss: {:.3f} " 
    print(template.format(test_loss))
    template = "-- Final Test Acc: {:.3f} " 
    print(template.format(test_acc))
    
    with open(fname+'_results_summary.txt', 'a') as text_file:
        print('*'*40)
        print(' '*20, alg_name)
        
        template = "-- Final Train Loss: {:.3f} " 
        print(template.format(train_loss),file=text_file)
        template = "-- Final Train Acc: {:.3f} " 
        print(template.format(train_acc),file=text_file)
        template = "-- Final Test Loss: {:.3f} " 
        print(template.format(test_loss),file=text_file)
        template = "-- Final Test Acc: {:.3f} " 
        print(template.format(test_acc),file=text_file)
    
        print('*'*40)
        
    return    