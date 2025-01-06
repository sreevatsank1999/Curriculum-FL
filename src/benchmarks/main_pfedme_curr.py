import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

from src.data import *
from src.models import *
from src.client import * 
from src.clustering import *
from src.utils import * 

def main_pfedme_curr(args,log=None):
    print(' ')
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    #print(str(args))
    ################################### build model
    print('-'*40)
    print('Building models for clients')
    print(f'MODEL: {args.model}, Dataset: {args.dataset}')
    users_model, net_glob, initial_state_dict, server_state_dict = init_nets(args, dropout_p=0.5)
    print('-'*40)
    print(net_glob)
    print('')

    total = 0 
    for name, param in net_glob.named_parameters():
        print(name, param.size())
        total += np.prod(param.size())
        #print(np.array(param.data.cpu().numpy().reshape([-1])))
        #print(isinstance(param.data.cpu().numpy(), np.array))
    print(f'total params {total}')
    print('-'*40)
    ################################# Getting Global Datasets
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                       args.datadir,
                                                                                       args.batch_size,
                                                                                       32,num_workers=8)
    ################################# Fixing all to the same Init and data partitioning and random users 
    #print(os.getcwd())

    # tt = '../initialization/' + 'traindata_'+args.dataset+'_'+args.partition+'.pkl'
    # with open(tt, 'rb') as f:
    #     net_dataidx_map = pickle.load(f)

    # tt = '../initialization/' + 'testdata_'+args.dataset+'_'+args.partition+'.pkl'
    # with open(tt, 'rb') as f:
    #     net_dataidx_map_test = pickle.load(f)

    # tt = '../initialization/' + 'traindata_cls_counts_'+args.dataset+'_'+args.partition+'.pkl'
    # with open(tt, 'rb') as f:
    #     traindata_cls_counts = pickle.load(f)

    # tt = '../initialization/' + 'testdata_cls_counts_'+args.dataset+'_'+args.partition+'.pkl'
    # with open(tt, 'rb') as f:
    #     testdata_cls_counts = pickle.load(f)

#     tt = '../' + args.model+'-init' +'.pth'
#     initial_state_dict = torch.load(tt, map_location=args.device)
#     net_glob.load_state_dict(initial_state_dict)

#     server_state_dict = copy.deepcopy(initial_state_dict)
#     for idx in range(args.num_users):
#         users_model[idx].load_state_dict(initial_state_dict)

    # tt = '../initialization/' + 'comm_users.pkl'
    # with open(tt, 'rb') as f:
    #     comm_users = pickle.load(f)
    ################################# Load Expert, Training expert model if required
    expert_pt_path = args.ptdir + '/' + args.dataset + '/' + args.model + '/' + 'expert.pt';
    net_expert = None
    w_expert=None
    if os.path.exists(expert_pt_path):        
        w_expert = torch.load(expert_pt_path,map_location=args.device.type);
        net_expert = copy.deepcopy(net_glob)
        net_expert.load_state_dict(w_expert)
    elif str(args.train_expert).lower()=='true':
        print('-'*40)
        print('Training Expert Model')
        net_expert = copy.deepcopy(net_glob)
        w_expert,train_loss,test_loss,test_acc,steps= train_model(net_expert,train_ds_global,test_ds_global,'expert',
                                                                    lr0=args.lr0,a=args.lr_sched_a,b=args.lr_sched_b,momentum=args.momentum,weight_decay=args.w_decay,
                                                                    num_epoch=10,batch_size=64,
                                                                    num_workers=8,logger=log,device=args.device)
        net_expert.load_state_dict(w_expert);
        print( 'Expert Training Done')
        print(f'       train_loss:{train_loss}')
        print(f'       test_loss:{test_loss}')
        print(f'       test_acc:{test_acc}')
        print('Saving Trained Expert Model')
        expert_pt_path = args.ptdir + '/' + args.dataset + '/' + args.model + '/' + 'expert.pt';
        torch.save(w_expert,expert_pt_path)
        
    ##################################### Data partitioning section 
    print('-'*40)
    print('Getting Clients Data')

    net_dataidx_map, net_dataidx_map_test, \
    traindata_cls_counts, testdata_cls_counts = get_clients_data(args,train_ds_global,test_ds_global,data_score_net=net_expert)

    print('-'*40)    
    ################################# Initializing Clients   
    print('-'*40)
    print('Initializing Clients')
    clients = []
    for idx in range(args.num_users):
        sys.stdout.flush()
        print(f'-- Client {idx}, Labels Stat {traindata_cls_counts[idx]}')

        noise_level=0
        dataidxs = net_dataidx_map[idx]
        dataidxs_test = net_dataidx_map_test[idx]

        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, 
                                                                       args.datadir, args.local_bs, 32, 
                                                                       dataidxs, noise_level, 
                                                                       dataidxs_test=dataidxs_test)

        clients.append(Client_pFedMe_curr(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep, 
                                           args.lr0,args.lr_sched_a,args.lr_sched_b,args.momentum,args.w_decay,
                                           args.device, args.lg_scoring, 
                                           args.pfedme_lambd, args.pfedme_K, args.pfedme_eta, 
                                           train_ds_local, test_dl_local, 
                                           args.ordering, args.pacing_f, args.pacing_a, args.pacing_b, w_expert,
                                           args.log_clientnet))

    client_set = ClientSet(clients);
    ################################ Score Client Data
    Client_pFedMe_curr.score_data_indices = {i:np.random.randint(0,
                                                                 len(clients[i].lds_train),
                                                                    dtype=np.int32,
                                                                    size=(np.random.binomial(len(clients[i].lds_train),args.data_score_sample_p,))
                                                                    ) for i in range(args.num_users)}
    
    print('-'*40)
    ###################################### Federation 
    print('Starting FL')
    print('-'*40)
    start = time.time()

    loss_train = []
    clients_local_acc = {i:[] for i in range(args.num_users)}
    clients_global_acc = {i:[] for i in range(args.num_users)}
    w_locals, loss_locals = [], []
    glob_acc = []

    buffer_list = [k for k,_ in net_glob.named_buffers()];
    w_glob = copy.deepcopy(initial_state_dict)
    [w_glob.pop(key) for key in buffer_list];
    if log:
        images,_ = next(iter(train_dl_global))
        log.add_graph(net_glob,images.to(args.device))
        log.add_weight_dist(w_glob,"Global");
    
    m = max(int(args.frac * args.num_users), 1)
    total_budget = args.rounds*m;
    
    client_curriculum = ClientScheduler(client_set,total_budget,ordering=args.client_ordering,scoring=args.lg_scoring,
                                        pacing_f=args.client_pacing_f,pacing_a=args.client_pacing_a,pacing_b=args.client_pacing_b);
    ################################ Score Clients
    ClientScheduler.score_client_indices = np.random.randint(0,
                                                             args.num_users,
                                                             dtype=np.int32,
                                                             size=(args.client_score_sample_n,))    
    
    torch.backends.cudnn.benchmark = True;        
    
    iteration=0
    nbupdate=0;    
    while nbupdate < total_budget:

        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #idxs_users = comm_users[iteration]
        current_client_set = client_curriculum.next_sched(w_glob=w_glob,curr_step=nbupdate,logger=log);
        client_loader = DataLoader(current_client_set,batch_size=args.client_bs,shuffle=True,collate_fn=client_collate);
        
        for n,client_batch in enumerate(client_loader):                    
            print(f'----- ROUND {iteration+1} -----')   
            sys.stdout.flush()
            steps_local=[];
            for client in client_batch:
                loss,steps = client.train(w_glob=copy.deepcopy(w_glob),logger=log)
                loss_locals.append(copy.deepcopy(loss))                                      
                steps_local.append(copy.deepcopy(steps))                                      

            nbupdate += len(client_batch);
            if log:
                log.global_step += np.max(steps_local);
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            template = '-- Average Train loss {:.3f}'
            print(template.format(loss_avg))
            if log:
                log.add_scalars({"loss":loss_avg},f"train_loss/ClientBatchAvg")
            
            ####### pFedMe ####### START
            w_locals = []
            for client in client_batch:
                w_locals.append(copy.deepcopy(client.get_state_dict()))

            ww = FedAvg(w_locals);
            w_glob = copy.deepcopy(interpol_weights(w_glob,ww,alpha=(1-args.pfedme_beta)))
            net_glob.load_state_dict(copy.deepcopy(w_glob),strict=False)

            ####### pFedMe ####### END
            test_loss, acc = eval_test(net_glob, args, test_dl_global)
            if log:
                log.add_weight_dist(w_glob,"Global");
                log.add_scalars({"Accuracy":acc},"Global/test_accuracy")
                log.add_scalars({"Loss":test_loss},"Global/test_loss")
            
            glob_acc.append(acc)
            template = "-- Global Acc: {:.3f}, Global Best Acc: {:.3f}"
            print(template.format(glob_acc[-1], np.max(glob_acc)))

            print_flag = False
            if iteration+1 in [int(0.10*args.rounds), int(0.5*args.rounds), int(0.8*args.rounds)]: 
                print_flag = True

            if print_flag:
                print('*'*25)
                print(f'Check Point @ Round {nbupdate+1} --------- {int((nbupdate+1)/total_budget*100)}% Completed')
                clients_local_acc,clients_global_acc,_,_,_,_ = eval_checkpoint(clients, test_dl_global, clients_local_acc, clients_global_acc, logger=log)

            loss_train.append(loss_avg)
            
            iteration += 1;
                        
            ## clear the placeholders for the next round 
            loss_locals.clear()

            ## calling garbage collector 
            gc.collect()

    end = time.time()
    duration = end-start
    
    if log:
        log.add_histogram(torch.tensor(list(Client_pFedMe_curr.train_count.values())),f"Clients/train_count")
    print('-'*40)
    ############################### Testing Local Results 
    print('*'*25)
    print('---- Testing Final Local Results ----')
    clients_local_acc,clients_global_acc,temp_acc,temp_best_acc,_,_ = eval_checkpoint(clients, test_dl_global, clients_local_acc, clients_global_acc, logger=log)
    ############################### FedAvg Final Results
    print('-'*40)
    print('FINAL RESULTS')
    template = "-- Global Acc Final: {:.3f}" 
    print(template.format(glob_acc[-1]))

    template = "-- Global Acc Avg Final 10 Rounds: {:.3f}" 
    print(template.format(np.mean(glob_acc[-10:])))

    template = "-- Global Best Acc: {:.3f}"
    print(template.format(np.max(glob_acc)))

    template = ("-- Avg Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_acc)))

    template = ("-- Avg Best Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_best_acc)))

    print(f'-- FL Time: {duration/60:.2f} minutes')
    print('-'*40)
    ############################### FedAvg+ (FedAvg + FineTuning)
    print('-'*40)
    print('pFedMe_Curr+ ::: FedAvg Curr LG Loss + Local FineTuning')
    sys.stdout.flush()

    steps_local = [];
    for idx in range(args.num_users): 
        _,steps = clients[idx].train(w_glob=copy.deepcopy(w_glob),logger=log)
        steps_local.append(copy.deepcopy(steps))   
        
    if log:
        log.global_step += np.max(steps_local);
    
    clients_local_acc,clients_global_acc,temp_acc,_,temp_glob_acc,_ = eval_checkpoint(clients, test_dl_global, clients_local_acc, clients_global_acc, logger=log)
    
    fedavg_ft_client_local = np.mean(temp_acc) 
    fedavg_ft_client_global = np.mean(temp_glob_acc) 
    print(f'-- pFedMe_Curr+ :: AVG Client Local Acc: {np.mean(temp_acc):.2f}')
    print(f'-- pFedMe_Curr+ :: AVG Client Global Acc: {np.mean(temp_glob_acc):.2f}')
    ############################# Saving Print Results 
    
    final_glob = glob_acc[-1]
    avg_final_glob = np.mean(glob_acc[-10:])
    best_glob = np.max(glob_acc)
    avg_final_local = np.mean(temp_acc)
    avg_best_local = np.mean(temp_best_acc)
    
    return final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, fedavg_ft_client_local, fedavg_ft_client_global, duration

def run_pfedme_curr(args, fname):
    alg_name = 'pFedMe_Curr'
            
    path = args.logdir + args.exp_label + '/' + args.alg +'/' + args.dataset + '/' + args.partition + '/' + args.model + '/'    
    mkdirs(path)    
    log = TBLogger(path + '/' + "tblog_"+ args.log_filename);        
    
    exp_final_glob=[]
    exp_avg_final_glob=[]
    exp_best_glob=[]
    exp_avg_final_local=[]
    exp_avg_best_local=[]
    exp_fedavg_ft_client_local=[]
    exp_fedavg_ft_client_global=[]
    exp_fl_time=[]
    
    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))
        
        log.trial_id = trial;
        log.global_step = 0;
        
        final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, \
        fedavg_ft_client_local, fedavg_ft_client_global, duration = main_pfedme_curr(args,log=log)
        
        exp_final_glob.append(final_glob)
        exp_avg_final_glob.append(avg_final_glob)
        exp_best_glob.append(best_glob)
        exp_avg_final_local.append(avg_final_local)
        exp_avg_best_local.append(avg_best_local)
        exp_fedavg_ft_client_local.append(fedavg_ft_client_local)
        exp_fedavg_ft_client_global.append(fedavg_ft_client_global)
        exp_fl_time.append(duration/60)
        
        print('*'*40)
        print(' '*20, 'End of Trial %d'%(trial+1))
        print(' '*20, 'Final Results')
        
        template = "-- Global Final Acc: {:.3f}" 
        print(template.format(exp_final_glob[-1]))
        
        template = "-- Global Avg Final 10 Rounds Acc : {:.3f}" 
        print(template.format(exp_avg_final_glob[-1]))

        template = "-- Global Best Acc: {:.3f}"
        print(template.format(exp_best_glob[-1]))

        template = ("-- Avg Final Local Acc: {:3.2f}")
        print(template.format(exp_avg_final_local[-1]))

        template = ("-- Avg Best Local Acc: {:3.2f}")
        print(template.format(exp_avg_best_local[-1]))

        print(f'-- pFedMe_Curr+ Fine Tuning Clients AVG Client Local Acc: {exp_fedavg_ft_client_local[-1]:.2f}')
        print(f'-- pFedMe_Curr+ Fine Tuning Clients AVG Client Global Acc: {exp_fedavg_ft_client_global[-1]:.2f}')
        print(f'-- FL Time: {exp_fl_time[-1]:.2f} minutes')
        
        
    print('*'*40)
    print(' '*20, alg_name)
    print(' '*20, 'Avg %d Trial Results'%args.ntrials)
    
    template = "-- Global Final Acc: {:.3f} +- {:.3f}" 
    print(template.format(np.mean(exp_final_glob), np.std(exp_final_glob)))

    template = "-- Global Avg Final 10 Rounds Acc: {:.3f} +- {:.3f}" 
    print(template.format(np.mean(exp_avg_final_glob), np.std(exp_avg_final_glob)))

    template = "-- Global Best Acc: {:.3f} +- {:.3f}"
    print(template.format(np.mean(exp_best_glob), np.std(exp_best_glob)))

    template = ("-- Avg Final Local Acc: {:3.2f} +- {:.3f}")
    print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)))

    template = ("-- Avg Best Local Acc: {:3.2f} +- {:.3f}")
    print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)))

    template = '-- pFedMe_Curr+ Fine Tuning Clients AVG Local Acc: {:.2f} +- {:.2f}'
    print(template.format(np.mean(exp_fedavg_ft_client_local), np.std(exp_fedavg_ft_client_local)))
    template = '-- pFedMe_Curr+ Fine Tuning Clients AVG Global Acc: {:.2f} +- {:.2f}'
    print(template.format(np.mean(exp_fedavg_ft_client_global), np.std(exp_fedavg_ft_client_global)))
    
    print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes')
    
    with open(fname+'_results_summary.txt', 'a') as text_file:
        print('*'*40, file=text_file)
        print(' '*20, alg_name, file=text_file)
        print(' '*20, 'Avg %d Trial Results'%args.ntrials, file=text_file)
    
        template = "-- Global Final Acc: {:.3f} +- {:.3f}" 
        print(template.format(np.mean(exp_final_glob), np.std(exp_final_glob)), file=text_file)

        template = "-- Global Avg Final 10 Rounds Acc: {:.3f} +- {:.3f}" 
        print(template.format(np.mean(exp_avg_final_glob), np.std(exp_avg_final_glob)), file=text_file)

        template = "-- Global Best Acc: {:.3f} +- {:.3f}"
        print(template.format(np.mean(exp_best_glob), np.std(exp_best_glob)), file=text_file)

        template = ("-- Avg Final Local Acc: {:3.2f} +- {:.3f}")
        print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)), file=text_file)

        template = ("-- Avg Best Local Acc: {:3.2f} +- {:.3f}")
        print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)), file=text_file)

        template = '-- pFedMe_Curr+ Fine Tuning Clients AVG Local Acc: {:.2f} +- {:.2f}'
        print(template.format(np.mean(exp_fedavg_ft_client_local), np.std(exp_fedavg_ft_client_local)), file=text_file)
        template = '-- pFedMe_Curr+ Fine Tuning Clients AVG Global Acc: {:.2f} +- {:.2f}'
        print(template.format(np.mean(exp_fedavg_ft_client_global), np.std(exp_fedavg_ft_client_global)), file=text_file)
        
        print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes', file=text_file)
        print('*'*40)
        
    return