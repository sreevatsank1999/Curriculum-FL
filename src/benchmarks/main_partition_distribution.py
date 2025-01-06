import sys
import os
from torch.utils.tensorboard import SummaryWriter

from src.data import *
from src.models import *
from src.client import * 
from src.clustering import *
from src.utils import * 

def main_partition_distribution(args,log=None):
    
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
    else:
        print(f'Expert model required, not found at path: {expert_pt_path}');
        sys.exit();
        
    ##################################### Data partitioning section 
    print('-'*40)
    print('Getting Partition Data')

    net_dataidx_map, net_dataidx_map_test, \
    traindata_cls_counts, testdata_cls_counts = get_clients_data(args,train_ds_global,test_ds_global,data_score_net=net_expert)

    print('-'*40)    
    ################################# Initializing Partition   
    print('-'*40)
    print('Initializing Partition')
    clients = []
    for idx in range(args.num_partitions):
        sys.stdout.flush()
        print(f'-- Client {idx}, Labels Stat {traindata_cls_counts[idx]}')

        noise_level=0
        dataidxs = net_dataidx_map[idx]
        dataidxs_test = net_dataidx_map_test[idx]

        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, 
                                                                       args.datadir, args.local_bs, 32, 
                                                                       dataidxs, noise_level, 
                                                                       dataidxs_test=dataidxs_test)

        clients.append(Client_FedAvg_Curr_LG_Loss(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep, 
                                           args.lr0,args.lr_sched_a,args.lr_sched_b,args.momentum,args.w_decay,
                                           args.device, args.lg_scoring,
                                           train_ds_local, test_dl_local, 
                                           args.ordering, args.pacing_f, args.pacing_a, args.pacing_b, w_expert,
                                           args.log_clientnet))

    client_set = ClientSet(clients);
    ################################ Score Client Data
    Client_FedAvg_Curr_LG_Loss.score_data_indices = {i:np.random.randint(0,
                                                                          len(clients[i].lds_train),
                                                                          dtype=np.int32,
                                                                          size=(np.random.binomial(len(clients[i].lds_train),args.data_score_sample_p,))
                                                                          ) for i in range(args.num_users)}
    
    print('-'*40)
    
    total_budget = 1;
    
    client_curriculum = ClientScheduler(client_set,total_budget,ordering=args.client_ordering,scoring=args.lg_scoring,
                                        pacing_f=args.client_pacing_f,pacing_a=args.client_pacing_a,pacing_b=args.client_pacing_b);
    
    ################################ Score Partition
    ClientScheduler.score_client_indices = np.random.randint(0,
                                                             args.num_users,
                                                             dtype=np.int32,
                                                             size=(args.client_score_sample_n,))
        
    torch.backends.cudnn.benchmark = True;     
    
    current_client_set = client_curriculum.next_sched(w_glob=w_expert,curr_step=0,logger=log);
    
    order = [i for i in range(len(clients))]
    ind_loss_mean  = {};
    ind_loss_std  = {};

    for i in range(len(clients)):            
        _,stat = clients[i].scoring_func(w_glob=w_expert);
        indloss_mean = [np.mean([v[j] for k,v in stat.items()]) for j in range(len(list(stat.values())[0]))];
        indloss_std = [np.std([v[j] for k,v in stat.items()]) for j in range(len(list(stat.values())[0]))];
        
        ind_loss_mean[i]=indloss_mean[0];
        ind_loss_std[i]=indloss_std[0];

    if log:
        log.add_histogram(torch.tensor(list(ind_loss_mean.values())),f"score/Partitions");
        log.add_histogram(torch.tensor(list(ind_loss_std.values())),f"score/intraPartitionVariance");
    
    interpartdiff = np.std(list(ind_loss_mean.values()));
    intrapartdiff = np.mean(list(ind_loss_std.values()));
    
    return interpartdiff,intrapartdiff
    
    
def run_partition_distribution(args, fname):
    alg_name = 'PartitionDistribution'
            
    path = args.logdir + args.exp_label + '/' + args.alg +'/' + args.dataset + '/' + args.partition + '/' + args.model + '/'    
    mkdirs(path)    
    log = TBLogger(path + '/' + "tblog_"+ args.log_filename);        
    
    exp_interpartdiff=[]
    exp_intrapartdiff=[]
    
    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))
        
        log.trial_id = trial;
        log.global_step = 0;
        
        interpartdiff,intrapartdiff = main_partition_distribution(args,log=log)
        
        exp_interpartdiff.append(interpartdiff)
        exp_intrapartdiff.append(intrapartdiff)
        
        print('*'*40)
        print(' '*20, 'End of Trial %d'%(trial+1))
        print(' '*20, 'Final Results')
        
        template = "-- Inter Partition Difficulty (std): {:.3f}" 
        print(template.format(exp_interpartdiff[-1]))
        
        template = "-- Intra Partition Difficulty (std): {:.3f}" 
        print(template.format(exp_intrapartdiff[-1]))
        
        
    print('*'*40)
    print(' '*20, alg_name)
    print(' '*20, 'Avg %d Trial Results'%args.ntrials)
    
    template = "-- Inter Partition Difficulty: {:.3f} +- {:.3f}" 
    print(template.format(np.mean(exp_interpartdiff), np.std(exp_interpartdiff)))

    template = "-- Intra Partition Difficulty: {:.3f} +- {:.3f}" 
    print(template.format(np.mean(exp_intrapartdiff), np.std(exp_intrapartdiff)))
    
    with open(fname+'_results_summary.txt', 'a') as text_file:
        print('*'*40, file=text_file)
        print(' '*20, alg_name, file=text_file)
        print(' '*20, 'Avg %d Trial Results'%args.ntrials, file=text_file)
            
        template = "-- Inter Partition Difficulty: {:.3f} +- {:.3f}" 
        print(template.format(np.mean(exp_interpartdiff), np.std(exp_interpartdiff)),file=text_file)

        template = "-- Intra Partition Difficulty: {:.3f} +- {:.3f}" 
        print(template.format(np.mean(exp_intrapartdiff), np.std(exp_intrapartdiff)),file=text_file)
        
        print('*'*40)
        
    return