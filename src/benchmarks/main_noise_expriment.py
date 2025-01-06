import sys
import os
from torch.utils.tensorboard import SummaryWriter

from src.data import *
from src.models import *
from src.client import * 
from src.clustering import *
from src.utils import * 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()

def main_noise_experiment(args,log=None):
    
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
    print('-'*40)
    
    torch.backends.cudnn.benchmark = True;     
    
    net_expert.eval();
    net_expert.to(args.device)    
    criterion = nn.CrossEntropyLoss(reduction="sum").to(args.device);
    
    nb_samp=1;    
    indx=np.random.permutation(list(range(len(train_ds_global))))[:nb_samp];
    
    path = args.logdir + args.exp_label + '/' + args.alg +'/' + args.dataset + '/' + args.partition + '/' + args.model + '/' 
    
    loss_n = [];
    for n in (0.0, 0.05, 0.1, 0.2, 0.5):
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., n, None, 0)
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]);
        # transform_train = transforms.Compose([
        #                                 transforms.Resize(256),
        #                                 transforms.CenterCrop(224),
        #                                 transforms.ToTensor(),
        #                                 # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 #                         std=[0.229, 0.224, 0.225])
        #                                 AddGaussianNoise(0., n, None, 0)
        #                             ]);
        train_ds_global.transform = transform_train;
        images=None;labels=np.array([]);
        for j in range(nb_samp):
            image,label = train_ds_global[indx[j]];
            if images is None:
                images = image.unsqueeze(0);
            else:
                images = torch.cat((images,image.unsqueeze(0)));
            labels=np.append(labels,label);

            
        images=images.to(args.device);
        labels=torch.from_numpy(labels).type(torch.LongTensor).to(args.device);
        
        with torch.no_grad():
            output = net_expert(images);
            loss = criterion(output,labels);
            
            loss_n.append(loss.cpu().numpy());

        for j in range(nb_samp):
            img = images[j].transpose(0,1).transpose(1,2).cpu().numpy();
            # img=(img+1)/2;
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img);
            plt.show();
            loss_r = round(loss.cpu().numpy().item(),4);
            plt.savefig(f'{path}/I{indx[j]}_n{n}_L{loss_r}.png')
            # break;
            
    return loss_n;
    
def run_noise_experiment(args, fname):
    alg_name = 'NoiseExp'
            
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
        
        loss_n = main_noise_experiment(args,log=log)    
        
        print('*'*40)
        print(' '*20, 'End of Trial %d'%(trial+1))
        print(' '*20, 'Final Results')
        
        template = "-- Loss N: {}" 
        print(template.format(loss_n))
        
        # template = "-- Intra Partition Difficulty (std): {:.3f}" 
        # print(template.format(exp_intrapartdiff[-1]))
        
        
    # print('*'*40)
    # print(' '*20, alg_name)
    # print(' '*20, 'Avg %d Trial Results'%args.ntrials)
    
    # template = "-- Inter Partition Difficulty: {:.3f} +- {:.3f}" 
    # print(template.format(np.mean(exp_interpartdiff), np.std(exp_interpartdiff)))

    # template = "-- Intra Partition Difficulty: {:.3f} +- {:.3f}" 
    # print(template.format(np.mean(exp_intrapartdiff), np.std(exp_intrapartdiff)))
    
    # with open(fname+'_results_summary.txt', 'a') as text_file:
    #     print('*'*40, file=text_file)
    #     print(' '*20, alg_name, file=text_file)
    #     print(' '*20, 'Avg %d Trial Results'%args.ntrials, file=text_file)
            
    #     template = "-- Inter Partition Difficulty: {:.3f} +- {:.3f}" 
    #     print(template.format(np.mean(exp_interpartdiff), np.std(exp_interpartdiff)),file=text_file)

    #     template = "-- Intra Partition Difficulty: {:.3f} +- {:.3f}" 
    #     print(template.format(np.mean(exp_intrapartdiff), np.std(exp_intrapartdiff)),file=text_file)
        
    #     print('*'*40)
        
    return