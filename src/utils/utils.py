import os,sys
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import random
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from .datasets import MNIST_truncated, MNIST_rotated, CIFAR10_truncated, CIFAR10_rotated, CIFAR100_truncated, CIFAR100_levelnoise, SVHN_custom, \
FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData, ImageFolder_custom, USPS_truncated, \
STL10_truncated

from src.datasets import FOOD101, ISIC2019_truncated, Flowers102_truncated
from src.datasets import PathMNIST_truncated, DermaMNIST_truncated, BloodMNIST_truncated, TissueMNIST_truncated, OrganAMNIST_truncated, OrganCMNIST_truncated, OrganSMNIST_truncated
from src.data import *
from src.transform import *

from math import sqrt

import torch.nn as nn

import torch.optim as optim
import torchvision.utils as vutils
import time
import random
import copy 

import sklearn.datasets as sk
from sklearn.datasets import load_svmlight_file

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_usps_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    usps_train_ds = USPS_truncated(datadir, train=True, download=True, transform=transform)
    usps_test_ds = USPS_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = usps_train_ds.data, usps_train_ds.target
    X_test, y_test = usps_test_ds.data, usps_test_ds.target

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return (X_train, y_train, X_test, y_test)

def load_mnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_mnist_rotated_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_rotated(datadir, rotation=0, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_rotated(datadir, rotation=0, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_fmnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_svhn_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    svhn_train_ds = SVHN_custom(datadir, train=True, download=True, transform=transform)
    svhn_test_ds = SVHN_custom(datadir, train=False, download=True, transform=transform)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    # X_train = X_train.data.numpy()
    # y_train = y_train.data.numpy()
    # X_test = X_test.data.numpy()
    # y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar10_rotated_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_rotated(datadir, rotation=0, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_rotated(datadir, rotation=0, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_food101_data(datadir):    

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])

    food101_train_ds = FOOD101(datadir, train=True, download=True, transform=transform)
    food101_test_ds = FOOD101(datadir, train=False, download=True, transform=transform)

    y_train =  torch.tensor(food101_train_ds.dataobj._labels);
    y_test =  torch.tensor(food101_test_ds.dataobj._labels);
    X_train =[]; X_test =[];

    return (X_train, y_train, X_test, y_test)

def load_food101n_data(datadir):

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])

    # Food101N Dataset link (https://iudata.blob.core.windows.net/food101/Food-101N_release.zip)

    food101n_train_ds = ImageFolder_custom(datadir+'food101n/train/', transform=transform)
    food101n_test_ds = ImageFolder_custom(datadir+'food101n/test/', transform=transform)

    X_train, y_train = food101n_train_ds.data, food101n_train_ds.target
    X_test, y_test = food101n_test_ds.data, food101n_test_ds.target

    return (X_train, y_train, X_test, y_test)

def load_miniimagenet_data(datadir):
    
    print(datadir)
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])
    
    # Mini-ImageNet Dataset link (https://drive.google.com/drive/folders/17a09kkqVivZQFggCw9I_YboJ23tcexNM?usp=sharing)
    
    xray_train_ds = ImageFolder_custom(datadir+'mini-imagenet/train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'mini-imagenet/test/', transform=transform)

    X_train, y_train = torch.Tensor([s[0] for s in xray_train_ds.samples]), torch.Tensor([s[1] for s in xray_train_ds.samples])
    X_test, y_test = torch.Tensor([s[0] for s in xray_test_ds.samples]), torch.Tensor([s[1] for s in xray_test_ds.samples])

    y_train = F.one_hot(y_train);
    y_test = F.one_hot(y_test);
    
    return (X_train, y_train, X_test, y_test)

def load_tinyimagenet_data(datadir):
    print(datadir)
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'tiny-imagenet-200/train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'tiny-imagenet-200/val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

def load_celeba_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    celeba_train_ds = CelebA_custom(datadir, split='train', target_type="attr", download=True, transform=transform)
    celeba_test_ds = CelebA_custom(datadir, split='test', target_type="attr", download=True, transform=transform)

    gender_index = celeba_train_ds.attr_names.index('Male')
    y_train =  celeba_train_ds.attr[:,gender_index:gender_index+1].reshape(-1)
    y_test = celeba_test_ds.attr[:,gender_index:gender_index+1].reshape(-1)

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (None, y_train, None, y_test)

def load_femnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FEMNIST(datadir, train=True, transform=transform, download=True)
    mnist_test_ds = FEMNIST(datadir, train=False, transform=transform, download=True)

    X_train, y_train, u_train = mnist_train_ds.data, mnist_train_ds.targets, mnist_train_ds.users_index
    X_test, y_test, u_test = mnist_test_ds.data, mnist_test_ds.targets, mnist_test_ds.users_index

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    u_train = np.array(u_train)
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    u_test = np.array(u_test)

    return (X_train, y_train, u_train, X_test, y_test, u_test)

def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {int(unq[i]): unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    #logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(dataset, train_ds, test_ds, logdir, partition, n_parties, ordering='rand', ordering_f=1.0, score_net=None, device=torch.device('cpu'), beta=0.4, local_view=False):
    #np.random.seed(2020)
    #torch.manual_seed(2020)

    y_train = train_ds.target;
    n_train = len(train_ds);
    
    train_dl = data.DataLoader(dataset=train_ds, batch_size=64, pin_memory=False, shuffle=False, drop_last=False,num_workers=8,persistent_workers=False)
    
    if score_net:
        _,_,ind_loss = eval_on_dataloader(score_net,train_dl,device);
        # Nord = int(ordering_f*len(ind_loss));
        # Pord = np.random.permutation(list(range(len(ind_loss))));
        # order = np.argsort(ind_loss[Pord[:Nord]]);      # Ordered part
        order = np.argsort(ind_loss);      # Ordered part
        if ordering == "inc":
            order=order
        elif ordering == "rand":
            np.random.shuffle(order)
        elif  ordering == "dec":
            order = np.flip(order);
        else:
            print(f'Partition ordering={ordering} Does Not Exist')
            sys.exit()
        # order=np.append(order, Pord[Nord:]);       # Unordered part
    else:
        if ordering=='rand':
            order = np.random.permutation(n_train)
        else:
            print(f'Partition ordering={ordering} Cannot be computed as score_net=None')
            sys.exit()
    order_invmap = np.argsort(order)
            
    y_train = y_train[order];

    if partition == "homo":
        # idxs = np.random.permutation(n_train)
        idxs = order
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
        elif dataset in ('cifar100','cifar100-lvln','miniimagenet'):
            K = 100
        elif dataset in ('isic2019','bloodmnist','tissuemnist'):
            K = 8
        elif dataset in ('pathmnist'):
            K = 9
        elif dataset in ('dermamnist'):
            K = 7
        elif dataset in ('organamnist','organcmnist','organsmnist'):
            K = 11
        elif dataset == 'tinyimagenet':
            K = 200
        elif dataset == 'flowers102':
            K = 102
        elif dataset in ('food101','food101n'):
            K = 101

        #np.random.seed(2021)
        net_dataidx_map = {}

        beta = beta*K;
        
        idx_k = [[] for _ in range(K)]
        for k in range(K):
            idx_k[k] = np.where(y_train == k)[0]

        cls_p = np.array([len(idx_k[k]) for k in range(K)]);
        cls_p = cls_p / cls_p.sum();
        
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            proportions = np.random.dirichlet(beta*cls_p,size=(n_parties,)).transpose()
            for k in range(K):
                # np.random.shuffle(idx_k[k])
                proportions_k = proportions[k] / proportions[k].sum()
                
                proportions_k = (np.cumsum(proportions_k) * len(idx_k[k])).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k[k], proportions_k))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = order[idx_batch[j]]

    elif partition[:13] == "noniid-#label":
        num = eval(partition[13:])
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
        elif dataset in ('cifar100','miniimagenet'):
            K = 100
        elif dataset in ('isic2019','bloodmnist','tissuemnist'):
            K = 8
        elif dataset in ('pathmnist'):
            K = 9
        elif dataset in ('dermamnist'):
            K = 7
        elif dataset in ('organamnist','organcmnist','organsmnist'):
            K = 11
        elif dataset == 'tinyimagenet':
            K = 200
        elif dataset in ('food101','food101n'):
            K = 101
            
        print(f'K: {K}')
        if num == 10:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k = np.where(y_train==i)[0]
                # np.random.shuffle(idx_k)
                split = np.array_split(idx_k,n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(K)]
            contain=[]
            for i in range(n_parties):
                current=[i%K]
                times[i%K]+=1
                j=1
                while (j<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                if times[i] == 0:
                    continue;
                idx_k = np.where(y_train==i)[0]
                # np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        ids+=1
        
        for j in range(n_parties):
            np.random.shuffle(net_dataidx_map[j])
            net_dataidx_map[j] = order[net_dataidx_map[j]]
        
    elif partition[:14] == "noniid1-#label":
        print('Modified Non-IID partitioning')
        num = eval(partition[14:])
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
        elif dataset in ('cifar100','miniimagenet'):
            K = 100
        elif dataset in ('isic2019','bloodmnist','tissuemnist'):
            K = 8
        elif dataset in ('pathmnist'):
            K = 9
        elif dataset in ('dermamnist'):
            K = 7
        elif dataset in ('organamnist','organcmnist','organsmnist'):
            K = 11
        elif dataset == 'tinyimagenet':
            K = 200
        elif dataset in ('food101','food101n'):
            K = 101
            
        print(f'Dataset {dataset}, K: {K}, {partition}')
        if num == 10:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k = np.where(y_train==i)[0]
                # np.random.shuffle(idx_k)
                split = np.array_split(idx_k,n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(K)]
            contain=[]
                
            #aa = np.random.randint(low=0, high=K, size=num)
            aa = np.random.choice(np.arange(K), size=num, replace=False)
            remain = np.delete(np.arange(K), aa)
            #print(f'Client 0 , {len(aa)}')
            #print(f'Unique a {len(np.unique(aa))}')
            #print(f'Unique remain {len(np.unique(remain))}')
            contain.append(copy.deepcopy(aa.tolist()))
            for el in aa:
                times[el]+=1
                
            for i in range(n_parties-1):
                x = np.random.randint(low=int(np.ceil(K/2)), high=K)
                y = np.random.randint(low=0, high=int(K/4)+1)

                rand = np.random.choice([0,1,2], size=1, replace=False)
                #print(rand)
                if rand == 0 or rand == 1:
                    s = int(np.ceil((x/K)*num))
                    if s==num and rand==0:
                        s = s-int(np.ceil(0.05*num))
                elif rand == 2:
                    s = int(np.ceil((y/K)*num))

                labels = np.random.choice(aa, size=s, replace=False).tolist()
                #print(f'Client {i} , {len(labels)}, S {s}')
                #print(labels)
                labels.extend(np.random.choice(remain, size=(num-s), replace=False).tolist())
                #print(f'Client {i+1} , {len(labels)}')
                #ccc = np.unique(labels)
                #print(f'Client {i+1} , {len(ccc)}')
                
                for el in labels:
                    times[el]+=1
                contain.append(labels)
                #print(len(labels))

            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                # np.random.shuffle(idx_k)
                #print(f'{i}: {times[i]}')
                split = np.array_split(idx_k,times[i])
                #print(f'len(split) {len(split)}, times[i] {times[i]}')
                ids=0
                for j in range(n_parties):
                    #print(f'Client {i}, {len(contain[j])}')
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        ids+=1
                
        for j in range(n_parties):
            np.random.shuffle(net_dataidx_map[j])
            net_dataidx_map[j] = order[net_dataidx_map[j]]
            
    elif partition == "iid-diff-quantity":
        # idxs = np.random.permutation(n_train)
        idxs = order
        min_size = 0
        while min_size < 10:
            proportions_k = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions_k = proportions_k/proportions_k.sum()
            min_size = np.min(proportions_k*len(idxs))
        proportions_k = (np.cumsum(proportions_k)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs,proportions_k)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "real" and dataset == "femnist":
        u_train = train_dl.dataset.users_index;
        num_user = u_train.shape[0]
        user = np.zeros(num_user+1,dtype=np.int32)
        for i in range(1,num_user+1):
            user[i] = user[i-1] + u_train[i-1]
        no = np.random.permutation(num_user)
        batch_idxs = np.array_split(no, n_parties)
        net_dataidx_map = {i:np.zeros(0,dtype=np.int32) for i in range(n_parties)}
        for i in range(n_parties):
            for j in batch_idxs[i]:
                net_dataidx_map[i]=np.append(net_dataidx_map[i], np.arange(user[j], user[j+1]))
    
    print(f'partition: {partition}')
    traindata_cls_counts = record_net_data_stats(y_train[order_invmap], net_dataidx_map, logdir)
    print('Data statistics Train:\n %s \n' % str(traindata_cls_counts))
    
    y_test = test_ds.target;
    if local_view:
        net_dataidx_map_test = {i: [] for i in range(n_parties)}
        for k_id, stat in traindata_cls_counts.items():
            labels = list(stat.keys())
            for l in labels:
                idx_k = np.where(y_test==l)[0]
                net_dataidx_map_test[k_id].extend(idx_k.tolist())

        testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test, logdir)
        print('Data statistics Test:\n %s \n' % str(testdata_cls_counts))
    else: 
        net_dataidx_map_test = None 
        testdata_cls_counts = None 

    return (net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts)

def get_clients_data(args,train_ds_global,test_ds_global,data_score_net=None):

    if args.partition[0:2] == 'sc':
        if args.dataset == 'cifar10':        
            if args.partition == 'sc_niid_dir':
                print('Loading CIFAR10 SuperClass NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR10_SuperClass_NIID_DIR(train_ds_global, test_ds_global, args)

            elif args.partition[0:7] == 'sc_niid':
                print('Loading CIFAR10 SuperClass NIID for all clients')

                num = eval(args.partition[7:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR10_SuperClass_NIID(train_ds_global, test_ds_global, num, args)
                
            elif args.partition == 'sc_old_niid_dir':
                print('Loading CIFAR10 SuperClass OLD NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR10_SuperClass_Old_NIID_DIR(train_ds_global, test_ds_global, args)
            
            elif args.partition[0:11] == 'sc_old_niid':
                print('Loading CIFAR10 SuperClass OLD NIID for all clients')

                num = eval(args.partition[11:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR10_SuperClass_Old_NIID(train_ds_global, test_ds_global, num, args)

        elif args.dataset == 'cifar100':
            if args.partition == 'sc_niid_dir':
                print('Loading CIFAR100 SuperClass NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR100_SuperClass_NIID_DIR(train_ds_global, test_ds_global, args)
                
            elif args.partition[0:7] == 'sc_niid':
                print('Loading CIFAR100 SuperClass NIID for all clients')

                num = eval(args.partition[7:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR100_SuperClass_NIID(train_ds_global, test_ds_global, args)
                
            elif args.partition == 'sc_old_niid_dir':
                print('Loading CIFAR100 SuperClass OLD NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR100_SuperClass_Old_NIID_DIR(train_ds_global, test_ds_global, args)
                
            elif args.partition[0:11] == 'sc_old_niid':
                print('Loading CIFAR100 SuperClass OLD NIID for all clients')

                num = eval(args.partition[11:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR100_SuperClass_Old_NIID(train_ds_global, test_ds_global, args)
        
        elif args.dataset == 'stl10':
            if args.partition == 'sc_niid_dir':
                print('Loading STL10 SuperClass NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = STL10_SuperClass_NIID_DIR(train_ds_global, test_ds_global, args)
                
            elif args.partition[0:7] == 'sc_niid':
                print('Loading STL10 SuperClass NIID for all clients')

                num = eval(args.partition[7:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = STL10_SuperClass_NIID(train_ds_global, test_ds_global, num, args)
                
            elif args.partition == 'sc_old_niid_dir':
                print('Loading STL10 SuperClass OLD NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = STL10_SuperClass_Old_NIID_DIR(train_ds_global, test_ds_global, args)
                
            elif args.partition[0:11] == 'sc_old_niid':
                print('Loading STL10 SuperClass OLD NIID for all clients')

                num = eval(args.partition[11:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = STL10_SuperClass_Old_NIID(train_ds_global, test_ds_global, num, args)
                
        elif args.dataset == 'fmnist':
            if args.partition == 'sc_niid_dir':
                pass
            elif args.partition[0:7] == 'sc_niid':
                print('Loading FMNIST SuperClass NIID for all clients')

                num = eval(args.partition[7:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = FMNIST_SuperClass_NIID(train_ds_global, test_ds_global, args)
    else:
        print(f'Loading {args.dataset}, {args.partition} for all clients')
        args.local_view = True
        net_dataidx_map, net_dataidx_map_test, \
        traindata_cls_counts, testdata_cls_counts = partition_data(args.dataset, train_ds_global, test_ds_global,
        args.logdir, args.partition, args.num_partitions, ordering=args.partition_difficulty_dist, ordering_f=args.partition_ordering_f,
            score_net=data_score_net, device=args.device, beta=args.beta, local_view=args.local_view)

    return net_dataidx_map, net_dataidx_map_test, \
            traindata_cls_counts, testdata_cls_counts



def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True
        
    model.to(device)
    
    w = model.state_dict()
    name = list(w.keys())[0]
    print(f'COMP ACC {w[name][0,0,0]}')
            
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
                
                #pred = out.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                #correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)

def eval_on_dataloader(net,ds_loader,device):            
    net.to(device)
    net.eval()
    test_loss = torch.tensor(0.0,device=device);
    ind_loss = torch.tensor(np.zeros(shape=(len(ds_loader.dataset))),device=device);
    correct = torch.tensor(0.0,device=device);
    criterion = nn.CrossEntropyLoss(reduction="none").to(device)
    with torch.no_grad():
        head=0;
        for i,(data,target) in enumerate(ds_loader):
            data, target = data.to(device), target.type(torch.LongTensor).to(device)
            bs= len(target);
            
            output = net(data)
            indloss=criterion(output, target)
            ind_loss[head:head+bs] = indloss;
            test_loss += torch.sum(indloss)  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().sum()
            
            head+=bs;
    test_loss /= len(ds_loader.dataset)
    accuracy = 100. * correct / len(ds_loader.dataset)
    return test_loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy(),ind_loss.detach().cpu().numpy()

def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def load_model(model, model_index, device="cpu"):
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, num_workers=1, net_id=None, total=0, dataidxs_test=None,
                  same_size=False, target_transform=None, rotation=0):
    train_ds=None; test_ds=None;
    
    if dataset in ('mnist', 'mnist_rotated', 'femnist', 'fmnist', 'cifar10', 'cifar10_rotated', 'cifar100','cifar100-lvln',
                   'svhn', 'tinyimagenet', 'generated', 'covtype', 'a9a', 'rcv1', 'SUSY', 'usps', 'stl10', 'isic2019',
                   'bloodmnist','tissuemnist','pathmnist','dermamnist','organamnist','organcmnist','organsmnist',
                   'food101', 'food101n', 'miniimagenet','flowers102'):
        if dataset == 'mnist' or dataset == 'mnist_rotated':
            if dataset == 'mnist':
                ds_obj = MNIST_truncated
            elif dataset == 'mnist_rotated':
                ds_obj = MNIST_rotated
            
            if same_size:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Pad(2, fill=0, padding_mode='constant'),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    AddGaussianNoise(0., noise_level, net_id, total), 
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Pad(2, fill=0, padding_mode='constant'),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    AddGaussianNoise(0., noise_level, net_id, total),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            else: 
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    AddGaussianNoise(0., noise_level, net_id, total), 
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    AddGaussianNoise(0., noise_level, net_id, total),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

        elif dataset == 'femnist':
            ds_obj = FEMNIST
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize((0.1307,), (0.3081,))
             ])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        elif dataset in ('bloodmnist','tissuemnist','pathmnist','dermamnist','organamnist','organcmnist','organsmnist'):
            if dataset == 'bloodmnist':
                ds_obj = BloodMNIST_truncated
            elif dataset == 'tissuemnist':
                ds_obj = TissueMNIST_truncated
            elif dataset == 'pathmnist':
                ds_obj = PathMNIST_truncated
            elif dataset == 'dermamnist':   
                ds_obj = DermaMNIST_truncated 
            elif dataset == 'organamnist':
                ds_obj = OrganAMNIST_truncated
            elif dataset == 'organcmnist':
                ds_obj = OrganCMNIST_truncated
            elif  dataset == 'organsmnist':
                ds_obj = OrganSMNIST_truncated
                
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                # transforms.Normalize((0.1307,), (0.3081,))
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                # transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            
        elif dataset == 'fmnist':
            ds_obj = FashionMNIST_truncated
            
            if same_size:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Pad(2, fill=0, padding_mode='constant'),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    AddGaussianNoise(0., noise_level, net_id, total), 
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Pad(2, fill=0, padding_mode='constant'),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    AddGaussianNoise(0., noise_level, net_id, total), 
                    transforms.Normalize((0.1307,), (0.3081,))
                 ])
            else: 
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    AddGaussianNoise(0., noise_level, net_id, total), 
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    AddGaussianNoise(0., noise_level, net_id, total), 
                    transforms.Normalize((0.1307,), (0.3081,))
                 ])

        elif dataset == 'svhn':
            ds_obj = SVHN_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])

        elif dataset == 'cifar10' or dataset == 'cifar10_rotated':
            if dataset == 'cifar10': 
                ds_obj = CIFAR10_truncated
            elif dataset == 'cifar10_rotated':
                ds_obj = CIFAR10_rotated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Lambda(lambda x: F.pad(
                #    Variable(x.unsqueeze(0), requires_grad=False),
                #    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                #transforms.ToPILImage(),
                #transforms.RandomCrop(32),
                #transforms.RandomHorizontalFlip(),
                #transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        elif dataset == 'cifar100':
            ds_obj = CIFAR100_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Lambda(lambda x: F.pad(
                #    Variable(x.unsqueeze(0), requires_grad=False),
                #    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                #transforms.ToPILImage(),
                #transforms.RandomCrop(32),
                #transforms.RandomHorizontalFlip(),
                #transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])

        elif dataset == 'isic2019':
            ds_obj = ISIC2019_truncated

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            transform_train = transforms.Compose([
                # transforms.ToTensor(),
                #transforms.Lambda(lambda x: F.pad(
                #    Variable(x.unsqueeze(0), requires_grad=False),
                #    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                # transforms.ToPILImage(),                
                transforms.ToTensor(),
                # ToCUDA(device),
                transforms.Resize(450),
                transforms.CenterCrop(360),
                transforms.Resize(256),
                transforms.RandomCrop(224)
                #transforms.RandomHorizontalFlip(),
                # AddGaussianNoise(0., noise_level, net_id, total)
                # transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.ToTensor(),
                # ToCUDA(device),
                transforms.Resize(450),
                transforms.CenterCrop(360),
                transforms.Resize(256),
                transforms.CenterCrop(224)
                # transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
        
        elif dataset == 'cifar100-lvln':
            ds_obj = CIFAR100_levelnoise

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Lambda(lambda x: F.pad(
                #    Variable(x.unsqueeze(0), requires_grad=False),
                #    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                #transforms.ToPILImage(),
                #transforms.RandomCrop(32),
                #transforms.RandomHorizontalFlip(),
                #transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)
                # transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)
                # transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
            train_ds = ds_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train,rngseed=1,nlevels=[0.0,0.05,0.1,0.2])
            test_ds = ds_obj(datadir, train=False, transform=transform_test,rngseed=1,nlevels=[0.0,0.05,0.1,0.2])
        
        elif dataset == 'stl10':
            ds_obj = STL10_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
            
        elif dataset == 'tinyimagenet':
            ds_obj = ImageFolder_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            train_ds = ds_obj(datadir+'tiny-imagenet-200/train/', dataidxs=dataidxs, transform=transform_train)
            test_ds = ds_obj(datadir+'tiny-imagenet-200/test/', transform=transform_test)
            
        elif dataset == 'usps': 
            ds_obj = USPS_truncated
            
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Pad(8, fill=0, padding_mode='constant'),
                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Pad(8, fill=0, padding_mode='constant'),
                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize((0.1307,), (0.3081,))
             ])

        elif dataset == 'flowers102':
            ds_obj = Flowers102_truncated
            transform_train = transforms.Compose([
                transforms.Resize(256),
                # transforms.RandomRotation(30),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), # For GPU purpose
                # As we are going to do transfer learning with a ImageNet pretrained VGG
                # so here we normalize the dataset being used here with the ImageNet stats
                # for better transfer learning performance
                transforms.Normalize([0.485, 0.456, 0.406], # RGB mean & std estied on ImageNet
                                     [0.229, 0.224, 0.225])
            ])

            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], # RGB mean & std estied on ImageNet
                                     [0.229, 0.224, 0.225])
            ])

        elif dataset == 'food101': 
            ds_obj = FOOD101;
            
            transform_train = transforms.Compose([
                                                    transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])
                                                ]);
            transform_test = transform_train;
            
        elif dataset == 'food101n': 
            ds_obj = ImageFolder_custom;
            
            transform_train = transforms.Compose([
                                                    transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])
                                                ]);
            transform_test = transform_train;
            
        elif dataset == 'miniimagenet': 
            ds_obj = ImageFolder_custom;
            
            transform_train = transforms.Compose([
                                                    transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])
                                                ]);
            transform_test = transform_train;
            
        else:   
            ds_obj = Generated
            transform_train = None
            transform_test = None
        
        if not train_ds or not test_ds:
            if dataset == 'mnist_rotated' or dataset == 'cifar10_rotated':
                train_ds = ds_obj(datadir, rotation=rotation, dataidxs=dataidxs, train=True, transform=transform_train, 
                                target_transform=target_transform, download=True)
                test_ds = ds_obj(datadir, rotation=rotation, dataidxs=dataidxs_test, train=False, transform=transform_test, 
                                target_transform=target_transform, download=True)
            else:
                train_ds = ds_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, 
                                    target_transform=target_transform, download=True)
                test_ds = ds_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, 
                                    target_transform=target_transform, download=True)
            
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, pin_memory=False, shuffle=True, drop_last=False,num_workers=num_workers,persistent_workers=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, pin_memory=False, shuffle=False, drop_last=False,num_workers=num_workers,persistent_workers=True)

    return train_dl, test_dl, train_ds, test_ds


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


def noise_sample(choice, n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.

    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    z = torch.randn(batch_size, n_z, 1, 1, device=device)
    idx = np.zeros((n_dis_c, batch_size))
    if(n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)

        c_tmp = np.array(choice)

        for i in range(n_dis_c):
            idx[i] = np.random.randint(len(choice), size=batch_size)
            for j in range(batch_size):
                idx[i][j] = c_tmp[int(idx[i][j])]

            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if(n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx


