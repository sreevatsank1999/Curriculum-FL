import argparse
    
## CIFAR-10 has 50000 training images (5000 per class), 10 classes, 10000 test images (1000 per class)
## CIFAR-100 has 50000 training images (500 per class), 100 classes, 10000 test images (100 per class)
## MNIST has 60000 training images (min: 5421, max: 6742 per class), 10000 test images (min: 892, max: 1135
## per class) --> in the code we fixed 5000 training image per class, and 900 test image per class to be 
## consistent with CIFAR-10 

## CIFAR-10 Non-IID 250 samples per label for 2 class non-iid is the benchmark (500 samples for each client)

def train_model_args_parser():
    parser = argparse.ArgumentParser()
    # random seed
    parser.add_argument('--seed', type=int, default=202207, help="random number generator seed (default: 202207)")
    
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    # SFDA arguments
    parser.add_argument('--num_epoch', type=int, default=10, help="the number of epochs")
    parser.add_argument('--batch_size', type=int, default=10, help="batch size")
    parser.add_argument('--lr0', type=float, default=0.01, help="init learning rate")
    parser.add_argument('--lr_sched_a', type=float, default=0.01, help="learning rate exp decay param a")
    parser.add_argument('--lr_sched_b', type=float, default=0.01, help="learning rate exp decay param b")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--w_decay', type=float, default=0.5, help="Target domain alignment weight decay")

    # model arguments
    parser.add_argument('--model', type=str, default='lenet5', help='model name')

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset: mnist, cifar10, cifar100")
    parser.add_argument('--alg', type=str, default='sfda_de', help='Algorithm')
    
    parser.add_argument('--savedir', type=str, default='../save/', help='save directory')
    parser.add_argument('--datadir', type=str, default='../data/', help='data directory')
    parser.add_argument('--logdir', type=str, default='../logs/', help='logs directory')
    parser.add_argument('--log_filename', type=str, default=None, help='The log file name')
    parser.add_argument('--ptdir', type=str, default='../pretrain/', help='pretrained model dir')
        
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--exp_label', type=str, default='test', help='Experiment Label')
    
    parser.add_argument('--load_initial', type=str, default='', help='define initial model path')

    args = parser.parse_args()
    return args
