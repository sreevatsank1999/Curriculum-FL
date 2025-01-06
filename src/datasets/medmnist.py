from mimetypes import init
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
from torchvision.datasets import Food101
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity
from functools import partial
from typing import Optional, Callable
from torch.utils.model_zoo import tqdm
import PIL
import tarfile
from tqdm import tqdm

import os
import os.path
import logging
import torchvision.datasets.utils as utils
from torch.utils.data import Subset
from scipy import ndimage

from medmnist import PathMNIST, DermaMNIST, BloodMNIST, TissueMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST

import csv
# from src.utils import mkdirs


class MedMNIST_manager:
    _instances = {}
    @staticmethod
    def get_instance(root, dataset, train, transform, target_transform, download=False):
        split = 'train' if train else 'test';
        
        key = f'{dataset}_{split}';
        if key not in MedMNIST_manager._instances:
            if dataset == 'pathmnist':
                MedMNIST_manager._instances[key] = PathMNIST(root=root, split=split, transform=transform, target_transform=target_transform, download=download);
            elif dataset == 'dermamnist':
                MedMNIST_manager._instances[key] = DermaMNIST(root=root, split=split, transform=transform, target_transform=target_transform, download=download);
            elif dataset == 'bloodmnist':
                MedMNIST_manager._instances[key] = BloodMNIST(root=root, split=split, transform=transform, target_transform=target_transform, download=download);
            elif dataset == 'tissuemnist':
                MedMNIST_manager._instances[key] = TissueMNIST(root=root, split=split, transform=transform, target_transform=target_transform, download=download);
            elif dataset == 'organamnist':
                MedMNIST_manager._instances[key] = OrganAMNIST(root=root, split=split, transform=transform, target_transform=target_transform, download=download);
            elif dataset == 'organcmnist':
                MedMNIST_manager._instances[key] = OrganCMNIST(root=root, split=split, transform=transform, target_transform=target_transform, download=download);
            elif dataset == 'organsmnist':
                MedMNIST_manager._instances[key] = OrganSMNIST(root=root, split=split, transform=transform, target_transform=target_transform, download=download);
                        
        return MedMNIST_manager._instances[key]

class PathMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        super(PathMNIST_truncated,self).__init__()
        
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset_base = f'{self.root}/PathMNIST';
        try:
            os.mkdir(self.dataset_base)
        except:
            pass;

        self.dataobj,self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        split = 'train' if self.train else 'test';
        dataobj = MedMNIST_manager.get_instance(self.dataset_base,'pathmnist',self.train,self.transform,self.target_transform,download=self.download)

        # data = dataobj.data
        target = torch.Tensor(dataobj.labels).squeeze(1);

        if self.dataidxs is not None:
            dataobj = Subset(dataobj, self.dataidxs);
            target = target[self.dataidxs]

        return dataobj,target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataobj[index];
        return img, target.squeeze(0);

    def __len__(self):
        return len(self.dataobj)

    
class DermaMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        super(DermaMNIST_truncated,self).__init__()
        
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset_base = f'{self.root}/DermaMNIST';
        try:
            os.mkdir(self.dataset_base)
        except:
            pass;

        self.dataobj,self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        split = 'train' if self.train else 'test';
        dataobj = MedMNIST_manager.get_instance(self.dataset_base,'dermamnist',self.train,self.transform,self.target_transform,download=self.download)

        # data = dataobj.data
        target = torch.Tensor(dataobj.labels).squeeze(1);

        if self.dataidxs is not None:
            dataobj = Subset(dataobj, self.dataidxs);
            target = target[self.dataidxs]

        return dataobj,target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataobj[index];
        return img, target.squeeze(0);

    def __len__(self):
        return len(self.dataobj)

class BloodMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        super(BloodMNIST_truncated,self).__init__()
        
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset_base = f'{self.root}/BloodMNIST';
        try:
            os.mkdir(self.dataset_base)
        except:
            pass;

        self.dataobj,self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        split = 'train' if self.train else 'test';
        dataobj = MedMNIST_manager.get_instance(self.dataset_base,'bloodmnist',self.train,self.transform,self.target_transform,download=self.download)

        # data = dataobj.data
        target = torch.Tensor(dataobj.labels).squeeze(1);

        if self.dataidxs is not None:
            dataobj = Subset(dataobj, self.dataidxs);
            target = target[self.dataidxs]

        return dataobj,target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataobj[index];
        return img, target.squeeze(0);

    def __len__(self):
        return len(self.dataobj)


class TissueMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        super(TissueMNIST_truncated,self).__init__()
        
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset_base = f'{self.root}/TissueMNIST';
        try:
            os.mkdir(self.dataset_base)
        except:
            pass;

        self.dataobj,self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        split = 'train' if self.train else 'test';
        dataobj = MedMNIST_manager.get_instance(self.dataset_base,'tissuemnist',self.train,self.transform,self.target_transform,download=self.download)

        # data = dataobj.data
        target = torch.Tensor(dataobj.labels).squeeze(1);

        if self.dataidxs is not None:
            dataobj = Subset(dataobj, self.dataidxs);
            target = target[self.dataidxs]

        return dataobj,target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataobj[index];
        return img, target.squeeze(0);

    def __len__(self):
        return len(self.dataobj)



class OrganAMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        super(OrganAMNIST_truncated,self).__init__()
        
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset_base = f'{self.root}/OrganAMNIST';
        try:
            os.mkdir(self.dataset_base)
        except:
            pass;

        self.dataobj,self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        split = 'train' if self.train else 'test';
        dataobj = MedMNIST_manager.get_instance(self.dataset_base,'organamnist',self.train,self.transform,self.target_transform,download=self.download)

        # data = dataobj.data
        target = torch.Tensor(dataobj.labels).squeeze(1);

        if self.dataidxs is not None:
            dataobj = Subset(dataobj, self.dataidxs);
            target = target[self.dataidxs]

        return dataobj,target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataobj[index];
        return img, target.squeeze(0);

    def __len__(self):
        return len(self.dataobj)



class OrganCMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        super(OrganCMNIST_truncated,self).__init__()
        
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset_base = f'{self.root}/OrganCMNIST';
        try:
            os.mkdir(self.dataset_base)
        except:
            pass;

        self.dataobj,self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        split = 'train' if self.train else 'test';
        dataobj = MedMNIST_manager.get_instance(self.dataset_base,'organcmnist',self.train,self.transform,self.target_transform,download=self.download)

        # data = dataobj.data
        target = torch.Tensor(dataobj.labels).squeeze(1);

        if self.dataidxs is not None:
            dataobj = Subset(dataobj, self.dataidxs);
            target = target[self.dataidxs]

        return dataobj,target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataobj[index];
        return img, target.squeeze(0);

    def __len__(self):
        return len(self.dataobj)



class OrganSMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        super(OrganSMNIST_truncated,self).__init__()
        
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset_base = f'{self.root}/OrganSMNIST';
        try:
            os.mkdir(self.dataset_base)
        except:
            pass;

        self.dataobj,self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        split = 'train' if self.train else 'test';
        dataobj = MedMNIST_manager.get_instance(self.dataset_base,'organsmnist',self.train,self.transform,self.target_transform,download=self.download)

        # data = dataobj.data
        target = torch.Tensor(dataobj.labels).squeeze(1);

        if self.dataidxs is not None:
            dataobj = Subset(dataobj, self.dataidxs);
            target = target[self.dataidxs]

        return dataobj,target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataobj[index];
        return img, target.squeeze(0);

    def __len__(self):
        return len(self.dataobj)

