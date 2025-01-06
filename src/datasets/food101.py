from mimetypes import init
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
from torchvision.datasets import Food101
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity
from functools import partial
from typing import Optional, Callable
from torch.utils.model_zoo import tqdm
import PIL
import tarfile

import os
import os.path
import logging
import torchvision.datasets.utils as utils
from torch.utils.data import Subset
from scipy import ndimage

    
class FOOD101(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        super(FOOD101,self).__init__()
        
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.dataobj,self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        if self.train:
            train = "train"
        else: 
            train = "test";
        
        dataobj = Food101(self.root, train, self.transform, self.target_transform, self.download)

        # data = dataobj.data
        target = torch.Tensor(dataobj._labels)

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
        return img, target

    def __len__(self):
        return len(self.dataobj)
