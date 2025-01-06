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

import csv
# from src.utils import mkdirs

class ISIC2019_manager:
    _instances = {}
    @staticmethod
    def get_instance(root, train, transform, target_transform):
        if train not in ISIC2019_manager._instances:
            ISIC2019_manager._instances[train] = ISIC2019(root, train, transform, target_transform);
        return ISIC2019_manager._instances[train]
    
class ISIC2019_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        super(ISIC2019_truncated,self).__init__()
        
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.dataobj,self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        
        dataobj = ISIC2019_manager.get_instance(self.root, self.train, self.transform, self.target_transform)

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


class ISIC2019(data.Dataset):
    
    def __init__(self, root, train, transform, target_transform):
        
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
            
        if train:
            self.data_root = f"{root}/ISIC2019/ISIC_2019_Training_Input";
        else:
            self.data_root = f"{root}/ISIC2019/ISIC_2019_Testing_Input";
        
        if train:
            self.gt_csv = os.path.join(f'{self.root}/ISIC2019', "ISIC_2019_Training_GroundTruth.csv");
        else:
            self.gt_csv = os.path.join(f'{self.root}/ISIC2019', "ISIC_2019_Testing_GroundTruth.csv");
        
        img_list = [];
        lbl_list = [];
        # Read csv and cache it
        with open(self.gt_csv, 'r') as f:
            reader = csv.reader(f,);
            header = next(reader);
            for l in reader:
                img_list.append(l[0]);
                lb_arr = np.array(l[1:-1])
                lbl = np.where( lb_arr == '1.0')[0];
                lbl_list.append(lbl.item());
        
        self.img_list = img_list;
        self.lbl_list = lbl_list;
        
        #check if file exists
        dataset = 'train' if train else 'test';
        if not os.path.exists(f'{self.root}/ISIC2019/.cache/{dataset}.pt'):
            self.cache(dataset);

        img_cache = torch.load(f'{self.root}/ISIC2019/.cache/{dataset}.pt');
        self.img_cache = img_cache;
        
        self._labels = torch.Tensor(lbl_list);
            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        # get line number idx from csv
        img_name = self.img_list[index];
        img = self.img_cache[img_name];
        
        target = self._labels[index];
        if self.target_transform:
            target = self.target_transform(target);
        
        return img, target

    def __len__(self):
        return len(self.img_list)      
    
    def cache(self, dataset, batch_size=128):
        
        preprocessed_data = {};
        batch_img = torch.Tensor();
        batch_img_name = [];
        # toTensor = transforms.ToTensor();
        with tqdm(total=len(self.img_list), desc='Loading Images') as pbar:
            for img_name in self.img_list:
                img = Image.open(os.path.join(self.data_root, img_name + ".jpg")).convert('RGB');
                img = self.transform(img);
                batch_img = torch.cat((batch_img, img.unsqueeze(0)), dim=0);
                batch_img_name.append(img_name);
                if len(batch_img) == batch_size:
                    # batch_transform_img = self.transform(batch_img);
                    batch_transform_img = batch_img;
                    preprocessed_data.update(dict(zip(batch_img_name, batch_transform_img)));
                    batch_img = torch.Tensor();
                    batch_img_name = [];
                pbar.update(1);
            # batch_transform_img = self.transform(batch_img);
            batch_transform_img = batch_img;
            preprocessed_data.update(dict(zip(batch_img_name, batch_transform_img)));
            pbar.update(len(batch_img_name));
            
            try:
                os.mkdir(f'{self.root}/ISIC2019/.cache');
            except Exception as _:
                pass
        
        torch.save(preprocessed_data, f'{self.root}/ISIC2019/.cache/{dataset}.pt');
        