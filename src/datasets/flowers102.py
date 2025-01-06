from mimetypes import init
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
from torchvision.datasets import Food101
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, download_and_extract_archive, download_url
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

from pathlib import Path


class Flowers102_manager:
    _instances = {}
    @staticmethod
    def get_instance(root, train, transform, target_transform, download=False):
        if train not in Flowers102_manager._instances:
            Flowers102_manager._instances[train] = Flowers102(root, train, transform, target_transform, download=download);
        return Flowers102_manager._instances[train]
    
    
class Flowers102_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        super(Flowers102_truncated,self).__init__()
        
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.dataobj,self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        
        dataobj = Flowers102_manager.get_instance(self.root, self.train, self.transform, self.target_transform, download=self.download)

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

class Flowers102(VisionDataset):
    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {False: "trnid", True: "tstid"} # Note this intentionally flipped. The original dataset is incorrecly flipped.

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False,
    ):
        super(Flowers102, self).__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"
        self._split = train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        self.target = []
        self.data = []
        for image_id in image_ids:
            self.target.append(image_id_to_label[image_id])
            self.data.append(self._images_folder / f"image_{image_id:05d}.jpg")

        
        #check if file exists
        dataset = 'train' if train else 'test';
        if not os.path.exists(f'{self._base_folder}/.cache/{dataset}.pt'):
            self.cache(dataset);

        img_cache = torch.load(f'{self._base_folder}/.cache/{dataset}.pt');
        self.img_cache = img_cache;
        
        self._labels = torch.Tensor(self.target);

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_file, target = self.data[idx], self.target[idx]
        # sample = PIL.Image.open(image_file).convert("RGB")
        sample = self.img_cache[image_file];
        
        # if self.transform:
        #     sample = self.transform(sample)

        # if self.target_transform:
        #     target = self.target_transform(target)

        return sample, target

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)
    
        
    def cache(self, dataset, batch_size=128):
        
        preprocessed_data = {};
        batch_img = torch.Tensor();
        batch_img_name = [];
        # toTensor = transforms.ToTensor();
        with tqdm(total=len(self.data), desc='Loading Images') as pbar:
            for img_name in self.data:
                img = Image.open(img_name).convert('RGB');
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
                os.mkdir(f'{self._base_folder}/.cache');
            except Exception as _:
                pass
        
        torch.save(preprocessed_data, f'{self._base_folder}/.cache/{dataset}.pt');
        

