import os
import numpy as np
import imageio
import glob
import re

import torch
from torch.utils.data import Dataset
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Callable,
    Tuple,
    TypeVar,
    Union
)

class WARWICKDataset(Dataset):
    """Warwick dataset."""

    def __init__(
        self,
        root_dir: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        save_npz: bool = True,
    ):
        self.train = train
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        
        data = []
        targets = []        

        if train:
            if os.path.isdir(f"{root_dir}/npz/train"):
                data= np.load(f"{root_dir}/npz/train/x_train.npy")
                targets = np.load(f"{root_dir}/npz/train/y_train.npy")
            else:
                for image_path in glob.glob(f"{root_dir}/Train/image_*.png"):
                    image_number = m = re.search('image_(.+?).png', image_path).group(1)
                    data.append(imageio.imread(f"{root_dir}/Train/image_{image_number}.png"))
                    targets.append(imageio.imread(f"{root_dir}/Train/label_{image_number}.png"))
                if save_npz:
                    os.makedirs(f"{root_dir}/npz/train")
                    np.save(f"{root_dir}/npz/train/x_train", data)
                    np.save(f"{root_dir}/npz/train/y_train", targets)
        else:
            if os.path.isdir(f"{root_dir}/npz/test"):
                data = np.load(f"{root_dir}/npz/test/x_test.npy")
                targets = np.load(f"{root_dir}/npz/test/y_test.npy")
            else:
                for image_path in glob.glob(f"{root_dir}/Test/image_*.png"):
                    image_number = m = re.search('image_(.+?).png', image_path).group(1)
                    data.append(imageio.imread(f"{root_dir}/Test/image_{image_number}.png"))
                    targets.append(imageio.imread(f"{root_dir}/Test/label_{image_number}.png"))
                if save_npz:
                    os.makedirs(f"{root_dir}/npz/test")
                    np.save(f"{root_dir}/npz/test/x_test", data)
                    np.save(f"{root_dir}/npz/test/y_test", targets)
                    
        self.data = np.array(data, dtype=np.float32)/255.0
        self.targets= np.array(targets, dtype=np.float32)/255.0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target