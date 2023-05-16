import os
import numpy as np
import imageio
import glob
import re

import torch
from torch.utils.data import Dataset
from typing import (
    Optional,
    Callable,
)

class WARWICKDataset(Dataset):
    """Warwick dataset."""

    def __init__(
        self,
        root_dir: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        n_train_images: int = 85,
        n_test_images: int = 60,
        save_npz: bool = True,
        train_npz: str = "npz/train",
        test_npz: str = "npz/test",
        run_device: str = "cpu",
    ):
        self.train = train
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        #self.run_device = run_device

        data = []
        targets = []        

        if train:
            if os.path.isdir(f"{root_dir}/{train_npz}"):
                data= np.load(f"{root_dir}/{train_npz}/x_train.npy")
                targets = np.load(f"{root_dir}/{train_npz}/y_train.npy")
            else:
                for indx in range(n_train_images):
                    data.append(imageio.imread(f"{root_dir}/Train/image_{'{:02}'.format(indx+1)}.png"))
                    targets.append(imageio.imread(f"{root_dir}/Train/label_{'{:02}'.format(indx+1)}.png"))
                if save_npz:
                    os.makedirs(f"{root_dir}/{train_npz}")
                    np.save(f"{root_dir}/{train_npz}/x_train", data)
                    np.save(f"{root_dir}/{train_npz}/y_train", targets)
        else:
            if os.path.isdir(f"{root_dir}/{test_npz}"):
                data = np.load(f"{root_dir}/{test_npz}/x_test.npy")
                targets = np.load(f"{root_dir}/{test_npz}/y_test.npy")
            else:
                for indx in range(n_test_images):
                    data.append(imageio.imread(f"{root_dir}/Test/image_{'{:02}'.format(indx+1)}.png"))
                    targets.append(imageio.imread(f"{root_dir}/Test/label_{'{:02}'.format(indx+1)}.png"))
                if save_npz:
                    os.makedirs(f"{root_dir}/{test_npz}")
                    np.save(f"{root_dir}/{test_npz}/x_test", data)
                    np.save(f"{root_dir}/{test_npz}/y_test", targets)

        # setup the data part
        self.data = torch.from_numpy(np.array(data, dtype=np.float32))
        self.data = self.data.permute(0, 3, 1, 2)/255.0

        # setup targets to be a 2 channel tensor
        # each channel represents a 0 / 1 class 
        # of the pixel color
        self.original_targets = torch.from_numpy(np.array(targets, dtype=np.float32))/255.0
        tmp = torch.zeros(self.original_targets.shape[0], 2, self.original_targets.shape[1], self.original_targets.shape[2])
        tmp[:, 0, :, :][self.original_targets[:, :, :]==0] = 1
        tmp[:, 1, :, :][self.original_targets[:, :, :]==1] = 1
        self.targets = tmp

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.data[index], self.targets[index], self.original_targets[index]