import numpy as np

import torch
from torch.utils.data.dataset import Dataset


class HyperData(Dataset):
    def __init__(self, dataset):
        self.data1 = dataset

    def __getitem__(self, index):
        img1 = torch.from_numpy(np.asarray(self.data1[index,:,:,:]))
        return img1

    def __len__(self):
        return len(self.data1)
