import glob
import os

import tifffile as tiff
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, X, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(device)
        self.len = self.x.shape[0]
        self.data_dim = self.x.shape[1]

    # print('data loaded on {}'.format(self.x.device))

    def get_dims(self):
        return self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index]

    def get_metadata(self):
        return {
            'n': self.len,
            'data_dim': self.data_dim,
        }


class UKBBT1Dataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.files = sorted(glob.glob(os.path.join(root_dir, '*.tiff')))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        sample = tiff.imread(img_name)
        plt.imshow(sample)
        plt.show()

        if self.transform:
            sample = self.transform(sample)
        return sample