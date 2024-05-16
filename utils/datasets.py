import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from mnist.loader import MNIST
from torch.utils.data import Dataset


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


class MorphomnistDataset(Dataset):

    def __init__(self, root_dir, transform=None, test=False, gz=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        mndata = MNIST(root_dir, gz=gz)

        if not test:
            images, labels = mndata.load_training()
            self.features = np.genfromtxt(root_dir / 'train-morpho-tas.csv', delimiter=',')[1:, :]
        else:
            images, labels = mndata.load_testing()
            self.features = np.genfromtxt(root_dir / 't10k-morpho-tas.csv', delimiter=',')[1:, :]

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = np.array(self.images[idx]).reshape(28, 28)
        # sample = sample[np.newaxis,:]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.features[idx], self.labels[idx]


class MorphomnistDecodeDataset(Dataset):
    def __init__(self, encodings, features, labels, device='cpu'):
        self.device = device
        self.encodings = torch.from_numpy(encodings).to(device)
        self.features = torch.from_numpy(features).to(device)
        self.labels = torch.from_numpy(labels).to(device)

        self.len = encodings.shape[0]

    # print('data loaded on {}'.format(self.x.device))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.encodings[index], self.features[index], self.labels[index]

    def get_metadata(self):
        return {
            'n': self.len
        }


class UKBBT1Dataset(Dataset):

    def __init__(self, csv_file_path, img_dir, transform=None):
        self.csv_file_path = csv_file_path
        self.img_dir = img_dir

        self.df = pd.read_csv(csv_file_path, low_memory=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(int(self.df.iloc[idx]['eid'])) + '.tiff'
        img = tiff.imread(self.img_dir / img_name)

        if self.transform:
            img = self.transform(img)

        return self.df.iloc[idx]['Sex'], self.df.iloc[idx]['Age'], self.df.iloc[idx]['BMI'], img

class EmmaDataset(Dataset):
    def __init__(self, csv_file_path, img_dir, transform=None):
        self.csv_file_path = csv_file_path
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file_path, low_memory=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename'].replace("nii.gz", "tiff")
        img = tiff.imread(self.img_dir / img_name)

        if self.transform:
            img = self.transform(img)

        return self.df.iloc[idx]['class_label'], self.df.iloc[idx]['bias_label'], img


class UKBBT13DDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.files = sorted(glob.glob(os.path.join(root_dir, '*.nii.gz')))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        sample = nib.load(img_name).get_fdata()

        if self.transform:
            sample = self.transform(sample)
        return sample
    
class UKBBT1DatasetOld(Dataset):

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

        if self.transform:
            sample = self.transform(sample)
        return sample

class MitacsDataset3D(Dataset):
    def __init__(self, img_dir, df, device='cpu', label_col='Group_bin', file_name_col = 'Subject', transform=None):
        self.device = device
        self.df = df
        self.img_dir = img_dir
        self.len = len(self.df)
        self.label_col = label_col
        self.file_name_col = file_name_col
        self.transform = transform

    # print('data loaded on {}'.format(self.x.device))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.df.loc[index, self.file_name_col] + '.nii.gz')
        #image = tiff.imread(img_path)
        image = nib.load(img_path).get_fdata()
        label = self.df.loc[index, self.label_col].astype('f4')
        if self.transform:
            image = self.transform(image)
        return image, label