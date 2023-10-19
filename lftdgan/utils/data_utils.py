"""
 > Modules for processing training/validation data  
 > Maintainer: https://github.com/xahidbuffon
"""
import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

 
class GetTrainingPairs(Dataset):
    """ Common data pipeline to organize and generate
         training pairs for various datasets   
    """
    def __init__(self, root, dataset_name, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.filesA, self.filesB = self.get_file_paths(root, dataset_name)          # 出现问题
        self.len = min(len(self.filesA), len(self.filesB))

    def __getitem__(self, index):
        img_A = Image.open(self.filesA[index % self.len])
        img_B = Image.open(self.filesB[index % self.len])
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        return {"A": img_A, "B": img_B}

    def __len__(self):
        return self.len

    def get_file_paths(self, root, dataset_name):
        if dataset_name=='train':
            filesA, filesB = [], []
            # sub_dirs = ['underwater_imagenet', 'underwater_dark', 'underwater_scenes']
            # for sd in sub_dirs:
            filesA += sorted(glob.glob(os.path.join(root,"train", 'LQ') + "/*.*"))
            filesB += sorted(glob.glob(os.path.join(root,"train", 'GT') + "/*.*"))
        if dataset_name == 'val':
            filesA, filesB = [], []
            # sub_dirs = ['underwater_imagenet', 'underwater_dark', 'underwater_scenes']
            # for sd in sub_dirs:
            filesA += sorted(glob.glob(os.path.join(root, "val", 'LQ') + "/*.*"))
            filesB += sorted(glob.glob(os.path.join(root, "val", 'GT') + "/*.*"))

        return filesA, filesB 



class GetValImage(Dataset):
    """ Common data pipeline to organize and generate
         vaditaion samples for various datasets   
    """
    def __init__(self, root, dataset_name, transforms_=None, sub_dir='validation'):
        self.transform = transforms.Compose(transforms_)
        self.files = self.get_file_paths(root, dataset_name)
        self.len = len(self.files)

    def __getitem__(self, index):
        img_val = Image.open(self.files[index % self.len])
        img_val = self.transform(img_val)
        return {"val": img_val}

    def __len__(self):
        return self.len

    def get_file_paths(self, root, dataset_name):
        if dataset_name=='train':
            files = []
            files += sorted(glob.glob(os.path.join(root, "val" , 'LQ') + "/*.*"))
        if dataset_name == 'val':
            filesA, filesB = [], []
            # sub_dirs = ['underwater_imagenet', 'underwater_dark', 'underwater_scenes']
            # for sd in sub_dirs:
            filesA += sorted(glob.glob(os.path.join(root, "val", 'LQ') + "/*.*"))
            filesB += sorted(glob.glob(os.path.join(root, "val", 'GT') + "/*.*"))

        return files


class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([0.6]), torch.tensor([0.6]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy


import torch.nn as nn
import torch_dct
class DCToperator(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, x):
        x = torch_dct.dct_2d(x) + self.epsilon
        return x