from utils import applyPCA, getAllPos, getTrainPos, getTestPos, createImgCubeAll, createImgCube, createImgPatch

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

import numpy as np
import torch
from skimage import io
import random

windowSize = 11


class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

        self.random_flip = transforms.RandomHorizontalFlip()

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)   
        y_pan = int(4 * y_ms)
        
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]  # dim：chw

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]  

        target = self.train_labels[index] 

        image_ms_view1 = self.random_flip(image_ms)
        image_ms_view2 = self.random_flip(image_ms)

        image_pan_view1 = self.random_flip(image_pan)
        image_pan_view2 = self.random_flip(image_pan)

        return image_ms_view1, image_ms_view2, image_pan_view1, image_pan_view2, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)

class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4
        self.random_flip = transforms.RandomHorizontalFlip()

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms) 
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        image_ms_view1 = self.random_flip(image_ms)
        image_ms_view2 = self.random_flip(image_ms)

        image_pan_view1 = self.random_flip(image_pan)
        image_pan_view2 = self.random_flip(image_pan)

        return image_ms_view1, image_ms_view2, image_pan_view1, image_pan_view2, 1, locate_xy

    def __len__(self):
        return len(self.gt_xy)

