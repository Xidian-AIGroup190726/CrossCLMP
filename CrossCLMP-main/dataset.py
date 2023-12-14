from utils import applyPCA, getAllPos, getTrainPos, getTestPos, createImgCubeAll, createImgCube, createImgPatch

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

import numpy as np
import torch
from skimage import io
import random

windowSize = 11


# 测试集对象
class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

        # 图像增广
        self.random_flip = transforms.RandomHorizontalFlip()

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]  # 取出当前标签的坐标
        x_pan = int(4 * x_ms)      # 计算不可以在切片过程中进行（？）
        y_pan = int(4 * y_ms)
        # 切出中心点的周围部分区域
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]  # dim：chw

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]  # 坐标信息

        target = self.train_labels[index]  # 当前像素点的类别信息

        # 图像增广生成view1/2
        # ms view1/2：
        image_ms_view1 = self.random_flip(image_ms)
        image_ms_view2 = self.random_flip(image_ms)
        # pan view1/2
        image_pan_view1 = self.random_flip(image_pan)
        image_pan_view2 = self.random_flip(image_pan)

        return image_ms_view1, image_ms_view2, image_pan_view1, image_pan_view2, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)


# 训练集对象
class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4
        # 图像增广
        self.random_flip = transforms.RandomHorizontalFlip()

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        # 图像增广生成view1/2
        # ms view1/2：
        image_ms_view1 = self.random_flip(image_ms)
        image_ms_view2 = self.random_flip(image_ms)
        # pan view1/2
        image_pan_view1 = self.random_flip(image_pan)
        image_pan_view2 = self.random_flip(image_pan)

        return image_ms_view1, image_ms_view2, image_pan_view1, image_pan_view2, 1, locate_xy

        # return image_ms, image_pan, index, locate_xy

    def __len__(self):
        return len(self.gt_xy)

# 预训练数据集
class HSI_LiDAR_DATA(Dataset):
    def __init__(self, HSI, LiDAR, xy, cut_size, transform_crop,
                 transform_lidar, transform_hsi, transform_noise):
        self.train_data1 = HSI
        self.train_data2 = LiDAR
        self.gt_xy = xy
        self.cut_hsi_size = cut_size
        self.cut_lidar_size = cut_size

        # 图像增广
        self.transform_lidar = transform_lidar
        self.transform_hsi = transform_hsi
        self.transform_crop = transform_crop
        self.transform_noise = transform_noise

        # hsi降维
        self.hsiall_pca = applyPCA(self.train_data1, 30)

    def __getitem__(self, index):
        x_hsi, y_hsi = self.gt_xy[index]
        x_lidar = int(x_hsi)
        y_lidar = int(y_hsi)
        # 切出中心点的周围部分区域
        hsi_cube = self.hsiall_pca[x_hsi:x_hsi + self.cut_hsi_size,
                   y_hsi:y_hsi + self.cut_hsi_size, :]

        lidar_patch = self.train_data2[x_lidar:x_lidar + self.cut_lidar_size,
                    y_lidar:y_lidar + self.cut_lidar_size, :].numpy()

        locate_xy = self.gt_xy[index]

        hsi_cube = torch.from_numpy(hsi_cube.transpose((2, 0, 1))).float()  # 调整通道 chw
        lidar_patch = torch.from_numpy(lidar_patch.transpose((2, 0, 1))).float()  # 调整通道 chw

        lidarhsi = torch.cat([lidar_patch, hsi_cube], 0)
        # 对总体数据一起做两次不同的 data augmentation
        view1 = self.transform_crop(lidarhsi)
        if(random.random() < 0.5):
            view2 = self.transform_crop(lidarhsi)
        else:
            view2 = view1

        lidarhsi = [view1, view2]

        lidar_ = [
            self.transform_noise(self.transform_lidar(lidarhsi[0][:7])),
            self.transform_noise(self.transform_lidar(lidarhsi[1][:7]))
        ]
        hsi_ = [
            self.transform_noise(self.transform_hsi(lidarhsi[0][7:])),
            self.transform_noise(self.transform_hsi(lidarhsi[1][7:]))
        ]
        lidar_view1 = lidar_[0]
        lidar_view2 = lidar_[1]

        hsi_view1 = hsi_[0]
        hsi_view2 = hsi_[1]

        return hsi_view1, hsi_view2, lidar_view1, lidar_view2, 1, locate_xy

    def __len__(self):
        return len(self.gt_xy)