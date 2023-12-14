import os
from shutil import copyfile

import numpy as np

import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random
import math


windowSize = 11

def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))


def applyPCA(X, numComponents):
    """
    apply PCA to the image to reduce dimensionality
  """
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def addZeroPadding(X, margin=2):
    """
    add zero padding to the image
  """
    newX = np.zeros(
        (X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX


def createImgCubeAll(X, pos, windowSize=11):
    """
    create Cube from pos list
    return imagecube gt
  """
    margin = (windowSize - 1) // 2
    zeroPaddingX = addZeroPadding(X, margin=margin)
    return np.array([
        zeroPaddingX[i:i + windowSize, j:j + windowSize, :] for i, j in pos
    ]), np.array(pos)


def createImgCube(X, pos, windowSize=11):
    """
    create Cube from pos list
    return imagecube gt
  """
    margin = (windowSize - 1) // 2
    zeroPaddingX = addZeroPadding(X, margin=margin)
    return np.array([
        zeroPaddingX[i:i + windowSize, j:j + windowSize, :]
        for i, j in pos[:, :2]
    ]), np.array(pos[:, 2])


def createPos(shape: tuple, pos: tuple, num: int):
    """
    creatre pos list after the given pos
  """
    if (pos[0] + 1) * (pos[1] + 1) + num > shape[0] * shape[1]:
        num = shape[0] * shape[1] - ((pos[0]) * shape[1] + pos[1])
    return [(pos[0] + (pos[1] + i) // shape[1], (pos[1] + i) % shape[1])
            for i in range(num)]


def createPosWithoutZero(gt):
    """
    creatre pos list without zero labels
  """
    # mask = gt > 0
    mask = np.ones((gt.shape[0], gt.shape[1]))

    for i, row in enumerate(mask):
        for j, row_element in enumerate(row):
            if row_element:
                for a in range(-12, 13):
                    for b in range(-12, 13):
                        if (i + a >= 0 and i + a < mask.shape[0] and j + b >= 0
                                and j + b < mask.shape[1]):
                            mask[i + a][j + b] = False
                mask[i][j] = True
    return [(i, j) for i, row in enumerate(mask)
            for j, row_element in enumerate(row) if row_element]


def createPosWithoutZeroVal(gt):
    """
    creatre pos list without zero labels
  """
    # mask = gt > 0
    mask = np.ones((gt.shape[0], gt.shape[1]))

    for i, row in enumerate(mask):
        for j, row_element in enumerate(row):
            if (i < 7 or j < 7):
                continue
            if row_element:
                for a in range(-12, 13):
                    for b in range(-12, 13):
                        if (i + a >= 0 and i + a < mask.shape[0] and j + b >= 0
                                and j + b < mask.shape[1]):
                            mask[i + a][j + b] = False
                mask[i][j] = True
    return [(i, j) for i, row in enumerate(mask)
            for j, row_element in enumerate(row) if row_element]


def splitTrainTestSet(X, gt, testRatio, randomState=456):
    """
    random split data set
  """
    X_train, X_test, gt_train, gt_test = train_test_split(
        X, gt, test_size=testRatio, random_state=randomState, stratify=gt)
    return X_train, X_test, gt_train, gt_test


def createImgPatch(lidar, pos, windowSize=11):
    """
    return lidar Img patches
  """

    margin = (windowSize - 1) // 2
    zeroPaddingLidar = np.zeros((lidar.shape[0] + 2 * margin,
                                 lidar.shape[1] + 2 * margin, lidar.shape[2]))

    zeroPaddingLidar[margin:lidar.shape[0] + margin,
                     margin:lidar.shape[1] + margin] = lidar

    return np.array([
        zeroPaddingLidar[i:i + windowSize, j:j + windowSize, :]
        for i, j in pos[:, :2]
    ])


def getAllPos(lidar_im):

    mask = np.ones((lidar_im.shape[0], lidar_im.shape[1]))
    for i, row in enumerate(mask):
        for j, row_element in enumerate(row):
            if row_element:
                for a in range(-12, 13):
                    for b in range(-12, 13):
                        if (i + a >= 0 and i + a < mask.shape[0] and j + b >= 0
                                and j + b < mask.shape[1]):
                            mask[i + a][j + b] = False
                mask[i][j] = True
    return np.array([(i, j) for i, row in enumerate(mask)
                     for j, row_element in enumerate(row) if row_element])


def getValPos(lidar_im):
    mask = np.ones((lidar_im.shape[0], lidar_im.shape[1]))
    for i, row in enumerate(mask):
        for j, row_element in enumerate(row):
            if (i < 7 or j < 7):
                continue
            if row_element:
                for a in range(-12, 13):
                    for b in range(-12, 13):
                        if (i + a >= 0 and i + a < mask.shape[0] and j + b >= 0
                                and j + b < mask.shape[1]):
                            mask[i + a][j + b] = False
                mask[i][j] = True
    return np.array([(i, j) for i, row in enumerate(mask)
                     for j, row_element in enumerate(row) if row_element])


def getTrainPos(train_im):
    return np.array([[i, j, train_im[i, j]] for i in range(train_im.shape[0])
                     for j in range(train_im.shape[1]) if train_im[i, j]])


def getTestPos(test_im):
    return np.array([[i, j, test_im[i, j]] for i in range(test_im.shape[0])
                     for j in range(test_im.shape[1]) if test_im[i, j]])


def set_random_seed(mySeed=0):
    torch.manual_seed(mySeed)
    torch.cuda.manual_seed(mySeed)
    torch.cuda.manual_seed_all(mySeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(mySeed)
    random.seed(mySeed)

# 这一步产生了两个
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform, ratio=1):
        self.base_transform = base_transform
        self.ratio = ratio

    def __call__(self, x):
        q = self.base_transform(x)
        if (random.random() < self.ratio):
            k = self.base_transform(x)
        else:
            k = q
        return [q, k]

# def TwoCropsTransform(base_transform, ratio=1):
#     q = base_transform(x)
#     if (random.random() < ratio):
#         k = base_transform(x)
#     else:
#         k = q
#     return [q, k]


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



