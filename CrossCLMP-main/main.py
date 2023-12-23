
import torch
import yaml
from models.mlp_head import MLPHead
from trainer import Trainer
import numpy as np
import cv2
from tifffile import imread
from dataset import MyData, MyData1
from models.online_network import MS_Model, PAN_Model
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


import os
os.environ["OMP_NUM_THREADS"] = "1"

# print(torch.__version__)
torch.manual_seed(0)


def CrossCLMP_main():

    Train_Rate = 0.05
    Unlabel_Rate = 0.05

    pan_np = imread('/path/to/pan.tif')
    print('The shape of the original pan:', np.shape(pan_np))

    ms4_np = imread('/path/to/ms4.tif')
    print('The shape of the original MS:', np.shape(ms4_np))

    label_np = np.load("/path/to/label.npy")
    print('The shape of the label：', np.shape(label_np))

    Ms4_patch_size = 16

    Interpolation = cv2.BORDER_REFLECT_101

    top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                    int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
    ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('The shape of the MS picture after padding', np.shape(ms4_np))

    Pan_patch_size = Ms4_patch_size * 4
    top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                    int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
    pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('The shape of the PAN picture after padding', np.shape(pan_np))

    # label_np=label_np.astype(np.uint8)
    label_np = label_np - 1


    label_element, element_count = np.unique(label_np, return_counts=True)
    print('类标：', label_element)
    print('各类样本数：', element_count)
    Categories_Number = len(label_element) - 1
    print('标注的类别数：', Categories_Number)
    label_row, label_column = np.shape(label_np)

    def to_tensor(image):
        max_i = np.max(image)
        min_i = np.min(image)
        image = (image - min_i) / (max_i - min_i)
        return image

    ground_xy = np.array([[]] * Categories_Number).tolist()
    ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column,
                                                                        2)
    unlabeled_xy = []

    count = 0
    for row in range(label_row):  # 行
        for column in range(label_column):
            ground_xy_allData[count] = [row, column]
            count = count + 1
            if label_np[row][column] != 255:
                ground_xy[int(label_np[row][column])].append([row, column])
            else:
                unlabeled_xy.append([row, column])

    length_unlabel = len(unlabeled_xy)
    using_length = length_unlabel * Unlabel_Rate
    unlabeled_xy = unlabeled_xy[0:int(using_length)]
    print("无标签数据使用了{}组数据".format(len(unlabeled_xy)))

    for categories in range(Categories_Number):
        ground_xy[categories] = np.array(ground_xy[categories])
        shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
        np.random.shuffle(shuffle_array)
        ground_xy[categories] = ground_xy[categories][shuffle_array]

    shuffle_array = np.arange(0, label_row * label_column, 1)
    np.random.shuffle(shuffle_array)
    ground_xy_allData = ground_xy_allData[shuffle_array]
    unlabeled_xy = np.array(unlabeled_xy)

    ground_xy_train = []
    ground_xy_test = []
    label_train = []
    label_test = []

    for categories in range(Categories_Number):
        categories_number = len(ground_xy[categories])
        # print('aaa', categories_number)
        for i in range(categories_number):
            if i < int(categories_number * Train_Rate):
                ground_xy_train.append(ground_xy[categories][i])
            else:
                ground_xy_test.append(ground_xy[categories][i])
        label_train = label_train + [categories for x in range(int(categories_number * Train_Rate))]
        label_test = label_test + [categories for x in range(categories_number - int(categories_number * Train_Rate))]

    label_train = np.array(label_train)
    label_test = np.array(label_test)
    ground_xy_train = np.array(ground_xy_train)
    ground_xy_test = np.array(ground_xy_test)


    shuffle_array = np.arange(0, len(label_test), 1)
    np.random.shuffle(shuffle_array)
    label_test = label_test[shuffle_array]
    ground_xy_test = ground_xy_test[shuffle_array]

    shuffle_array = np.arange(0, len(label_train), 1)
    np.random.shuffle(shuffle_array)
    label_train = label_train[shuffle_array]
    ground_xy_train = ground_xy_train[shuffle_array]


    label_train = torch.from_numpy(label_train).type(torch.LongTensor)
    label_test = torch.from_numpy(label_test).type(torch.LongTensor)
    ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
    ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
    ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)
    unlabeled_xy = torch.from_numpy(unlabeled_xy).type(torch.LongTensor)

    print('train：', len(label_train))

    ms4 = to_tensor(ms4_np)
    pan = to_tensor(pan_np)
    pan = np.expand_dims(pan, axis=0)
    ms4 = np.array(ms4).transpose((2, 0, 1))

    ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
    pan = torch.from_numpy(pan).type(torch.FloatTensor)

    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    unlabeled_data = MyData1(ms4, pan, unlabeled_xy, Ms4_patch_size)

    online_network1 = MS_Model().to(device)
    online_network2 = PAN_Model().to(device)


    predictor1 = MLPHead(in_channels=online_network1.g1[3].out_features,
                        **config['network']['projection_head']).to(device)
    predictor2 = MLPHead(in_channels=online_network2.g2[3].out_features,
                        **config['network']['projection_head']).to(device)
    
    predictor1_orth = MLPHead(in_channels=online_network1.g1[3].out_features,
                        **config['network']['projection_head']).to(device)
    predictor2_orth = MLPHead(in_channels=online_network2.g2[3].out_features,
                        **config['network']['projection_head']).to(device)

    target_network1 = MS_Model().to(device)
    target_network2 = PAN_Model().to(device)

    optimizer = torch.optim.Adam(list(online_network1.parameters()) + list(predictor1.parameters()) + list(predictor1_orth.parameters()) +
                                list(online_network2.parameters()) + list(predictor2.parameters()) + list(predictor2_orth.parameters()),
                                **config['optimizer']['params'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)

    trainer = Trainer(online_network1=online_network1,
                          online_network2=online_network2,
                          target_network1=target_network1,
                          target_network2=target_network2,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          predictor1=predictor1,
                          predictor2=predictor2,
                          predictor1_orth=predictor1_orth,
                          predictor2_orth=predictor2_orth,
                          device=device,
                          **config['trainer'])

    trainer.train(unlabeled_data)


if __name__ == '__main__':
    # MS-PAN Cross-source contrastive learning
    CrossCLMP_main()


