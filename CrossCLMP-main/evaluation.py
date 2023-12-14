from models.online_network import MS_Model, PAN_Model
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from tifffile import imread
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
os.environ["OMP_NUM_THREADS"] = "1"
device = torch.device('cuda:0')

# your pre-training result
pretrained_folder = 'None'


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        MS_BackBone = MS_Model()
        PAN_BackBone = PAN_Model()
        if pretrained_folder:
            try:
                checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')
                load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                         map_location=torch.device(torch.device(device)))
                MS_BackBone.load_state_dict(load_params['online_network_1_state_dict'])
                PAN_BackBone.load_state_dict(load_params['online_network_2_state_dict'])
                print("the model of MS&PAN has been loaded")
            except FileNotFoundError:
                print("Pre-trained weights not found. Training from scratch.")

        add_block = []
        add_block += [nn.Linear(512, 128)]
        add_block += [nn.BatchNorm1d(128)]
        add_block += [nn.ReLU(True)]
        add_block += [nn.Linear(128, num_classes)]
        add_block = nn.Sequential(*add_block)

        self.MS_BackBone = MS_BackBone
        self.PAN_BackBone = PAN_BackBone
        self.add_block = add_block

    def forward(self, ms, pan):

        feature_ms, orth_feature_ms = self.MS_BackBone(ms)
        feature_pan, orth_feature_pan = self.PAN_BackBone(pan)
        feature = torch.cat([feature_ms, feature_pan, orth_feature_ms, orth_feature_pan], 1)
        feature = feature.view(feature.size()[0], -1)
        result = self.add_block(feature)

        return result


Train_Rate = 0.01
BATCH_SIZE = 128
EPOCH = 35

pan_np = imread('/path/to/pan.tif')
print('原始pan图的形状;', np.shape(pan_np))

ms4_np = imread('/path/to/ms4.tif')
print('原始ms4图的形状：', np.shape(ms4_np))

label_np = np.load("/path/to/label.npy")
print('label数组形状：', np.shape(label_np))

Ms4_patch_size = 16
Interpolation = cv2.BORDER_REFLECT_101
top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),  # 7  8
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))  # 7  8
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)  # 长宽各扩15
print('补零后的ms4图的形状：', np.shape(ms4_np))

Pan_patch_size = Ms4_patch_size * 4
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),  # 28 32
                                                int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))  # 28 32
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)  # 长宽各扩60
print('补零后的pan图的形状：', np.shape(pan_np))

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
ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2)
count = 0
for row in range(label_row):
    for column in range(label_column):
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if label_np[row][column] != 255:
            ground_xy[int(label_np[row][column])].append([row, column])


for categories in range(Categories_Number):
    ground_xy[categories] = np.array(ground_xy[categories])
    shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
    np.random.shuffle(shuffle_array)
    ground_xy[categories] = ground_xy[categories][shuffle_array]

shuffle_array = np.arange(0, label_row * label_column, 1)
np.random.shuffle(shuffle_array)
ground_xy_allData = ground_xy_allData[shuffle_array]

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

print('训练样本数：', len(label_train))
print('测试样本数：', len(label_test))


ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
pan = np.expand_dims(pan, axis=0)
ms4 = np.array(ms4).transpose((2, 0, 1))

ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)

class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)
        y_pan = int(4 * y_ms)
        # 切出中心点的周围部分区域
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]  # dim：chw

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]

        return image_ms, image_pan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)


class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)

train_data = MyData(ms4, pan, label_train, ground_xy_train, Ms4_patch_size)
test_data = MyData(ms4, pan, label_test, ground_xy_test, Ms4_patch_size)
all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_data_loader = DataLoader(dataset=all_data, batch_size=BATCH_SIZE, shuffle=False,num_workers=0)

model = Net(num_classes=Categories_Number).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)

def train_model(model, train_loader, optimizer, epoch):
    model.train()
    correct = 0.0
    iters = epoch * len(label_train) // BATCH_SIZE
    print('训练样本数：', len(label_train))
    print('测试样本数：', len(label_test))

    for step, (ms, pan, label, _) in enumerate(train_loader):
        ms, pan, label = ms.to(device), pan.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(ms, pan)
        pred_train = output.max(1, keepdim=True)[1]
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()
        loss = F.cross_entropy(output, label.long())
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + step / iters)
        if step % 1 == 0:
            print("Train Epoch: {} \t Loss : {:.6f} \t step: {} ".format(epoch, loss.item(), step))
    print("Train Accuracy: {:.6f}".format(correct * 100.0 / len(train_loader.dataset)))


def test_model(model, test_loader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data, data1, target, _ in test_loader:
            data, data1, target = data.to(device), data1.to(device), target.to(device)
            output = model(data, data1)
            test_loss += F.cross_entropy(output, target.long()).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).long()).sum().item()
        test_loss = test_loss / len(test_loader.dataset)
        print("test-average loss: {:.4f}, Accuracy:{:.3f} \n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)
        ))


def test_second(model, test_loader, mode='2'):
    model.eval()
    test_loss = 0
    correct = 0.0
    test_matrix = np.zeros([Categories_Number, Categories_Number])
    loop = tqdm(test_loader, leave=True)
    with torch.no_grad():
        for batch_idx, (data1, data2, target, _) in enumerate(loop):
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            output = model(data1, data2)
            # test_loss += nn.CrossEntropyLoss(output, target.long())
            test_loss += F.cross_entropy(output, target.long()).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).long()).sum().item()
            for i in range(len(target)):
                test_matrix[int(pred[i].item())][int(target[i].item())] += 1
            loop.set_postfix(mode='test')
        loop.close()
        test_loss = test_loss / len(test_loader.dataset)
        print("test-average loss: {:.4f}, Accuracy:{:.3f} \n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)
        ))
    return test_matrix

def test_third(model, test_loader):
    model.eval()
    correct = 0.0
    test_bar=tqdm(test_loader)
    test_metric=np.zeros([Categories_Number,Categories_Number])
    with torch.no_grad():
        for data, data1, target, _  in test_bar:
            data, data1, target= data.to(device), data1.to(device), target.to(device)
            output= model(data,data1)
            # test_loss = F.cross_entropy(output[0], target.long()).item()
            pred = output.max(1, keepdim=True)[1]
            for i in range(len(target)):
                test_metric[int(pred[i].item())][int(target[i].item())]+=1
            correct += pred.eq(target.view_as(pred).long()).sum().item()
        print("test Accuracy:{:.3f} \n".format( 100.0 * correct / len(test_loader.dataset)))
    b=np.sum(test_metric,axis=0)
    accuracy=[]
    c=0
    for i in range(0,Categories_Number):
        a=test_metric[i][i]/b[i]
        accuracy.append(a)
        c+=test_metric[i][i]
        print('category {0:d}: {1:f}'.format(i,a))
    average_accuracy = np.mean(accuracy)
    overall_accuracy = c/np.sum(b,axis=0)
    kappa_coefficient = kappa(test_metric)
    print('AA: {0:f}'.format(average_accuracy))
    print('OA: {0:f}'.format(overall_accuracy))
    print('KAPPA: {0:f}'.format(kappa_coefficient))
    return 100.0 * correct / len(test_loader.dataset)


def aa_oa(matrix):
    accuracy = []
    b = np.sum(matrix, axis=0)
    c = 0
    on_display = []
    for i in range(1, matrix.shape[0]):
        a = matrix[i][i]/b[i]
        c += matrix[i][i]
        accuracy.append(a)
        on_display.append([b[i], matrix[i][i], a])
        print("Category:{}. Overall:{}. Correct:{}. Accuracy:{:.6f}".format(i, b[i], matrix[i][i], a))
    aa = np.mean(accuracy)
    oa = c / np.sum(b, axis=0)
    k = kappa(matrix)
    print("OA:{:.6f} AA:{:.6f} Kappa:{:.6f}".format(oa, aa, k))
    return [aa, oa, k, on_display]

def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)

import time
start=time.time()
for epoch in range(1, EPOCH+1):
    train_model(model,  train_loader, optimizer, epoch)
test_matrix = test_third(model, test_loader)
print(test_matrix)

torch.save(model, './transfer_model.pkl')
cnn = torch.load('./transfer_model.pkl')
cnn.to(device)


# 上色
class_count = np.zeros(Categories_Number)
out_clour = np.zeros((label_row, label_column, 3))
gt_clour = np.zeros((label_row, label_column, 3))
def clour_model(cnn, all_data_loader):
    train_bar = tqdm(all_data_loader)
    for step, (ms4, pan, gt_xy) in enumerate(train_bar):
        ms4 = ms4.to(device)
        pan = pan.to(device)
        with torch.no_grad():
            output = cnn(ms4, pan)
        pred_y = torch.max(output, 1)[1].to(device).data.squeeze()
        pred_y_numpy = pred_y.cpu().numpy()
        #print("pred_y###",pred_y_numpy)
        gt_xy = gt_xy.numpy()
        for k in range(len(gt_xy)):
            if pred_y_numpy[k] == 0:
                class_count[0] = class_count[0] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 255, 0]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 255, 0]
            elif pred_y_numpy[k] == 1:
                class_count[1] = class_count[1] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 0, 0]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 0, 0]
            elif pred_y_numpy[k] == 2:
                class_count[2] = class_count[2] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [33, 145, 237]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [33, 145, 237]
            elif pred_y_numpy[k] == 3:
                class_count[3] = class_count[3] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [201, 252, 189]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [201, 252, 189]
            elif pred_y_numpy[k] == 4:
                class_count[4] = class_count[4] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 0, 230]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 0, 230]
            elif pred_y_numpy[k] == 5:
                class_count[5] = class_count[5] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 0]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 0]
            elif pred_y_numpy[k] == 6:
                class_count[6] = class_count[6] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [240, 32, 160]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [240, 32, 160]
            elif pred_y_numpy[k] == 7:
                class_count[7] = class_count[7] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [221, 160, 221]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [221, 160, 221]
            elif pred_y_numpy[k] == 8:
                class_count[7] = class_count[8] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [140, 230, 240]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [140, 230, 240]
            elif pred_y_numpy[k] == 9:
                class_count[7] = class_count[9] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 255]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 255]
            if label_np[gt_xy[k][0]][gt_xy[k][1]] == 255:
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 0, 0]
    print(class_count)
    cv2.imwrite("/home/gpu/Experiment/gpt/exp/clour_images/beijing.png", out_clour)
    cv2.imwrite("/home/gpu/Experiment/gpt/exp/clour_images/beijing_gt.png", gt_clour)

# 上色
clour_model(model,  all_data_loader)


end=time.time()
print("程序process_1的运行时间为：{}".format(end-start))


print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))