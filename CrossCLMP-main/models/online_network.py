
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class PAN_Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(PAN_Model, self).__init__()

        self.f2 = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f2.append(module)
        # encoder
        self.f2 = nn.Sequential(*self.f2)
        # projection head Original
        self.g2 = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        # projection head Orthogonal
        self.g2_orth = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, pan):

        # pan
        pan = self.f2(pan)
        feature_pan = torch.flatten(pan, start_dim=1)
        out_pan = self.g2(feature_pan)
        out_pan_orth = self.g2_orth(feature_pan)

        return F.normalize(out_pan, dim=-1), F.normalize(out_pan_orth, dim=-1)



class MS_Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(MS_Model, self).__init__()

        self.f1 = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f1.append(module)
        # encoder
        self.f1 = nn.Sequential(*self.f1)
        # projection head Original
        self.g1 = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        # projection head Orthogonal
        self.g1_orth = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, ms):
        # ms
        ms = self.f1(ms)
        feature_ms = torch.flatten(ms, start_dim=1)
        out_ms = self.g1(feature_ms)
        out_ms_orth = self.g1_orth(feature_ms)

        return F.normalize(out_ms, dim=-1), F.normalize(out_ms_orth, dim=-1)

