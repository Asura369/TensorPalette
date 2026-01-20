from collections import namedtuple
import os
import torch
import torch.nn as nn
from torchvision import models

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, vgg_path="models/vgg16.pth"):
        super(Vgg16, self).__init__()
        
        # Load VGG16 from local file if exists, otherwise download
        vgg = models.vgg16(weights=None)
        if os.path.exists(vgg_path):
            print(f"[VGG16] Loading from local file: {vgg_path}")
            vgg.load_state_dict(torch.load(vgg_path, map_location='cpu'))
        else:
            print(f"[VGG16] Local file not found, downloading pretrained weights...")
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        vgg_pretrained_features = vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        # slice 1: relu1_2
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # slice 2: relu2_2
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # slice 3: relu3_3
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # slice 4: relu4_3
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
