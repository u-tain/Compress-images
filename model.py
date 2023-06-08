import torch
import random
import numpy as np
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def quant(B, image, mode:str='train'):
    if mode == "train":
        matrix = np.fromfunction(lambda i, j, k, g: (g+1)/(g+1)*(1/2**B)*random.normalvariate(-0.5, 0.5), (image.shape[0], 512, 16, 16), dtype=float)
        return image+torch.from_numpy(matrix).to('cuda')
    else:
        matrix = np.fromfunction(lambda i, j, k, g: (g+1)/(g+1)*(2**B), (image.shape[0], 512, 16, 16), dtype=float)
        return image*torch.from_numpy(matrix)+0.5
    

class Encoder(nn.Module):
    def __init__(self,B=2):
        super(Encoder, self).__init__()
        self.B = B
        self.enc = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.enc = nn.Sequential(*list(self.enc.children())[:-2])
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        x = self.enc(x)
        x = self.act(x)
        if self.training:
            x = quant(self.B, x)
        else: 
            x = quant(self.B, x, mode='eval')
        x = x.flatten(1)
        return x


class Decoder(nn.Module):
    def __init__(self, B=2):
        super(Decoder, self).__init__()
        self.B=B
        self.upconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2,stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True))
        
        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2,stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True))
        
        self.upconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2,stride=2)
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        
        self.upconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2,stride=2)
        self.conv5 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        
        self.upconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2,stride=2,padding=0)
        self.conv6 = nn.Sequential(nn.Conv2d(16, 3, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(3),)
        
        
    def forward(self, x):
        x = x.reshape(x.shape[0],512,16,16)
        # процесс обратный квантованию
        if not self.training:
            matrix = np.fromfunction(lambda i, j, k, g: (g+1)/(g+1)*(2**self.B), (x.shape[0], 512, 16, 16), dtype=float)
            x = (x-0.5)/torch.from_numpy(matrix)

        x = self.upconv1(x.float())
        x = self.conv2(x)+x
        
        x = self.upconv2(x)
        x = self.conv3(x)+x
        
        x = self.upconv3(x)
        x = self.conv4(x)+x
        
        x = self.upconv4(x)
        x = self.conv5(x)+x
        
        x = self.upconv5(x)
        x = self.conv6(x)
        return x
    