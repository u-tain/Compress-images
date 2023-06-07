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
        return image*torch.from_numpy(matrix).to('cuda')+0.5
    

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
    