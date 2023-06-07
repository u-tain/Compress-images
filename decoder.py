import torch
from torch import nn 


class Decoder(nn.Module):
    def __init__(self,in_channels=512):
        super(Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=256, kernel_size=2,stride=2)
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
        x = self.upconv1(x)
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
    