
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
class EEGNet_Module(nn.Module):
    #lets hard code for tensor[100,1,256,33]
    def __init__(self,channels=33,fs=256):
        super(EEGNet_Module, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, channels), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4)
        self.pooling3 = nn.MaxPool2d((2, 4))

    def forward(self, x:Tensor):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        #print('layer1',x.shape)
        x = x.permute(0, 3, 1, 2)
        #print('permute',x.shape)
        
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        #print('layer2',x.shape)
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        #print('layer3',x.shape)
        return x
