
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import torch
# import torchvision
# import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import os

# ConvBlocks + bilstm + Transformer
class ConvBlocks(nn.Module):
    def __init__(self,):
        super(ConvBlocks, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,512, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,512, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,512, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            nn.ReLU(),
            

        )
        
    def forward(self,inputs):
        out = self.conv(inputs)
        out = out.flatten(start_dim=1, end_dim=2)
        return out