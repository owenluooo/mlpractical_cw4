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




# Net2+bilstm
class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, 3)    
        self.dropout1 = nn.Dropout(0.25)       
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, 3) 
        self.dropout1 = nn.Dropout(0.25)       
        self.flattern = nn.Flatten(start_dim=1, end_dim=2)
        self.fc = nn.Linear(832, num_of_labels)
        self.dropout2 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # torch.Size([1, 1, 128, 22])
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)  # torch.Size([1, 64, 13, 1])

        x = self.flattern(x)  # torch.Size([1, 832, 1])
#         x = x.view(-1, 832) 
#         x = self.fc(x)
#         x = self.dropout2(x)
#         x = self.softmax(x)
        return x