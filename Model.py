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

from ConvBlocks import *
from MultiheadAttention import *
from TransformerEncoderLayer import *


class Model(nn.Module):
    def __init__(self,num_classes=9):
        super(Model, self).__init__()
        self.num_classes = num_classes
        
        self.conv = ConvBlocks()
        self.blstm = nn.LSTM(1024, hidden_size=int(320/2),bidirectional=True, batch_first=True)
        self.mha = TransformerEncoderLayer(embed_dim=320, num_heads=10,temp=0.2)
        self.fc1 = nn.Linear(320, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.fc3 = nn.Linear(1024, 320)
        self.fc4 = nn.Linear(512, 320)
    
    def forward(self, inputs):
        cnn_out = self.conv(inputs.cuda())
        cnn_out = cnn_out.permute(0,2,1)   
        cnn_out = cnn_out

        # bilstm layer
        rnn_out,_ = self.blstm(cnn_out)
        rnn_out = rnn_out.permute(1,0,2)
        rnn_out = rnn_out
        # print(rnn_out.shape)

        
     # Transformer layer
        rnn_out = self.fc3(cnn_out)   #同时加bilstm和Transformer，则删掉这一行
        mha_out = self.mha(rnn_out)
        mha_out = mha_out.permute(1,0,2)
        
        # print(mha_out.shape)

        
        pooled = torch.mean(rnn_out, dim=1)
        fc1_out = self.fc1(pooled)
        out = self.fc2(fc1_out)
        out = F.sigmoid(out)
#         print(out.shape) # torch.Size([1, 11])
        
        return out
        