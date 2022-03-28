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

from Model import *
from Net2 import *
from MultiheadAttention import *
from TransformerEncoderLayer import *
from ConvBlocks import *
from vit import *

with h5py.File('all-samples-train-whole.h5', 'r') as f:
    for key in f.keys():
        print(f[key].name)
        print(f[key].shape)

f.close()

# Open dataset
# keys = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'tru', 'voi', 'sax', 'vio']
keys = ['banjo', 'bass clarinet', 'bassoon', 'cello', 'clarinet', 'contrabassoon', 'cor anglais', 'double bass', 'flute', 'french horn', 'guitar', 'mandolin', 'oboe', 'saxophone', 'trombone', 'trumpet', 'tuba', 'viola', 'violin']
dataset = h5py.File('all-samples-train-whole.h5', 'r')
# dataset = h5py.File('IRMAS_mini.h5', 'r')

num_of_labels = len(keys)  # 2
num_of_tracks = sum([dataset[x].shape[0] for x in keys])  # 256

# Prepare data for training and testing
features = np.zeros((num_of_tracks, 128, 22), dtype=np.float32)   # (3, 128, 22)
labels = np.zeros((num_of_tracks, len(keys)), dtype=np.float32)   # (3, 2)

i = 0
for ki, k in enumerate(keys):
    features[i:i + len(dataset[k])] = np.nan_to_num(dataset[k])  # 使用0代替数组x中的nan元素
    labels[i:i + len(dataset[k]), ki] = 1
    i += len(dataset[k])

print(features.shape) # (2, 128, 22)
print(labels.shape)  # (2, 2)

# Split trainset to train and evaluation
# X_train,X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.4, random_state=0)
X_train, X_eval, Y_train, Y_eval = train_test_split(features, labels, test_size=0.1, random_state=1337)
print(X_train.shape)
print(X_eval.shape)

# Prepare Pytorch dataloader
X_train_torch = torch.from_numpy(X_train) # create a tensor from X_train
X_eval_torch = torch.from_numpy(X_eval)
Y_train_torch = torch.from_numpy(Y_train)
Y_eval_torch = torch.from_numpy(Y_eval)

print(type(X_train_torch))

trainset = torch.utils.data.TensorDataset(X_train_torch, Y_train_torch)  # 包装数据和目标张量的数据集
evalset = torch.utils.data.TensorDataset(X_eval_torch, Y_eval_torch)

print(type(trainset))

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2) # 取数据集
eval_dataloader = torch.utils.data.DataLoader(evalset, batch_size=1, shuffle=True, num_workers=2)    
        
# Initialisation
# net = Net2()
# net = ConvBlocks()
# net = Model(19).cuda()
net = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
).cuda()
# criterion = nn.CrossEntropyLoss()

# optimiser = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
# for epoch in range(25): 
#     running_loss = 0.0
#     for i, (inputs, labels) in enumerate(train_dataloader, start=0):

#         inputs = inputs.unsqueeze(1) # add one dimension # torch.Size([1, 1, 128, 22])
#         inputs = inputs[:, :, :, 0:16]

#         optimiser.zero_grad()
#         # forward + backward + optimize
#         outputs = net(inputs.cuda()) # torch.Size([1, 5])
#         _, labels = torch.max(labels, 1)
#         # print(outputs)
#         # print(labels)
#         loss = criterion(outputs, labels.cuda())
#         loss.backward()
#         optimiser.step()

#         # print(f'iteration: ', i)

#         # print statistics
#         running_loss += loss.item()
#         if i == len(train_dataloader) - 1:
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 100))
#             running_loss = 0.0
# print('Finished Training')
# PATH = './wave_model_han16net.pth'
# PATH = './wave_model_net.pth'
PATH = './wave_model_vit.pth'
# torch.save(net.state_dict(), PATH)

net.load_state_dict(torch.load(PATH))  # 将预训练的参数权重加载到新的模型之中

correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(eval_dataloader, start=0):
        inputs = inputs.unsqueeze(1) # add one dimension
        inputs = inputs[:, :, :, 0:16]
        inputs = inputs.to('cuda')
        outputs = net(inputs)
        newlabels = labels > 0
        indices =  newlabels.nonzero()     
        print(outputs.data)
        pred_sum = outputs.data.sum()
        _, pred_max = torch.max(outputs.data, 1)
        pred_max = outputs.data[0][pred_max]
        print(pred_sum)
        print(pred_max)
        pred_index = outputs.data / pred_max
        print(1)
        print(pred_index)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        print(indices)
        correct += (predicted == indices[0][1]).sum().item()
        print('label:')
        print(labels)
        print('newlabel:')
        print(indices[0][1])
        print('predicted:')
        print(predicted)        
        print('Total: %d' % total)
        print('Correct: %d' % correct)
        

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
