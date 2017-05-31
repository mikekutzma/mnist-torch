import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt

class ChunkSampler(sampler.Sampler):
    def __init__(self,num_samples,start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start,self.start+self.num_samples))

    def __len__(self):
        return self.num_samples

#total training samples = 60000
NUM_TRAIN = 58000
NUM_VAL = 2000

batch_size = 64

mnist_train = dset.MNIST('./data',train=True,download=False, transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=batch_size, sampler=ChunkSampler(NUM_TRAIN,0))

mnist_val = dset.MNIST('./data',train=True,download=True, transform=T.ToTensor())
loader_val = DataLoader(mnist_val, batch_size=batch_size, sampler=ChunkSampler(NUM_VAL,NUM_TRAIN))

mnist_train = dset.MNIST('./data',train=False,download=True, transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=batch_size)

dtype = torch.FloatTensor

print_every = 100

def reset(m):
    if hasattr(m,'reset_parameters'):
        m.reset_parameters()

class Flatten(nn.Module):
    def forward(self,x):
        N,C,H,W = x.size()
        return x.view(N,-1)


