import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import *

class LinTrans(nn.Module):
    def __init__(self, layers, dims):
        super(LinTrans, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.act = nn.Sigmoid()
        # self.act = nn.LeakyReLU()

    def scale(self, z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
    
        return z_scaled

    def forward(self, x):
        num_layer = len(self.layers)
        out = x
        for i in range(num_layer - 1):
            out = self.act(self.layers[i](out))
        out = self.layers[num_layer - 1](out)
        # out = self.scale(out)
        out = F.normalize(out)
        return out

class OUR(nn.Module):
    def __init__(self, lt_layers, dims):
        super(OUR, self).__init__()
        self.lt1 = LinTrans(lt_layers, dims)
        self.lt2 = LinTrans(lt_layers, dims)
    
    def forward(self, X):

        Z1, Z2 = self.lt1(X), self.lt2(X)
        return Z1, Z2
