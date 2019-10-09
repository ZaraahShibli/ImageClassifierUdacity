# Imports here
import torch
from torchvision import models, transforms, datasets
from torch import optim
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch.nn.functional as F
import unit
import argparse

ap = argparse.ArgumentParser(description='Train.py')


ap.add_argument('data_dir', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)


pa = ap.parse_args()
root = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
device = pa.gpu
epochs = pa.epochs
arch = pa.arch

def main():
    print("here",root)
    loaders = unit.load_data(root)
    
    model,criterion,optimizer = unit.network_construct(arch, dropout,hidden_layer1,lr)
    unit.train(10, loaders, model,criterion, optimizer,'model_vgg16.pt')
    unit.save_checkpoint(model,path,hidden_layer1,dropout,lr)
    print("Done Training!")


if __name__== "__main__":
    main()
        
