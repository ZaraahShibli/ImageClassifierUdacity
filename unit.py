import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
from collections import OrderedDict
import argparse


def transform_image(root):

    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         standard_normalization]),
                       'valid': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         standard_normalization]),
                       'test': transforms.Compose([transforms.Resize(size=(224,224)),
                                         transforms.ToTensor(), 
                                         standard_normalization])
                      }


    train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train']) 
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test']) 



    return train_data, valid_data, test_data


def load_data(root):
    
    data_dir = root    
    train_data,valid_data,test_data=transform_image(data_dir)
    
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20
    # percentage of training set to use as validation
    valid_size = 0.2
    
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size, 
                                               num_workers=num_workers,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=batch_size, 
                                               num_workers=num_workers,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=batch_size, 
                                               num_workers=num_workers,
                                               shuffle=False)

    loaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }
    return loaders


train_data,valid_data,test_data=transform_image('./flowers/')
loaders=load_data('./flowers/')

def network_construct(arch , dropout=0.5, hidden_layer1 = 4096,lr = 0.001):
    if arch =='vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088,hidden_layer1)),
                              ('relu1', nn.ReLU()),
                              ('d_out1',nn.Dropout(dropout)),
                              ('fc2', nn.Linear(hidden_layer1, 1024)),
                              ('relu2', nn.ReLU()),
                              ('d_out2',nn.Dropout(dropout)),
                              ('fc3', nn.Linear(1024, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))


    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr )    
    model.cuda()

    return model,criterion,optimizer


def train(n_epochs, loaders, model, criterion,optimizer,  save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    model.to('cuda')
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            #if use_cuda:
            data, target = data.to('cuda'), target.to('cuda')
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model.forward(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss = train_loss +((1/(batch_idx +1 ))*(loss.data - train_loss))
            
            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            data, target = data.to('cuda'), target.to('cuda')
            ## update the average validation loss
            output = model.forward(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
    # return trained model
    return model



def save_checkpoint(model=0,path='checkpoint.pth', hidden_layer1 = 4096,dropout=0.5,lr=0.001,epochs=3):

    model.class_to_idx =  train_data.class_to_idx
    model.cpu
    torch.save({'hidden_layer1':hidden_layer1,
                'dropout':dropout,
                'lr':lr,
                'nb_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
    

def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    lr=checkpoint['lr']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    model,_,_ = network_construct('vgg16', dropout,hidden_layer1,lr)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image_path):


    proc_img = Image.open(image_path)

    prepoceess_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    pymodel_img = prepoceess_img(proc_img)
    return pymodel_img


def predict(image_path, model=0, topk=5):
    
    model.to('cuda')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)