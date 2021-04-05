import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import scipy.io as scio
from torch.utils.data import Dataset, DataLoader
import transforms3d.quaternions as txq
import transforms3d.euler as txe
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch.optim as optim
import os
from fft_dataset_abs import *
from network import *

#import os

if __name__ == '__main__':  
    train_id = 'no_dropout'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    continue_training = 0
    trainset = FFT_DATASET_2D(1)
    validationset = FFT_DATASET_2D(2)
    batch_size = 32
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = True, num_workers=4)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle = False, num_workers=4)


    net = CNN_LSTM()
    model_name = 'test_'+train_id+'.pth'


    if continue_training:
        checkpoint = torch.load(model_name)
        net.load_state_dict(checkpoint['model_state_dict'])
    for name,parameters in net.named_parameters():
        print(name,':',parameters.size())


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    net = net.to(device)
    def weighted_mse_loss(inputs, target, weight=torch.tensor([10,10,1]).cuda()):
        return torch.sum(weight * (inputs - target) ** 2)

    train_criterion = nn.MSELoss(reduction='mean').to(device)
    val_criterion = nn.MSELoss(reduction='mean').to(device)

    optimizer = optim.Adam(list(net.parameters()), lr=0.00003)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 40, eta_min=0, last_epoch=-1)
    if continue_training:
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = 0
    if continue_training:
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    log_f = open('train_log_'+train_id+'.csv','a')
    previous_loss = 100000000
    for epoch in range(start_epoch, 30):  # loop over the dataset multiple times
        net.train()
        print(epoch)
        running_loss = 0.0
        for i, (x_azimuth, x_elevation, labels) in enumerate(trainloader):
            x_azimuth = x_azimuth.to(device)#, dtype=torch.float)
            x_elevation = x_elevation.to(device)#, dtype=torch.float)
            labels = labels.to(device)#, dtype=torch.float)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(x_azimuth, x_elevation)
            loss = torch.sqrt(train_criterion(outputs, labels)*3)
            #loss = weighted_mse_loss(labels, outputs)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i+1)))
        print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss/(i+1)))
        log_f.write('%.5f,'%(running_loss/(i+1)))
        #validation
        if epoch % 1 == 0:
            net.eval()
            validation_loss = 0
            with torch.no_grad():
                for i, (x_azimuth, x_elevation, labels) in enumerate(validationloader):
                    x_azimuth = x_azimuth.to(device)#, dtype=torch.float)
                    x_elevation = x_elevation.to(device)#, dtype=torch.float)
                    labels = labels.to(device)#, dtype=torch.float)
                    outputs = net(x_azimuth, x_elevation)
                    loss = torch.sqrt(train_criterion(outputs, labels)*3)
                    validation_loss += loss
            print('validation [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, validation_loss / (i+1)))
            if validation_loss < previous_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': previous_loss
                    #'scheduler_state_dict': scheduler.state_dict()
                }, model_name)
                previous_loss = validation_loss
                print('saved\n')
                log_f.write('saved\n')
            log_f.write('%.5f\n'%(validation_loss / (i+1)))
