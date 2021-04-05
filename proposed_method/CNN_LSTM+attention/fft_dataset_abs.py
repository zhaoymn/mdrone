import numpy as np
import pandas as pd
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import scipy.io as scio
from torch.utils.data import Dataset, DataLoader
import transforms3d.quaternions as txq
import transforms3d.euler as txe
from scipy.io import loadmat




def qlog(q):
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q

def process_poses(poses_in, mean_t=np.zeros(3), std_t=np.ones(3), align_R=np.eye(3), align_t=np.zeros(3), align_s=1):
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((4, 4))[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()
    return poses_out



def xyz2polar(point):
    epsilon=1e-100
    x = point[0]
    y = point[1]
    z = point[2]
    r = np.linalg.norm(point)
    phi = np.arcsin(z/(r+epsilon))
    theta = np.arctan(y/(x+epsilon))
    pointPolar=np.array([r, theta, phi])
    return pointPolar

class FFT_DATASET_2D(Dataset):
    def __init__(self, mode):
        self.base_dir = #path of preprocessed data
        self.gt_dir = #ground truth dir
        self.validation_seq_file = #path to validation_sequences.csv
        self.train_seq_file = #path to train_sequences.csv
        self.test_seq_file =  #path to test_sequences.csv
        self.validation_sequences = np.genfromtxt(self.validation_seq_file)
        self.train_sequences = np.genfromtxt(self.train_seq_file)
        self.test_sequences = np.genfromtxt(self.test_seq_file)
        #self.train_sequences = training_number
        self.mode = mode
        
        self.gt = np.zeros((100* 600, 6))
        for i in range(0, 100):
            tmp = np.load(self.gt_dir + str(int(i+1)) + '.npy')
            self.gt[600*i:600*(i+1), :] = process_poses(np.reshape(tmp[0:600], (600, 16)))
        if mode == 1:#train
            self.total_sequence = 80
        else: #test
            self.total_sequence = 10
        self.maxv = -1
        self.minv = 10000
    def __len__(self):
        return self.total_sequence*600

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #for i in idx:
        i = idx
        if self.mode == 1:
            sequence_id = int(np.floor(i/600))
            sequence = int(self.train_sequences[sequence_id])
        else:
            sequence_id = int(np.floor(i/600)) 
            sequence = int(self.validation_sequences[sequence_id])
        #load data
        #print(sequence)
        dataFile = self.base_dir + str(int(sequence)) + '/' + str(i % 600 + 1) + '.mat'
        tmp_data = loadmat(dataFile)['parsed_data']
        
        x_azimuth = tmp_data[:,0:2,:,:]
        output_azimuth = np.abs(x_azimuth)
        
        x_elevation = tmp_data[:,2:6,:,:]
        output_elevation = np.abs(x_elevation)

        #print(tmp_data.shape)
        label = np.reshape(self.gt[int(sequence-1) * 600 + (i%600), 0:3], 3)
        #label = xyz2polar(label)
#                print(label)
        #data1 = np.array(data1, dtype=np.float32)
        #data2 = np.array(data2, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        output_azimuth = np.array(output_azimuth, dtype=np.float32)
        output_elevation = np.array(output_elevation, dtype=np.float32)
        
        return output_azimuth, output_elevation, label

