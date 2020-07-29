# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""

import torchvision.transforms as transforms
import os

model_dir = 'Model_and_Log'
cuda_num = 1
class_num = 6

source_dataset_path = '1_multistage_malaria_classification/train'
target_dataset_path = '1_multistage_malaria_classification/test'
source_batch_size = 15
target_batch_size = 15
class1_num = 100
class2_num = 100
class3_num = 100
class4_num = 100
class5_num = 100
class6_num = 100
train_class1_num = 44
train_class2_num = 107
train_class3_num = 5000
train_class4_num = 253
train_class5_num = 79
train_class6_num = 1373

model_name = 'ResNet18'
features_lr = 2e-5
default_balance = 1
iter_num = 30000
test_frequency = 1000

bal_file = 'balance.txt'
mmd_file = 'mmd_loss.txt'
con_file = 'con_loss.txt'
model_file = 'model.pt'
# GCN
GCN_source_batch_size = 15
GCN_lr = 0.0000001
GCN_iter_num = 200
GCN_test_frequency = 1
# all
kmeans_model = 'model.pt'
kclus_model = 'model.pt'
features_dim_num = 1024
time_file = 'time.txt'
acc_file = 'acc.txt'
loss_file = 'loss_val.txt'


normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
normTransform = transforms.Normalize(normMean, normStd)
trainTransform = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomCrop(128, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normTransform
])
testTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])




