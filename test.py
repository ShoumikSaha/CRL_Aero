import argparse
import os
import sys
import logging
import torch
import time
from model import Encoder, model_dict
from my_resnet import ResNet1D
from my_mlp import *
from dataset import *
from utils import *
import pickle
import csv
from random import randint
import pandas as pd

from data.data_editor import *

ckpt_path = './save/NACA_models/RnC_MLP_CD_NACA_MLP_ep_100_lr_0.5_d_0.1_wd_0.0001_mmt_0.9_bsz_256_aug_crop_temp_2_label_l1_feature_l2_trial_8/ckpt_epoch_100.pth'
regressor_path = './save/NACA_models/RnC_MLP_CD_NACA_MLP_ep_100_lr_0.5_d_0.1_wd_0.0001_mmt_0.9_bsz_256_aug_crop_temp_2_label_l1_feature_l2_trial_8/Regressor_NACA_ep_100_lr_0.05_d_0.1_wd_0_mmt_0.9_bsz_256_trial_0_best.pth'

def set_model(ckpt_path, dataset='NACA'):
    
    model = MLP_NACA(input_size=404, n_classes=512, verbose=False)
    criterion = torch.nn.L1Loss()

    #dim_in = model_dict[opt.model][1]
    dim_in = model.n_classes
    dim_out = get_label_dim(dataset)
    regressor = torch.nn.Linear(dim_in, dim_out)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.device_count() > 1:
        model.encoder = torch.nn.DataParallel(model.encoder)
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    # model = model.cuda()
    # regressor = regressor.cuda()
    # criterion = criterion.cuda()
    torch.backends.cudnn.benchmark = True

    model.load_state_dict(state_dict)
    print(model)

    ckpt_state = torch.load(regressor_path)
    regressor.load_state_dict(ckpt_state['state_dict'])
    print(regressor)

    return model, regressor, criterion

def get_output(model, regressor, input):
    model.eval()
    regressor.eval()
    with torch.no_grad():
        feature = model(input)
        #print(feature)
        output = regressor(feature)
    return output




model, regressor, criterion = set_model(ckpt_path)
#print(model)


df = pd.read_csv('/data/NACA Dataset/all_data_filtered_normalized.csv')
new_csv_file = '/data/NACA Dataset/all_data_filtered_normalized_cd.csv'
#print(df)
mae = 0
frac = 0.0


for index, row in df.iterrows():
    label = np.asarray([row['CD']]).astype(np.float32)

    aoa = np.asarray([row['AOA1']]).astype(np.float32)
    re = np.asarray([row['RE']]).astype(np.float32)
    input = np.concatenate((aoa, re))


    for i in range(0, 201):
        x_feat_name = 'x' + str(i)
        y_feat_name = 'y' + str(i)
        x = np.asarray([row[x_feat_name]]).astype(np.float32)
        y = np.asarray([row[y_feat_name]]).astype(np.float32)
        input = np.concatenate((input, x, y))

    input = np.resize(input, (1, 404, 1))
    output = get_output(model, regressor, torch.tensor(input))
    print(output, label)
    #edit_csv_value(new_csv_file, index, 'output', output.item())

    mae += abs(output - label)
    frac += abs((output - label)/label)

    

print(mae/df.shape[0])
print(frac/df.shape[0])

