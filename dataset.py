import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
import torch


# class AgeDB(data.Dataset):
#     def __init__(self, data_folder, transform=None, split='train'):
#         df = pd.read_csv(f'./data/agedb.csv')
#         self.df = df[df['split'] == split]
#         self.split = split
#         self.data_folder = data_folder
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, index):
#         row = self.df.iloc[index]
#         label = np.asarray([row['age']]).astype(np.float32)
#         img = Image.open(os.path.join(self.data_folder, row['path'])).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)

#         return img, label
    
class Airfoil(data.Dataset):
    def __init__(self, data_folder, transform=None, split='train'):
        df = pd.read_csv(f'./data/normalized_airfoil_data.csv')
        self.df = df[df['split'] == split]
        self.split = split
        self.data_folder = data_folder
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]

        label = np.asarray([row['cd']]).astype(np.float32)

        camber = np.asarray([row['max camber']]).astype(np.float32)
        camber_pos = np.asarray([row['max camber position']]).astype(np.float32)
        thickness = np.asarray([row['max thickness']]).astype(np.float32)
        thickness_pos = np.asarray([row['max thickness position']]).astype(np.float32)
        re = np.asarray([row['re']]).astype(np.float32)
        aoa = np.asarray([row['aoa']]).astype(np.float32)
        input = np.concatenate((camber, camber_pos, thickness, thickness_pos, re, aoa))
        #input = input*255.0
        #print(input.shape)
        input = np.resize(input, (6, 1))
        #print(input.shape)
        #image = Image.fromarray(np.uint8(input)).convert('RGB')
        if self.transform is not None:
            input = self.transform(input)
            # input[0].reshape([6, 1])
            # input[1].reshape([6, 1])

        return input, label


class Ellipsoid(data.Dataset):
    def __init__(self, data_folder, transform=None, split='train', label_name='cd'):
        df = pd.read_csv(f'./data/Ellipsoid Dataset/normalized_ellipsoid_dataset_w_split.csv')
        self.df = df[df['split'] == split]
        self.split = split
        self.data_folder = data_folder
        self.transform = transform
        self.label_name = label_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        if self.label_name == 'cd':
            label = np.asarray([row['cd']]).astype(np.float32)
        elif self.label_name == 'cl':
            label = np.asarray([row['cl']]).astype(np.float32)

        a = np.asarray([row['a']]).astype(np.float32)
        b = np.asarray([row['b']]).astype(np.float32)
        c = np.asarray([row['c']]).astype(np.float32)
        aoa = np.asarray([row['aoa']]).astype(np.float32)

        input = np.concatenate((a, b, c, aoa))
        input = np.resize(input, (4, 1))
        if self.transform is not None:
            input = self.transform(input)

        return input, label
    
class NACA(data.Dataset):
    def __init__(self, data_folder, transform=None, split='train', label_name='cd'):
        df = pd.read_csv(f'./data/NACA Dataset/all_data_filtered_normalized.csv')
        self.df = df[df['split'] == split]
        self.split = split
        self.data_folder = data_folder
        self.transform = transform
        self.label_name = label_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        if self.label_name == 'cd':
            label = np.asarray([row['CD']]).astype(np.float32)
        elif self.label_name == 'cl':
            label = np.asarray([row['CL']]).astype(np.float32)

        aoa = np.asarray([row['AOA1']]).astype(np.float32)
        re = np.asarray([row['RE']]).astype(np.float32)
        input = np.concatenate((aoa, re))


        for i in range(0, 201):
            x_feat_name = 'x' + str(i)
            y_feat_name = 'y' + str(i)
            x = np.asarray([row[x_feat_name]]).astype(np.float32)
            y = np.asarray([row[y_feat_name]]).astype(np.float32)
            input = np.concatenate((input, x, y))

        input = np.resize(input, (404, 1))
        if self.transform is not None:
            input = self.transform(input)

        return input, label