# coding:utf8
import os
from PIL import Image, ImageFilter
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import csv
import random
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


class FreqMaskGenerator1:
    def __init__(self,
                 input_size=224,
                 mask_radius1=64,
                 mask_radius2=999,
                 sample_ratio=0.5):
        self.input_size = input_size
        self.mask_radius1 = mask_radius1
        self.mask_radius2 = mask_radius2
        self.sample_ratio = sample_ratio
        self.allpass_mask = np.ones((self.input_size, self.input_size), dtype=int)
        self.mask = np.ones((self.input_size, self.input_size), dtype=int)
        for y in range(self.input_size):
            for x in range(self.input_size):
                if ((x - self.input_size // 2) ** 2 + (y - self.input_size // 2) ** 2) >= self.mask_radius1 ** 2 \
                        and ((x - self.input_size // 2) ** 2 + (y - self.input_size // 2) ** 2) < self.mask_radius2 ** 2:
                    self.mask[y, x] = 0

    def __call__(self, label):
        val = random.choice([0, 0, 0, 1])
        if val == 0:
            return self.mask
        else:
            return self.allpass_mask



class PatchCNN(data.Dataset):
    def __init__(self, csv_root_list, transforms=None, train=False, val=False, test=False, data_root=None):
        self.train = train
        self.val   = val
        self.test  = test
        if train:
            self.split = 'train'
        elif val:
            self.split = 'val'
        elif test:
            self.split = 'test'
        else:
            print ('False dataset Type input!')
        print ('dataset Type: ', self.split)
        
        self.csv_root_list = csv_root_list
        self.ReadCsv()
        
        
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        # self.transforms = T.Compose([
        #     T.ToTensor(),
        #     normalize
        # ])
        self.transforms = T.Compose([
            T.ToTensor()
        ])
        
        self.freq_mask_generator = FreqMaskGenerator1(
            input_size=224,
            mask_radius1=64,
            mask_radius2=999,
            sample_ratio=0.5
        )

            
    def __getitem__(self, index):
        img_path_origin = self.origin_datapath[index]
        img_path_target = self.target_datapath[index]
        label = self.label[index]
        
        img_origin = Image.open(img_path_origin)
        img_t_origin = self.transforms(img_origin)
        img_target = Image.open(img_path_target)
        img_t_target = self.transforms(img_target)
        
        mask = self.freq_mask_generator(label)
        # print ('type(mask): ', type(mask))

        
        return img_t_origin, img_t_target, mask, label
          
    def __len__(self):
        return len(self.origin_datapath)
    

    
    def ReadCsv(self):
        self.origin_datapath = []
        self.target_datapath = []
        self.label   = []
        for csv_path in self.csv_root_list:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for i, data in enumerate(reader):
                    self.origin_datapath.append(data[0])
                    self.target_datapath.append(data[1])
                    img_label = 0 if data[2] == 'legal' else 1
                    self.label.append(img_label)

        if not len(self.origin_datapath) == len(self.label):
            print ('false length corresponding!')
        else:
            self.data_len = len(self.origin_datapath)
            print ('data length: ', self.data_len)
            
            
class TestDataset(data.Dataset):
    def __init__(self, csv_root_list, transforms=None, train=False, val=False, test=False, data_root=None):
        self.train = train
        self.val   = val
        self.test  = test
        if train:
            self.split = 'train'
        elif val:
            self.split = 'val'
        elif test:
            self.split = 'test'
        else:
            print ('False dataset Type input!')
        print ('dataset Type: ', self.split)
        
        self.csv_root_list = csv_root_list
        self.ReadCsv()
        
        
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        self.transforms = T.Compose([
            T.ToTensor()
        ])

            
    def __getitem__(self, index):
        img_path = self.datapath[index]
        img_name = self.dataname[index]
        label = self.label[index]
        
        img = Image.open(img_path)
        img_t = self.transforms(img)
        
        return img_t, img_path, img_name, label
          
    def __len__(self):
        return len(self.datapath)
    

    
    def ReadCsv(self):
        self.datapath = []
        self.dataname = []
        self.label   = []
        for csv_path in self.csv_root_list:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for i, data in enumerate(reader):
                    self.datapath.append(data[0])
                    self.dataname.append(data[2])
                    img_label = 0 if data[1] == 'legal' else 1
                    self.label.append(img_label)

        if not len(self.datapath) == len(self.label):
            print ('false length corresponding!')
        else:
            self.data_len = len(self.datapath)
            print ('data length: ', self.data_len)
            
          

    
if __name__ == '__main__':
    csv_root = ['./D1+D4.csv']
    dataset = PatchCNN(csv_root, train=True)
    dataset = PatchCNN(csv_root, val=True)
    dataset = PatchCNN(csv_root, test=True)
