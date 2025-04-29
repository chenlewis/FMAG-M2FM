from torch.utils import data
import os
from torchvision.transforms import transforms as T
from PIL import Image
import random
# import cv2
# from torch.utils.data import DataLoader
# from sklearn.model_selection import KFold
# import numpy as np
# import torch

class Copy_Detection(data.Dataset):
    def __init__(self, root,  transforms=None, train=True, test=False):
        super(Copy_Detection, self).__init__()
        self.test = test

        #加载路径
        # imgs = [os.path.join(root, img) for img in os.listdir(root)]
        imgs_0 = [os.path.join(root, '0', img) for img in os.listdir(os.path.join(root, '0'))]
        imgs_1 = [os.path.join(root, '1', img) for img in os.listdir(os.path.join(root, '1'))]

        if self.test:
            self.imgs = imgs_0 + imgs_1

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
            )
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    normalize
                ])
            else:
                # 训练集
                normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                self.transforms = T.Compose([
                    T.Resize((224, 224)),
                    T.RandomHorizontalFlip(p=0.5), 
                ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.test:
            label = int(os.path.basename(os.path.dirname(img_path)))  # Extract label from parent folder name
        else:
            label = int(os.path.basename(os.path.dirname(img_path)))  # Extract label from filename

        data = Image.open(img_path)
        #data = Image.open(img_path).convert('')
        data = self.transforms(data)

        return data, label,img_path

    def __len__(self):
        return len(self.imgs)


class Copy_Detection_1(data.Dataset):
    def __init__(self, root, transforms=None, train=True):
        super(Copy_Detection_1, self).__init__()
        self.train = train
        self.root = root

        if self.train:
            self.imgs = imgs_0[:int(0.8 * len(imgs_0))] + imgs_1[:int(0.8 * len(imgs_1))]
        else:
            self.imgs = imgs_0[int(0.8 * len(imgs_0)):] + imgs_1[int(0.8 * len(imgs_1)):]

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            if not self.train:
                self.transforms = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    normalize
                ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]

        if self.train:
            label = int(os.path.basename(os.path.dirname(img_path)))  # Extract label from parent folder name
        else:
            label = int(os.path.basename(os.path.dirname(img_path)))
            
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

































