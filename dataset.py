import os
import os.path
import glob
import sys
import torch
import torch.utils.data as data
import numpy as np
import random
import cv2
import torchvision 
# from util import *
import ipdb 

class DataFolder(data.Dataset):
    def __init__(self, path_img, path_mask, mode):
        self.path_img = path_img
        self.path_mask = path_mask  
        self.mode = mode 
        self.get_data_list()

    def get_data_list(self):
        self.img_fns = []
        for fn in glob.iglob(self.path_img + '*tif'):
            img_id = os.path.basename(fn)
            # img_id = os.path.splitext(img_id)[0]
            self.img_fns.append(img_id)
        print('Number of {} images loaded: {}'.format(self.mode, len(self.img_fns)))

    def __getitem__(self, index):
        img_id = self.img_fns[index]

        img = cv2.imread(self.path_img + img_id, cv2.IMREAD_UNCHANGED).astype(np.float32)
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)

        mask = cv2.imread(self.path_mask + img_id, cv2.IMREAD_UNCHANGED)
        mask = torch.from_numpy(mask)

        return img, mask, img_id 

    def __len__(self):
        return len(self.img_fns)