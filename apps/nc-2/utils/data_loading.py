import logging
import numpy as np
import torch
from PIL import Image
from functools import partial
from multiprocessing import Pool
import os
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2

class TwoFolders(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_files = sorted(os.listdir(images_dir))
        self.col_files = sorted(os.listdir(masks_dir))
        self.transform = transform
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        img_filename = self.img_files[index]
        mask_filename = self.col_files[index]
        
        img_path = os.path.join(self.images_dir, img_filename)
        mask_path = os.path.join(self.masks_dir, mask_filename)
        
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0) / 255
        
        if self.transform:
            image = self.transform(image)
        
        return image, mask