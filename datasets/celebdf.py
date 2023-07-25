import os
import random
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class FaceForensics(Dataset):
    def __init__(self,
                opt,
                split = 'train',
                downsample = 1,
                transforms=None,
                method=None,
                balance=False
                ):
        """
        Args:
            data_root (str): Root for images.
            transform (type, optional): Data transform. Defaults to None.
            downsample (int, optional): Defaults to 10, downsample to each video.
            balance (bool, optional): Whether balance the fake and real image. Defaults to True.
            method (str): a special forgery method. If none, data for four method are used.
        """

        self.data_root = opt['datasets']['root'] 
        self.split = split
        self.transforms = transforms
        self.downsample = opt['datasets']['train']['downsample'] if self.split == 'train' else 1

        self.balabce = opt['datasets'][split]['balance']  
        self.method = opt['datasets'][split]['method']   

        self.items = self._load_items()
    
        print(f'celebdf(test): Total number of data: {len(self.items)}')

    def _load_items(self):
        items = []
        with open(self.data_root, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label, video_path = line[:-1].split(',')
                imgs = os.listdir(video_path)
                for i in range(0, len(imgs), self.downsample):
                    image_path = os.path.join(video_path, imgs[i])
                    items.append({
                        'image_path': image_path,
                        'label': int(label)
                    })
        return items

    def __getitem__(self, index):
        item = self.items[index]
        label = item['label']
        
        image = cv2.cvtColor(cv2.imread(item['image_path']), cv2.COLOR_BGR2RGB)
        image = self.transforms[self.split](image=image)['image']

        return image, label, image

    def __len__(self):
        return len(self.items)

    