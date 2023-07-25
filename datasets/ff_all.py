import json
import os
import random
import cv2
import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image


class FaceForensics(Dataset):
    def __init__(self,
                 opt,
                 split = 'train',
                 transforms=None,
                 ):

        self.json_file = opt['datasets'][split]['split']
        self.split = split
        self.transforms = transforms
        self.downsample = opt['datasets']['train']['downsample'] if self.split == 'train' else 1

        self.data_path = opt['datasets']['root']  
        self.balabce = opt['datasets'][split]['balance']  
        self.method = opt['datasets'][split]['method']    

        if self.method is None:
            self.method = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

        self.real_items = self._load_items('youtube')
        self.fake_items = self._load_items(self.method)

        pos_len = len(self.real_items)
        neg_len = len(self.fake_items)

        if self.split == 'train':
            neg_len_copy = neg_len
            temp_items = []
            while(neg_len_copy >= pos_len):
                temp_items += self.real_items
                neg_len_copy -= pos_len
            temp_items += np.random.choice(self.real_items, neg_len_copy, replace=False).tolist()
            self.real_items = temp_items
            pos_len = len(self.real_items)
        
        print(f'ff_all_v3({self.method}): Total number of data: {pos_len+neg_len} | pos: {pos_len}, neg: {neg_len}')

        if self.balabce == True:
            np.random.seed(0)
            if pos_len > neg_len:
                self.real_items = np.random.choice(self.real_items, neg_len, replace=False).tolist()
            else:
                self.fake_items = np.random.choice(self.fake_items, pos_len, replace=False).tolist()
            image_len = len(self.real_items)
            print(f'After balance total number of data: {image_len*2} | pos: {image_len}, neg: {image_len}')

        self.items = self.real_items + self.fake_items
        self.items = sorted(self.items, key=lambda x: x['img_path'])


    def _load_frames(self, video_path, label):
        # load json
        with open(self.json_file) as f:
            split = json.load(f)
        frames, video_names = [], []
        for i, s in enumerate(split):
            video_names.append(s[0] + '_' + s[1] + '.mp4')
            video_names.append(s[1] + '_' + s[0] + '.mp4')
        # video_name = '_'.join(s) + '.mp4' if label == 1 else s[0] + '.mp4'
        for video_name in video_names:
            imgs = os.listdir(os.path.join(video_path, video_name))
            for i in range(0, len(imgs), self.downsample):
                img_path = os.path.join(video_path, video_name, imgs[i])
                frames.append({
                    'img_path': img_path,
                    'label': label
                })
        return frames
        
    def _load_items(self, method):
        items = []
        if isinstance(method, list):
            label = int(1)
            for m in method:
                self.data_root = os.path.join(self.data_path, m)
                frames = self._load_frames(self.data_root, label)
                items.extend(frames)

        else:
            if method == "youtube_v1":
                label = int(0) 
                self.data_root = os.path.join(self.data_path, method)
                # self.root = os.path.join(self.data_root, method)
            else:
                label = int(1)
                self.data_root = os.path.join(self.data_path, method)
                    
            frames = self._load_frames(self.data_root, label)
            items.extend(frames)

        return items

    def __getitem__(self, index):
        item = self.items[index]
        label = item['label']

        image, mask, _ = np.split(cv2.imread(item['img_path']), 3, axis=1)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        region = mask
        
        if self.split == 'train':
            aug_image = self.transforms['base'](image=image, masks=[mask, region])
            image = aug_image['image']
            mask, region = aug_image['masks']

        image = self.transforms[self.split](image=image)['image']
        mask = self.transforms['mask'](image=mask)['image']

        return image, label, mask

    def __len__(self):
        return len(self.items)


    
