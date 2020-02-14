import torch.utils.data as data
from PIL import Image
import os
import json
from torchvision import transforms
import random
import numpy as np
import pandas as pd


def default_loader(path):
    return Image.open(path).convert('RGB')


class YFCC(data.Dataset):
    def __init__(self, root, meta_data, split_name, im_size_resize, im_size_crop, is_train=True):

        # load annotations
        print('Loading annotations from: ' + os.path.basename(meta_data))
        da = pd.read_csv(meta_data)

        # set up the filenames and annotations
        self.imgs = da[da['split'] == split_name]['path'].values
        self.classes = da[da['split'] == split_name]['class'].values
        self.lon = da[da['split'] == split_name]['lon'].values
        self.lat = da[da['split'] == split_name]['lat'].values

        # print out some stats
        print split_name
        print '\t' + str(len(self.imgs)) + ' images'
        print '\t' + str(len(set(self.classes))) + ' classes'

        self.root = root
        self.is_train = is_train
        self.loader = default_loader
        self.im_size_resize = im_size_resize
        self.im_size_crop = im_size_crop

        # augmentation params
        self.mu_data = [0.485, 0.456, 0.406]
        self.std_data = [0.229, 0.224, 0.225]
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.25

        # augmentations
        self.resize = transforms.Resize(self.im_size_resize)
        self.resize_rand = transforms.RandomResizedCrop(self.im_size_crop)
        self.center_crop = transforms.CenterCrop(self.im_size_crop)
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        im_id = self.imgs[index]
        img = self.loader(path)
        class_id = self.classes[index]

        if self.is_train:
            img = self.resize_rand(img)
            img = self.flip_aug(img)
            #img = self.color_aug(img)
        else:
            img = self.resize(img)
            img = self.center_crop(img)

        img = self.tensor_aug(img)
        img = self.norm_aug(img)

        return img, im_id, class_id

    def __len__(self):
        return len(self.imgs)
