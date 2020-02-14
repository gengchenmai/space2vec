# iNatularist image loader

import torch.utils.data as data

from PIL import Image
import os
import json
import numpy as np

def default_loader(path):
    return Image.open(path).convert('RGB')

class IGNAT_Loader(data.Dataset):

    def __init__(self, root, ann_file, transform=None, target_transform=None,
                 loader=default_loader):

        # assumes classes and im_ids are in same order

        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        imgs = [aa['file_name'] for aa in ann_data['images']]
        im_ids = [aa['id'] for aa in ann_data['images']]

        if 'annotations' in ann_data.keys():
            # if we have class labels
            classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            # otherwise dont have class info so set to 0
            classes = [0]*len(im_ids)

        idx_to_class = {cc['id']: cc['name'] for cc in ann_data['categories']}

        print('\t' + str(len(imgs)) + ' images')
        print('\t' + str(len(idx_to_class)) + ' classes')

        self.ids = im_ids
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.idx_to_class = idx_to_class
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        target = self.classes[index]
        im_id = self.ids[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, im_id

    def __len__(self):
        return len(self.imgs)
