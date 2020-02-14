"""
Plots metadata stats for both iNat datasets.

Only counts datapoints for where there is location info.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os

import sys
sys.path.append('../')
from geo_prior import models
from geo_prior.paths import get_paths
import geo_prior.datasets as dt


inat_year = '2018'  # '2017' or '2018'
model_path = '../models/model_inat_' + inat_year + '_full_final.pth.tar'
data_dir = get_paths('inat_' + inat_year + '_data_dir')
min_num_exs = 100
seed = 2001
dpi = 150.0
op_dir = 'images/metadata_stats/'
if not os.path.isdir(op_dir):
    os.makedirs(op_dir)


# load class info
with open(data_dir + 'categories' + inat_year + '.json') as da:
    cls_data = json.load(da)
class_names = [cc['name'] for cc in cls_data]
class_ids = [cc['id'] for cc in cls_data]
supercat_names = [cc['supercategory'] for cc in cls_data]
supercat_un, supercat_ids = np.unique(supercat_names, return_inverse=True)

# load user info
train_locs, train_classes, train_users, train_dates = dt.load_inat_data(data_dir,
        'train' + inat_year + '_locations.json', 'train' + inat_year + '.json', True)
assert (train_users==-1).sum() == 0
un_users, cnt_users = np.unique(train_users, return_counts=True)
print('\t {} unique photographers'.format(len(un_users)))

# number of unique classes observed by each individual
classes_per_user = []
for uu in un_users:
    inds = np.where(train_users==uu)[0]
    num_un_classes = np.unique(train_classes[inds]).shape[0]
    classes_per_user.append(num_un_classes)


label_font_size = 12
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.close('all')
plt.figure(1, figsize=(8,4))

plt.subplot(1, 2, 1)
plt.plot(np.sort(cnt_users), lw=2, c=colors[0])
plt.ylabel('number of observations', fontsize=label_font_size)
plt.xlabel('sorted photographers', fontsize=label_font_size)
plt.grid(True)
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.plot(np.sort(classes_per_user), lw=2, c=colors[1])
plt.ylabel('number of unique categories', fontsize=label_font_size)
plt.xlabel('sorted photographers', fontsize=label_font_size)
plt.yscale('log')
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle('iNat' + inat_year + '', fontsize=label_font_size+4)


op_file = op_dir + 'iNat'+inat_year+'_metadata.pdf'
print('Saving fig to: ' + op_file)
plt.savefig(op_file)

plt.show()