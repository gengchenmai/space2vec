"""
Plots all the observation locations from the training set.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('../')
from geo_prior.paths import get_paths
import geo_prior.datasets as dt


op_dir = 'images/all_locs/'
if not os.path.isdir(op_dir):
    os.makedirs(op_dir)

# load ocean mask
mask = np.load(get_paths('mask_dir') + 'ocean_mask.npy')
mask_lines = (np.gradient(mask)[0]**2 + np.gradient(mask)[1]**2)
mask_lines[mask_lines > 0.0] = 1.0

params = {}
params['dataset'] = 'inat_2017'  # inat_2018, inat_2017, birdsnap, nabirds, yfcc
params['meta_type'] = ''
params['map_range'] = (-180, 180, -90, 90)

# load dataset
op = dt.load_dataset(params, 'val', True, True)
train_locs = op['train_locs']
train_classes = op['train_classes']
train_users = op['train_users']
train_dates = op['train_dates']
classes = op['classes']
#class_of_interest = op['class_of_interest']


# plot GT locations
plt.close('all')
im_width  = mask_lines.shape[1]
im_height = mask_lines.shape[0]
plt.figure(num=0, figsize=[im_width/250, im_height/250], dpi=100)
plt.imshow(1-mask_lines, extent=params['map_range'], cmap='gray')

#inds = np.where(train_classes==class_of_interest)[0]
#print('{} instances of: '.format(len(inds)) + classes[class_of_interest])
inds = np.arange(train_locs.shape[0])

# the color of the dot indicates the date
colors = np.sin(np.pi*train_dates[inds])
plt.scatter(train_locs[inds, 0], train_locs[inds, 1], c=colors, s=2, cmap='magma', vmin=0, vmax=1)

plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().set_frame_on(False)
plt.tight_layout()

op_file_name = op_dir + params['dataset'] + '_all_locs.png'
plt.savefig(op_file_name, dpi=500, bbox_inches='tight', pad_inches=0)

