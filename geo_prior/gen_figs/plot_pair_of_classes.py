"""
Plots location predictions for a pair of classes.
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
import geo_prior.grid_predictor as grid


model_path = '../models/model_inat_2018_full_final.pth.tar'
data_dir = get_paths('inat_2018_data_dir')
op_dir = 'images/pair_ims/'
if not os.path.isdir(op_dir):
    os.makedirs(op_dir)

class_of_interest_1 = 2610  # Bufo bufo - European toad
class_of_interest_2 = 2612  # Bufo spinosus - Spiny toad
time_of_year = 0.5  # i.e. half way through the year

# load class names
with open(data_dir + 'categories2018.json') as da:
    cls_data = json.load(da)
class_names = [cc['name'] for cc in cls_data]
class_ids = [cc['id'] for cc in cls_data]
class_dict = dict(zip(class_ids, class_names))


print('Class of interest 1: ' + str(class_of_interest_1) + ' ' + class_dict[class_of_interest_1])
print('Class of interest 2: ' + str(class_of_interest_2) + ' ' + class_dict[class_of_interest_2])


# load model
net_params = torch.load(model_path, map_location='cpu')
params = net_params['params']
model = models.FCNet(num_inputs=params['num_feats'], num_classes=params['num_classes'],
                     num_filts=params['num_filts'], num_users=params['num_users']).to(params['device'])

model.load_state_dict(net_params['state_dict'])
model.eval()


# load ocean mask
mask = np.load(get_paths('mask_dir') + 'ocean_mask.npy')


# grid predictor - for making dense predictions for each lon/lat location
gp = grid.GridPredictor(mask, params, mask_only_pred=True)


# make predictions for both classes
if not params['use_date_feats']:
    print('Trained model not using date features')

grid_pred_1 = gp.dense_prediction(model, class_of_interest_1, time_step=time_of_year)
grid_pred_2 = gp.dense_prediction(model, class_of_interest_2, time_step=time_of_year)


plt.close('all')
plt.figure(0)
plt.imshow(1-grid_pred_1, cmap='afmhot', vmin=0, vmax=1)
plt.title(class_dict[class_of_interest_1] + ' ' + str(class_of_interest_1))

plt.figure(1)
plt.imshow(1-grid_pred_2, cmap='afmhot', vmin=0, vmax=1)
plt.title(class_dict[class_of_interest_2] + ' ' + str(class_of_interest_2))


print('Saving images to: ' + op_dir)
op_file_name_1 = op_dir + str(class_of_interest_1).zfill(4) + '.png'
op_file_name_2 = op_dir + str(class_of_interest_2).zfill(4) + '.png'
plt.imsave(op_file_name_1, 1-grid_pred_1, cmap='afmhot', vmin=0, vmax=1)
plt.imsave(op_file_name_2, 1-grid_pred_2, cmap='afmhot', vmin=0, vmax=1)

plt.show()
