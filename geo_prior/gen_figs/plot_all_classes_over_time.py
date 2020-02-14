"""
Plots sum of predictions for all classes over time.
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
op_dir = 'images/all_classes_over_time/'
if not os.path.isdir(op_dir):
    os.makedirs(op_dir)

# load class names
with open(data_dir + 'categories2018.json') as da:
    cls_data = json.load(da)
class_names = [cc['name'] for cc in cls_data]
class_ids = [cc['id'] for cc in cls_data]
class_dict = dict(zip(class_ids, class_names))

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
assert params['use_date_feats'] == True
gp = grid.GridPredictor(mask, params, mask_only_pred=True)

print('Making predictions for each time step for all classes.')
pred_ims = []
max_val = 0.0
for ii, tm in enumerate(np.linspace(0,1,12)):
    grid_pred, max_val_pred = gp.dense_prediction_sum(model, tm, mask_op=False)
    pred_ims.append(grid_pred)
    if max_val_pred > max_val:
        max_val = max_val_pred

print('Saving images to: ' + op_dir)
for ii in range(len(pred_ims)):
    op_file_name = op_dir + 'mean_' + str(ii) + '.png'
    grid_pred = pred_ims[ii]/max_val
    grid_pred = grid_pred*gp.mask + gp.mask_lines

    plt.imsave(op_file_name, 1-grid_pred, cmap='afmhot', vmin=0, vmax=1)

