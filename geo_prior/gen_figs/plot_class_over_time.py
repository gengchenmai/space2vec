"""
Plots location predictions for a given class over time.
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


# 3731 Hylocichla mustelina - Wood thrush
# 1447 Danaus plexippus - Monarch butterfly
# 4210 Loxodonta africana - Africian bush elephant
# 488 Apis mellifera - Western honey bee
class_of_interest = 3731

model_path = '../models/model_inat_2018_full_final.pth.tar'
data_dir = get_paths('inat_2018_data_dir')

op_dir = 'images/class_over_time_ims/'
if not os.path.isdir(op_dir):
    os.makedirs(op_dir)

# load class names
with open(data_dir + 'categories2018.json') as da:
    cls_data = json.load(da)
class_names = [cc['name'] for cc in cls_data]
class_ids = [cc['id'] for cc in cls_data]
class_dict = dict(zip(class_ids, class_names))


# select classes of interest
print('Class of interest: ' + str(class_of_interest) + ' ' + class_dict[class_of_interest])
op_dir = op_dir + str(class_of_interest) + '/'
if not os.path.isdir(op_dir):
    os.makedirs(op_dir)


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


print('Making predictions for each time step')
print('Saving images to: ' + op_dir)
pred_ims = []
for ii, tm in enumerate(np.linspace(0,1,12)):
   grid_pred = gp.dense_prediction(model, class_of_interest, tm)
   pred_ims.append(grid_pred)
   op_file_name = op_dir + str(class_of_interest).zfill(4) + '_' + str(ii) + '.png'
   plt.imsave(op_file_name, 1-grid_pred, cmap='afmhot', vmin=0, vmax=1)

