"""
Extracts features from a trained network for each geo location (and time), performs
dimensionality reduction, and generates an output image.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import decomposition

import sys
sys.path.append('../')
from geo_prior import models
from geo_prior.paths import get_paths
import geo_prior.datasets as dt
import geo_prior.grid_predictor as grid


model_path = '../models/model_inat_2018_full_final.pth.tar'
num_time_steps = 12
num_ds_dims = 3
seed = 2001
op_dir = 'images/map_ims/'
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
gp = grid.GridPredictor(mask, params, mask_only_pred=True)

# compute intermediate network features
print('Computing features.')
feats = []
for time_step in np.linspace(0,1,num_time_steps+1)[:-1]:
    net_feats = gp.dense_prediction_masked_feats(model, time_step).data.cpu().numpy()
    feats.append(net_feats)


# downsample features - choose middle time step
dsf = decomposition.FastICA(n_components=num_ds_dims, random_state=seed)
dsf.fit(feats[len(feats) // 2])

op_ims = []
mask_inds = np.where(mask.ravel() == 1)[0]
mins = []
maxes = []
for ii in range(len(feats)):
    feats_ds = dsf.transform(feats[ii])

    # convert into image
    op_im = np.ones((mask.shape[0]*mask.shape[1], num_ds_dims))
    op_im[mask_inds] = feats_ds
    mins.append(op_im[mask_inds].min())
    maxes.append(op_im[mask_inds].max())
    op_ims.append(op_im)

# normalize to same range
min_val = np.min(mins)
max_val = (np.max(maxes) - min_val)
for ii in range(len(op_ims)):
    op_ims[ii][mask_inds] -= min_val
    op_ims[ii][mask_inds] /= max_val

for ii in range(len(op_ims)):
    op_ims[ii] = op_ims[ii].reshape((mask.shape[0], mask.shape[1], num_ds_dims))

print('Saving images to: ' + op_dir)
for ii in range(len(op_ims)):
    plt.imsave(op_dir + 'map_im_' + str(ii).zfill(3) + '.png', (op_ims[ii]*255).astype(np.uint8))

plt.close('all')
plt.figure(0)
plt.imshow(op_ims[len(op_ims)//2])
plt.show()

