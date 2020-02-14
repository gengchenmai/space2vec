"""
Plots the learned user embedding matrix.

First run plot_class_embedding.py to generate the class embedding.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import json
from sklearn.manifold import TSNE
import os

import sys
sys.path.append('../')
from geo_prior import models
from geo_prior.paths import get_paths
import geo_prior.datasets as dt
import geo_prior.grid_predictor as grid


users_of_interest = [344, 345, 42]
users_of_interest_cols = ['r', 'y', 'k']

model_path = '../models/model_inat_2018_full_final.pth.tar'
data_dir = get_paths('inat_2018_data_dir')
min_num_exs = 100
seed = 2001
dpi = 150.0
num_time_steps = 12
op_dir = 'images/user_ims/'
if not os.path.isdir(op_dir):
    os.makedirs(op_dir)

# this has been precomputed by plot_class_embedding.py
if os.path.isfile('images/class_ims/all_classes.npz'):
    class_embedding = np.load('images/class_ims/all_classes.npz')
else:
    print('Error: Need to run plot_class_embedding.py first.')
    sys.exit()

# load class info
with open(data_dir + 'categories2018.json') as da:
    cls_data = json.load(da)
class_names = [cc['name'] for cc in cls_data]
class_ids = [cc['id'] for cc in cls_data]
supercat_names = [cc['supercategory'] for cc in cls_data]
supercat_un, supercat_ids = np.unique(supercat_names, return_inverse=True)

# load user info
train_locs, train_classes, train_users, train_dates, _ = dt.load_inat_data(data_dir,
    'train2018_locations.json', 'train2018.json', True)
assert (train_users==-1).sum() == 0
un_users, train_users, cnt_users = np.unique(train_users, return_inverse=True, return_counts=True)


# load model
net_params = torch.load(model_path, map_location='cpu')
params = net_params['params']
model = models.FCNet(num_inputs=params['num_feats'], num_classes=params['num_classes'],
                     num_filts=params['num_filts'], num_users=params['num_users']).to(params['device'])

model.load_state_dict(net_params['state_dict'])
model.eval()


# load params
user_emb = net_params['state_dict']['user_emb.weight'].numpy()
class_emb = net_params['state_dict']['class_emb.weight'].numpy()
# currently this will not work if there is a bias term in the model
assert model.inc_bias is False
def sig(x):
    return 1.0 / (1.0 + np.exp(-x))
user_class_affinity = sig(np.dot(user_emb, class_emb.T))


# filter based on num of examples from each user
print('\nTotal number of users ' + str(user_emb.shape[0]))
new_user_inds = np.where(cnt_users >= min_num_exs)[0]
user_emb = user_emb[new_user_inds, :]
train_users = train_users[new_user_inds]
user_class_affinity = user_class_affinity[new_user_inds, :]
print('Total number of users after filtering ' + str(user_emb.shape[0]))


print('Performing TSNE on users with min ' + str(min_num_exs) + ' examples')
user_emb_low = TSNE(n_components=2, random_state=seed).fit_transform(user_emb)


# load ocean mask
mask = np.load(get_paths('mask_dir') + 'ocean_mask.npy')
mask_lines = (np.gradient(mask)[0]**2 + np.gradient(mask)[1]**2)
mask_lines[mask_lines > 0.0] = 1.0
gp = grid.GridPredictor(mask, params, mask_only_pred=True)
loc_emb = gp.dense_prediction_masked_feats(model, 0.5).data.cpu().numpy()


# compute locations where users go
mask_inds = np.where(mask.ravel() == 1)[0]
user_loc = np.zeros((mask.shape[0]*mask.shape[1]))
user_loc[mask_inds] = sig(np.dot(loc_emb, user_emb.T)).sum(1)
user_loc = user_loc.reshape((mask.shape[0], mask.shape[1]))
user_loc = np.log(1 + user_loc)
user_loc[mask_lines==1] = user_loc.max()

plt.close('all')
cmap_r = cm.afmhot.reversed()

# plot user affinity for each location
plt.figure(1)
plt.imshow(user_loc, cmap=cmap_r)
plt.title('photographer location affinity (log)')
plt.xticks([])
plt.yticks([])
plt.imsave(op_dir + 'photographer_loc_log.png', user_loc, cmap=cmap_r)


# plotting selected users
plt.figure(2)
plt.scatter(user_emb_low[:,0], user_emb_low[:,1], s=5)
plt.title('photographer embedding')
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.savefig(op_dir + 'photographer_embed.png', dpi=dpi)
for ii, ind in enumerate(users_of_interest):
    plt.plot(user_emb_low[ind,0], user_emb_low[ind,1], users_of_interest_cols[ii]+'o')
plt.savefig(op_dir + 'photographer_embed_with_inds.png', dpi=dpi)


# plot users affinity for each class
for ii, ind in enumerate(users_of_interest):
    plt.figure(ii+3)
    #pt_size = (np.exp(1-user_class_affinity[users_of_interest[ii], :])-1)*10
    pt_size = (user_class_affinity[users_of_interest[ii], :])
    plt.scatter(class_embedding['emb'][:,0], class_embedding['emb'][:,1], s=3, c=pt_size)#, cmap='magma')
    plt.axis('equal')
    plt.axis('off')
    plt.title(users_of_interest[ii])
    plt.tight_layout()
    plt.savefig(op_dir + str(users_of_interest[ii]) + '_photographer_affinity.png', dpi=dpi)

plt.show()