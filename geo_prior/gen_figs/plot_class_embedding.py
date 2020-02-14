"""
Plots the learned class embedding matrix in 2D using TSNE.
Also launches interactive gui to get class names for each point - the names will
be displayed on the console.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.manifold import TSNE

import sys
sys.path.append('../')
from geo_prior import models
from geo_prior.paths import get_paths


seed = 2001
model_path = '../models/model_inat_2018_full_final.pth.tar'
data_dir = get_paths('inat_2018_data_dir')
op_dir = 'images/class_ims/'
if not os.path.isdir(op_dir):
    os.makedirs(op_dir)


with open(data_dir + 'categories2018.json') as da:
    cls_data = json.load(da)
class_names = [cc['name'] for cc in cls_data]
class_ids = [cc['id'] for cc in cls_data]
supercat_names = [cc['supercategory'] for cc in cls_data]
supercat_un, supercat_ids = np.unique(supercat_names, return_inverse=True)


# load model
net_params = torch.load(model_path, map_location='cpu')
params = net_params['params']
model = models.FCNet(num_inputs=params['num_feats'], num_classes=params['num_classes'],
                     num_filts=params['num_filts'], num_users=params['num_users']).to(params['device'])

model.load_state_dict(net_params['state_dict'])
model.eval()

weights = net_params['state_dict']['class_emb.weight'].numpy()
if 'class_emb.bias' in net_params['state_dict'].keys():
    biases = net_params['state_dict']['class_emb.bias'].numpy()
    weights = np.hstack((weights, biases[..., np.newaxis]))


print('Performing TSNE on all classes ...')
emb = TSNE(n_components=2, random_state=seed).fit_transform(weights)


plt.close('all')
plt.figure(0, figsize=(10,10))
plt.scatter(emb[:,0], emb[:,1], s=10)
plt.title('all classes')
plt.xticks([])
plt.yticks([])

print('Saving images to: ' + op_dir)
plt.savefig(op_dir + 'all_classes.png')
np.savez(op_dir + 'all_classes', emb=emb, class_names=class_names, class_ids=class_ids)
plt.show()


plt.close('all')
fig, ax = plt.subplots(num=1)
group_of_interest = 'all classes'
inds = np.arange(len(class_names))
classes_of_interest = class_names

clss = []
def onpick(event):
    ind = event.ind
    ii = ind[0]
    print(classes_of_interest[ii], class_names.index(classes_of_interest[ii]))
    clss.append(class_names.index(classes_of_interest[ii]))

plt.title(group_of_interest)
col = plt.scatter(emb[inds,0], emb[inds,1], s=2, picker=True)
plt.xticks([])
plt.yticks([])
fig.canvas.mpl_connect('pick_event', onpick)
plt.show()
