"""
Loads in dense feature files and saves them as sparse numpy files.
"""

import numpy as np
from scipy import sparse

data_dir = '/media/macaodha/ssd_data/inat_2017/inat2017_preds/'
#splits = ['train', 'val', 'test']
splits = ['train']

for split in splits:
    print('Loading : ' + 'inat2017_'+split+'_preds.npy')
    dd = np.load(data_dir + 'inat2017_'+split+'_preds.npy')
    dd[dd<0.000001] = 0.0
    sp = sparse.csr_matrix(dd)
    sparse.save_npz(data_dir + 'inat2017_'+split+'_preds_sparse.npy', sp)

