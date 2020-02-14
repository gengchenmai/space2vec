# No official train/test split is provided with the YFCC100M_GEO100 dataset from ICCV 15.
# This code generates a random split where the train size is 75%, val is 5%, and test is 20%.

import pandas as pd
import numpy as np

train_size = 0.75
test_size = 0.2
val_size = 0.05

np.random.seed(2001)

root_dir = '/media/macaodha/Data/datasets/YFCC100M_GEO100/'
op_file_name = root_dir + 'train_test_split.csv'
op_cls_file_name = root_dir + 'class_names.csv'
ip_file_name = root_dir + 'photo2gps.txt'

da = pd.read_csv(ip_file_name, sep=' ', names=['path', 'lat', 'lon'])

ids = np.arange(da.shape[0])
np.random.shuffle(ids)

train_pt = int(len(ids)*train_size)
val_pt = train_pt + int(len(ids)*val_size)

train_ids = ids[:train_pt]
val_ids = ids[train_pt:val_pt]
test_ids = ids[val_pt:]


print('train', len(train_ids))
print('val  ', len(val_ids))
print('test ', len(test_ids))

print('total', da.shape[0])
print('sum splits', len(train_ids)+len(val_ids)+len(test_ids))

split = ['']*len(ids)
for jj in train_ids:
    split[jj] = 'train'
for jj in val_ids:
    split[jj] = 'val'
for jj in test_ids:
    split[jj] = 'test'

cname = [cc.split('/')[0] for cc in da['path'].values]
class_unique_names, class_label = np.unique(cname, return_inverse=True)

da['split'] = split
da['class'] = class_label

class_dict = {'id':range(len(class_unique_names)), 'name':class_unique_names}
da_cls = pd.DataFrame.from_dict(class_dict)

da.to_csv(op_file_name, sep=',', index=False)
da_cls.to_csv(op_cls_file_name, sep=',', index=False)

