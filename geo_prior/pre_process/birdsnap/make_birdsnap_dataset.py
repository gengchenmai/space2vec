"""
The original BirdSnap dataset is not available to download, either is the location info.
We had to rescrape it, and this resulted in 7.5% missing test images and
6% missing train. Only 35-45% of the overall data has ground truth locations.

This script creates a new train, val, test split and applies location data from eBird.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import os
from calendar import monthrange
import random
import datetime

ip_dir = '/media/macaodha/Data/datasets/birdsnap/'
im_dir = ip_dir + 'images_sm/'

inat_path = '/media/macaodha/ssd_data/'
inat_year = '2017'
op_file = ip_dir + 'birdsnap_with_loc_2019.json'
np.random.seed(2001)
ims_per_class_val = 2

# Note need to run create_ebird.py script to generate these json files from the raw eBird data
ebird_path = '/media/macaodha/Data/ebird/'
ebird_train_path = ebird_path + '2015_ebird_raw.json'
ebird_valtest_path = ebird_path + '2016_ebird_raw.json'


# get stats of new birdsnap download
ip_new = ip_dir + 'birdsnap_manual.csv'
bs_man = pd.read_csv(ip_new)
bs_man['loc_lat'] = bs_man['loc_lat'].replace('None', np.nan)
bs_man['loc_lon'] = bs_man['loc_lon'].replace('None', np.nan)
bs_man['owner_nsid'] = bs_man['owner_nsid'].replace('TRYFAIL', None)
print 'total entries                ', len(bs_man)

month_count = np.cumsum([monthrange(2018, ii+1)[1] for ii in range(12)])
month_count = np.hstack(([0], month_count))

dates_c = []
dates = []
for ll in bs_man['date_taken'].values:
    if isinstance(ll, str):
        dt = ll.split(' ')[0]
        month = int(dt.split('-')[1])-1
        day = int(dt.split('-')[2])-1
        dt = (month_count[month] + day) / 365.0
        dates_c.append(round(dt,5))
        dates.append(ll)
    else:
        dates_c.append(ll)
        dates.append(ll)

im_ids = bs_man['cur_photo_id'].values.tolist()
_, user_ids = np.unique(bs_man['owner_nsid'].values, return_inverse=True)
lon = [round(float(ll),5) for ll in bs_man['loc_lon'].values]
lat = [round(float(ll),5) for ll in bs_man['loc_lat'].values]
print 'images with location info    ', len(lon) - np.isnan(lon).sum()

ims_per_user = bs_man['owner_nsid'].value_counts().values
print 'num images with valid users  ', ims_per_user.sum()
print 'num unique users             ', len(ims_per_user)
print 'num unique users > 25 ims    ', (ims_per_user > 25).sum()


# get overlap with iNat
data_dir = inat_path + '/inat_'+inat_year+'/'
with open(data_dir + 'inat'+inat_year+'_anns/categories'+inat_year+'.json') as da_inat:
    cls_data = json.load(da_inat)
inat = [cc['name'] for cc in cls_data]
#inat = [cc.lower() for cc in inat]  # does not matter


bs_data = ip_dir + 'birdsnap_orig/species.txt'
bs_classes = pd.read_csv(bs_data, sep='\t')['scientific'].values.tolist()
#bs_classes = [cc.lower() for cc in bs_classes]

print '\nspecies counts'
print len(bs_classes), 'birdsnap'
print len(inat), 'inat' + inat_year

overlap = []
missing = []
for cc in bs_classes:
    if cc in inat:
        overlap.append(cc)
    else:
        missing.append(cc)
print len(overlap), 'overlapping classes'



# create new dataset
bs_data = ip_dir + 'birdsnap_orig/images.txt'
bs = pd.read_csv(bs_data, sep='\t')

test_data = ip_dir + 'birdsnap_orig/test_images.txt'
test_ims = pd.read_csv(test_data, sep='\t')['path'].values.tolist()


# create val set from train images
is_train = np.ones(bs.shape[0], dtype=bool)
on_disk = np.zeros(bs.shape[0], dtype=bool)
for index, row in bs.iterrows():
    if row['path'] in test_ims:
        is_train[index] = False

    im_path = os.path.basename(row['url']).split('_')[0] + '.jpg'
    on_disk[index] = os.path.isfile(im_dir + im_path)

classes = bs['species_id'].values
is_val = np.zeros(bs.shape[0], dtype=bool)
for cc in np.unique(classes):
    inds = np.where(classes == cc)[0]
    valid_inds = inds[is_train[inds] & on_disk[inds]]
    val_inds = np.random.choice(valid_inds, ims_per_class_val, replace=False)
    is_val[val_inds] = True
    is_train[val_inds] = False


# get ebird location info to add to the dataset
print '\nloading ebird data'
with open(ebird_train_path) as da:
    ebird_train = json.load(da)
with open(ebird_valtest_path) as da:
    ebird_valtest = json.load(da)
assert ebird_train['species'] == ebird_valtest['species']
ebird_species = [ss.replace('_', ' ') for ss in ebird_train['species']]
ebird_classes_of_int = [ebird_species.index(ss) if ss in ebird_species else np.nan for ss in bs_classes]


def filter_ebird(ebird_data, ebird_classes_of_int, min_checklists=100):
    # filter data - only keep checklists with species of interest
    # also remove users that have done a small number of checklists

    un_obs, un_cnt = np.unique(ebird_data['observer_ids'], return_counts=True)
    un_obs = un_obs[np.where(un_cnt>min_checklists)[0]].tolist()

    keep_inds = []
    num_events = len(ebird_data['check_lists'])
    for ii in range(num_events):

        if ebird_data['observer_ids'][ii] in un_obs:

            overlap = list(set(ebird_classes_of_int)&set(ebird_data['check_lists'][ii]))
            if len(overlap) > 0:
                keep_inds.append(ii)

    ebird_data['lon_lat_tm'] = [ebird_data['lon_lat_tm'][kk] for kk in keep_inds]
    ebird_data['check_lists'] = [ebird_data['check_lists'][kk] for kk in keep_inds]
    ebird_data['observer_ids'] = [ebird_data['observer_ids'][kk] for kk in keep_inds]

    print '\tnum orig events', num_events, ', after filtering', len(ebird_data['check_lists'])
    return ebird_data


def sample_ebird(ebird_data, ebird_classes_of_int, classes, ebird_lon_lat, ebird_users, split_inds):

    ebird_rand_inds = range(len(ebird_data['check_lists']))
    num_classes = len(np.unique(classes))
    for bscls, ebcls in enumerate(ebird_classes_of_int):
        random.shuffle(ebird_rand_inds)
        inds = np.where((classes==bscls) & (split_inds))[0]
        cnt = 0
        for ii, ebind in enumerate(ebird_rand_inds):
            if ebcls in ebird_data['check_lists'][ebind]:
                ebird_lon_lat[inds[cnt]] = ebird_data['lon_lat_tm'][ebind]
                ebird_users[inds[cnt]] = str(ebird_data['observer_ids'][ebind])
                cnt += 1
            if cnt >= len(inds):
                break

    return ebird_lon_lat, ebird_users

# this is slow
# can result in duplicat locations if two species of interest are observed in the same location
ebird_lon_lat = [[np.nan, np.nan, np.nan]]*bs.shape[0] # each entry will be list with [lon, lat, tm]
ebird_users = [None]*bs.shape[0]
print 'sampling ebird for train'
ebird_train = filter_ebird(ebird_train, ebird_classes_of_int)
ebird_lon_lat, ebird_users = sample_ebird(ebird_train,   ebird_classes_of_int, classes, ebird_lon_lat, ebird_users, is_train)

print 'sampling ebird for val/test'
ebird_valtest = filter_ebird(ebird_valtest, ebird_classes_of_int)
ebird_lon_lat, ebird_users = sample_ebird(ebird_valtest, ebird_classes_of_int, classes, ebird_lon_lat, ebird_users, ~is_train)


# turn ebird user_ids into ints
not_miss = np.where(np.array(ebird_users) != None)[0]
_, ebird_users_int = np.unique(np.array(ebird_users)[not_miss], return_inverse=True)
for ii, ind in enumerate(not_miss):
    ebird_users[ind] = ebird_users_int[ii]

common_users = list(set(np.array(ebird_users)[is_train])&set(np.array(ebird_users)[~is_train]))
print len(common_users), 'users overlapping in train and (val+test)'


# construct dataset
train = []
test = []
val = []
for index, row in bs.iterrows():
    dp = {}
    dp['class_id'] = row['species_id'] - 1  # start with 0
    im_id = os.path.basename(row['url']).split('_')[0]
    dp['im_path'] = im_id + '.jpg'
    dp['orig_path'] = row['path']
    dp['orig_date'] = dates[ind]
    ind = im_ids.index(int(im_id))

    dp['orig_meta'] = {}
    dp['orig_meta']['lon'] = lon[ind]
    dp['orig_meta']['lat'] = lat[ind]
    dp['orig_meta']['date'] = dates_c[ind]
    dp['orig_meta']['user_id'] = user_ids[ind]

    dp['ebird_meta'] = {}
    dp['ebird_meta']['lon'] = ebird_lon_lat[ind][0]
    dp['ebird_meta']['lat'] = ebird_lon_lat[ind][1]
    dp['ebird_meta']['date'] = ebird_lon_lat[ind][2]
    dp['ebird_meta']['user_id'] = ebird_users[ind]

    if on_disk[index]:
        dp['valid_image'] = True
    else:
        dp['valid_image'] = False

    if is_train[index]:
        train.append(dp)
    elif is_val[index]:
        val.append(dp)
    else:
        test.append(dp)


print '\nrevised birdnap stats'
num_train_ims = np.sum([dd['valid_image'] for dd in train])
print 'num missing train ims        ', len(train) - num_train_ims
num_test_ims = np.sum([dd['valid_image'] for dd in test])
print 'num missing test ims         ', len(test) - num_test_ims
num_val_ims = np.sum([dd['valid_image'] for dd in val])
print 'num missing val ims          ', len(val) - num_val_ims


# save file
bs_op = {}
bs_op['train'] = train
bs_op['test']  = test
bs_op['val']  = val
bs_op['classes'] = bs_classes
bs_op['info'] = 'BirdsSnap dataset. Generated on ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print 'saving output to ', op_file
with open(op_file, 'w') as da:
    json.dump(bs_op, da)

