import json
import pandas as pd
import numpy as np
import random

np.random.seed(2001)
nabirds_dir = '/media/macaodha/Data/datasets/nabirds/'
ebird_path = '/media/macaodha/Data/ebird/'
op_file = nabirds_dir + 'nabirds_with_loc_2019.json'
ims_per_class_val = 2

ims = pd.read_csv(nabirds_dir + 'images.txt', sep=' ', names=['id', 'path'])['path'].values
is_train = pd.read_csv(nabirds_dir + 'train_test_split.txt', sep=' ', names=['id', 'split'])['split'].values.astype(np.bool)
classes = pd.read_csv(nabirds_dir + 'image_class_labels.txt', sep=' ', names=['id', 'label'])['label'].values
un_classes, classes = np.unique(classes, return_inverse=True)
num_ims = len(ims)

with open(nabirds_dir + 'classes.txt') as fp:
    class_names = fp.readlines()
class_names = [cc[cc.index(' ')+1:].replace('\n', '') for cc in class_names]
class_names_orig = class_names[:]
class_names = [cc.replace(' ', '_') for cc in class_names]
class_names = [cc[:cc.index('(')-1] if '(' in cc else cc for cc in class_names]

# this species has been split so this is a hack
# running twice as NABirds has male and female as separate classes
class_names[class_names.index('Western_Scrub-Jay')] = 'California_Scrub-Jay'
class_names[class_names.index('Western_Scrub-Jay')] = 'California_Scrub-Jay'

class_names_of_int = [class_names[cc] for cc in un_classes]
class_names_orig_of_int = [class_names_orig[cc] for cc in un_classes]
print 'total classes - inc m/f/j:', len(class_names_of_int)
print 'unique classes           :', len(np.unique(class_names_of_int))


print '\ngetting ebird species overlap'
ebird_train_path = ebird_path + '2015_ebird_raw.json'
ebird_valtest_path = ebird_path + '2016_ebird_raw.json'
ebird_classes_common = pd.read_csv(ebird_path + 'ERD2016SS/doc/taxonomy.csv')['PRIMARY_COM_NAME'].values.tolist()
ebird_classes_sci = pd.read_csv(ebird_path + 'ERD2016SS/doc/taxonomy.csv')['SCI_NAME'].values.tolist()

missing = []
ebird_classes_of_int_name = []
for cc in class_names_of_int:
    if cc not in ebird_classes_common:
        missing.append(cc)
        ebird_classes_of_int_name.append(None)
    else:
        ind = ebird_classes_common.index(cc)
        ebird_classes_of_int_name.append(ebird_classes_sci[ind])

if len(missing) > 0:
    print len(missing), 'species not in ebird taxonomy'


# make val set
is_val = np.zeros(num_ims, dtype=bool)
for cc in np.unique(classes):
    inds = np.where(classes == cc)[0]
    valid_inds = inds[is_train[inds]]
    val_inds = np.random.choice(valid_inds, ims_per_class_val, replace=False)
    is_val[val_inds] = True
    is_train[val_inds] = False


print '\nloading ebird data'
with open(ebird_train_path) as da:
    ebird_train = json.load(da)
with open(ebird_valtest_path) as da:
    ebird_valtest = json.load(da)
assert ebird_train['species'] == ebird_valtest['species']

ebird_classes_of_int = [ebird_train['species'].index(ss) if ss in ebird_train['species'] else np.nan for ss in ebird_classes_of_int_name]
print np.isnan(ebird_classes_of_int).sum(), 'species not in ebird data'

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
ebird_lon_lat = [[np.nan, np.nan, np.nan]]*num_ims # each entry will be list with [lon, lat, tm]
ebird_users = [None]*num_ims
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
for ii in range(num_ims):
    dp = {}
    dp['class_id'] = classes[ii]
    dp['im_path'] = ims[ii]
    dp['valid_image'] = True

    dp['ebird_meta'] = {}
    dp['ebird_meta']['lon'] = ebird_lon_lat[ii][0]
    dp['ebird_meta']['lat'] = ebird_lon_lat[ii][1]
    dp['ebird_meta']['date'] = ebird_lon_lat[ii][2]
    dp['ebird_meta']['user_id'] = ebird_users[ii]

    if is_train[ii]:
        train.append(dp)
    elif is_val[ii]:
        val.append(dp)
    else:
        test.append(dp)


# save file
bs_op = {}
bs_op['train'] = train
bs_op['test']  = test
bs_op['val']  = val
bs_op['classes'] = class_names_orig_of_int
bs_op['classes_sci'] = ebird_classes_of_int_name

print '\nsaving output to:\n', op_file
with open(op_file, 'w') as da:
    json.dump(bs_op, da)
