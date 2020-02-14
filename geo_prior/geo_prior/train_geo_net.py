import numpy as np
import matplotlib.pyplot as plt
import math
import os
import torch
import pickle

import models
import utils as ut
import datasets as dt
import grid_predictor as grid
from paths import get_paths
import losses as lo


class LocationDataLoader(torch.utils.data.Dataset):
    def __init__(self, loc_feats, labels, users, num_classes, is_train):
        self.loc_feats = loc_feats
        self.labels = labels
        self.users = users
        self.is_train = is_train
        self.num_classes = num_classes


    def __len__(self):
        return len(self.loc_feats)

    def __getitem__(self, index):
        loc_feat  = self.loc_feats[index, :]
        loc_class = self.labels[index]
        user      = self.users[index]
        if self.is_train:
            return loc_feat, loc_class, user
        else:
            return loc_feat, loc_class





def train(model, data_loader, optimizer, epoch, params):
    model.train()

    # adjust the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = params['lr'] * (params['lr_decay'] ** epoch)

    loss_avg = ut.AverageMeter()
    inds = torch.arange(params['batch_size']).to(params['device'])

    for batch_idx, (loc_feat, loc_class, user_ids) in enumerate(data_loader):
        '''
        loc_feat: (batch_size, input_feat_dim)
        loc_class: (batch_size)
        user_ids: (batch_size)
        '''
        optimizer.zero_grad()

        loss = lo.embedding_loss(model, params, loc_feat, loc_class, user_ids, inds)

        loss.backward()
        optimizer.step()

        loss_avg.update(loss.item(), len(loc_feat))

        if (batch_idx % params['log_frequency'] == 0 and batch_idx != 0) or (batch_idx == (len(data_loader)-1)):
            print('[{}/{}]\tLoss  : {:.4f}'.format(batch_idx * params['batch_size'], len(data_loader.dataset), loss_avg.avg))


def test(model, data_loader, params):
    # NOTE the test loss only tracks the BCE it is not the full loss used during training
    # the test loss is the -log() of the correct class prediction probability, the lower is better
    model.eval()
    loss_avg = ut.AverageMeter()

    inds = torch.arange(params['batch_size']).to(params['device'])
    with torch.no_grad():

        for loc_feat, loc_class in data_loader:
            '''
            loc_feat: (batch_size, input_feat_dim)
            loc_class: (batch_size)
            '''
            # loc_pred: (batch_size, num_classes)
            loc_pred = model(loc_feat)
            # pos_loss: (batch_size)
            pos_loss = lo.bce_loss(loc_pred[inds[:loc_feat.shape[0]], loc_class])
            loss = pos_loss.mean()

            loss_avg.update(loss.item(), loc_feat.shape[0])

    print('Test loss   : {:.4f}'.format(loss_avg.avg))


def plot_gt_locations(params, mask, train_classes, class_of_interest, classes, train_locs, train_dates, op_dir):
    '''
    plot GT locations for the class of interest, with mask in the backgrpund
    Args:
        params:
        mask: (1002, 2004) mask for the earth, 
              (lat,  lon ), so that when you plot it, it will be naturally the whole globe
        train_classes: [batch_size, 1], the list of image category id
        class_of_interest: 0
        classes: a dict(), class id => class name
        train_locs: [batch_size, 2], location data
        train_dates: [batch_size, 1], the list of date
        op_dir: 
    '''
    
    im_width  = (params['map_range'][1] - params['map_range'][0]) // 45  # 8
    im_height = (params['map_range'][3] - params['map_range'][2]) // 45  # 4
    plt.figure(num=0, figsize=[im_width, im_height])
    plt.imshow(mask, extent=params['map_range'], cmap='tab20')

    '''
    when np.where(condition, x, y) with no x,y, it like np.asarray(condition).nonzero()
    np.where(train_classes==class_of_interest) return a tuple, 
    a tuple of arrays, one for each dimension of a, 
    containing the indices of the non-zero elements in that dimension
    '''
    # inds: the indices in train_classes 1st dim where the class id == class_of_interest
    inds = np.where(train_classes==class_of_interest)[0]
    print('{} instances of: '.format(len(inds)) + classes[class_of_interest])

    # the color of the dot indicates the date
    colors = np.sin(np.pi*train_dates[inds])
    plt.scatter(train_locs[inds, 0], train_locs[inds, 1], c=colors, s=2, cmap='magma', vmin=0, vmax=1)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().set_frame_on(False)

    op_file_name = op_dir + 'gt_' + str(class_of_interest).zfill(4) + '.jpg'
    plt.savefig(op_file_name, dpi=400, bbox_inches='tight',pad_inches=0)


def main():

    # hyper params
    params = {}
    params['dataset'] = 'birdsnap'  # inat_2018, inat_2017, birdsnap, nabirds, yfcc
    if params['dataset'] in ['birdsnap', 'nabirds']:
        params['meta_type'] = 'ebird_meta'  # orig_meta, ebird_meta
    else:
        params['meta_type'] = ''
    params['batch_size'] = 1024
    params['lr'] = 0.001
    params['lr_decay'] = 0.98
    params['num_filts'] = 256  # embedding dimension
    params['num_epochs'] = 30
    params['log_frequency'] = 50
    # params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:1")
    #     params['device'] = device
    # else:
    #     params['device'] = 'cpu'
    params['device'] = 'cpu'

    params['balanced_train_loader'] = True
    params['max_num_exs_per_class'] = 100
    params['map_range'] = (-180, 180, -90, 90)

    # specify feature encoding for location and date
    params['use_date_feats'] = False  # if False date feature is not used
    params['loc_encode']     = 'encode_cos_sin'  # encode_cos_sin, encode_3D, encode_none
    params['date_encode']    = 'encode_cos_sin' # encode_cos_sin, encode_none


    # specify loss type
    # appending '_user' models the user location and object affinity - see losses.py
    params['train_loss'] = 'full_loss_user'  # full_loss_user, full_loss

    ############# add new parameters #########################
    params['spa_enc_type'] = 'theory'   # the type of space encoder type
    params['frequency_num'] = 64        # The number of frequency used in the space encoder
    params['max_radius'] = 360          # The maximum spatial context radius in the space encoder
    params['min_radius'] = 0.0005       # The minimum spatial context radius in the space encoder
    params['spa_f_act'] = "relu"        # The final activation function used by spatial relation encoder
    params['freq_init'] = "geometric"   # The frequency list initialization method
    params['spa_enc_use_postmat'] = True    # whether to use post matrix in spa_enc
    params['num_rbf_anchor_pts'] = 200  # The number of RBF anchor points used in the "rbf" space encoder
    params['rbf_kernal_size'] = 1    # The RBF kernal size in the "rbf" space encoder
    



    params['num_hidden_layer'] = 1      # The number of hidden layer in feedforward NN in the (global) space encoder
    params['hidden_dim'] = 512          # The hidden dimention in feedforward NN in the (global) space encoder 
    params['use_layn'] = True           # use layer normalization or not in feedforward NN in the (global) space encoder
    params['skip_connection'] = True    # skip connection or not in feedforward NN in the (global) space encoder
    params['dropout'] = 0.5             # The dropout rate used in feedforward NN in the (global) space encoder

    ##########################################################

    print('Dataset   \t' + params['dataset'])
    op = dt.load_dataset(params, 'val', True, True)
    # train_locs: np.arrary, [batch_size, 2], location data
    train_locs = op['train_locs']
    # train_classes: np.arrary, [batch_size], the list of image category id
    train_classes = op['train_classes']
    # train_users: np.arrary, [batch_size], the list of user id
    train_users = op['train_users']
    # train_dates: np.arrary, [batch_size], the list of date
    train_dates = op['train_dates']
    val_locs = op['val_locs']
    val_classes = op['val_classes']
    val_users = op['val_users']
    val_dates = op['val_dates']
    class_of_interest = op['class_of_interest']
    classes = op['classes']

    params['num_classes'] = op['num_classes']


    # params['rbf_anchor_pt_ids']: the samples indices in train_locs whose correponding points are unsed as rbf anbchor points
    if params['spa_enc_type'] == 'rbf':
        params['rbf_anchor_pt_ids'] = list(np.random.choice(np.arange(len(train_locs)), 
                    params['num_rbf_anchor_pts'], replace=False))
        
    else:
        params['rbf_anchor_pt_ids'] = None

    if params['meta_type'] == '':
        params['model_file_name'] = "../models/model_{}_{}.pth.tar".format(params['dataset'], params['spa_enc_type']) 
    else:
        params['model_file_name'] = "../models/model_{}_{}_{}.pth.tar".format(params['dataset'], params['meta_type'], params['spa_enc_type'])
    op_dir = "image/ims_{}_{}/".format(params['dataset'], params['spa_enc_type'])
    if not os.path.isdir(op_dir):
        os.makedirs(op_dir)

    # process users
    # NOTE we are only modelling the users in the train set - not the val
    # un_users: a sorted list of unique user id
    # train_users: the indices in un_users which indicate the original train user id
    un_users, train_users = np.unique(train_users, return_inverse=True)
    train_users = torch.from_numpy(train_users).to(params['device'])
    params['num_users'] = len(un_users)
    if 'user' in params['train_loss']:
        assert (params['num_users'] != 1)  # need to have more than one user

    # print stats
    print('\nnum_classes\t{}'.format(params['num_classes']))
    print('num train    \t{}'.format(len(train_locs)))
    print('num val      \t{}'.format(len(val_locs)))
    print('train loss   \t' + params['train_loss'])
    print('model name   \t' + params['model_file_name'])
    print('num users    \t{}'.format(params['num_users']))
    if params['meta_type'] != '':
        print('meta data    \t' + params['meta_type'])

    # load ocean mask for plotting
    mask = np.load(get_paths('mask_dir') + 'ocean_mask.npy').astype(np.int)

    # data loaders
    train_labels = torch.from_numpy(train_classes).to(params['device'])
    # train_feats: torch.tensor, shape [batch_size, 2] or [batch_size, 3]
    train_feats = ut.generate_model_input_feats(
                spa_enc_type = params['spa_enc_type'], 
                locs = train_locs, 
                dates = train_dates, 
                params = params,
                device = params['device'])
    train_dataset = LocationDataLoader(train_feats, train_labels, train_users, params['num_classes'], True)
    if params['balanced_train_loader']:
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=params['batch_size'],
                       sampler=ut.BalancedSampler(train_classes.tolist(), params['max_num_exs_per_class'],
                       use_replace=False, multi_label=False), shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=params['batch_size'], shuffle=True)

    val_labels = torch.from_numpy(val_classes).to(params['device'])
    val_feats = ut.generate_model_input_feats(
                spa_enc_type = params['spa_enc_type'], 
                locs = val_locs, 
                dates = val_dates, 
                params = params,
                device = params['device'])
    val_dataset = LocationDataLoader(val_feats, val_labels, val_users, params['num_classes'], False)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=params['batch_size'], shuffle=False)

    # create model
    params['num_feats'] = train_feats.shape[1]
    # model = models.FCNet(num_inputs=params['num_feats'], num_classes=params['num_classes'],
    #                      num_filts=params['num_filts'], num_users=params['num_users']).to(params['device'])
    model = ut.get_model(
            train_locs = train_locs,
            params = params, 
            spa_enc_type = params['spa_enc_type'], 
            num_inputs = params['num_feats'], 
            num_classes = params['num_classes'], 
            num_filts = params['num_filts'], 
            num_users = params['num_users'], 
            device = params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # set up grid to make dense prediction across world
    gp = grid.GridPredictor(mask, params)

    # plot ground truth
    plt.close('all')
    plot_gt_locations(params, mask, train_classes, class_of_interest, classes, train_locs, train_dates, op_dir)


    # main train loop
    for epoch in range(0, params['num_epochs']):
        print('\nEpoch\t{}'.format(epoch))
        train(model, train_loader, optimizer, epoch, params)
        test(model, val_loader, params)

        # save dense prediction image
        # grid_pred: (1002, 2004)
        grid_pred = gp.dense_prediction(model, class_of_interest)
        op_file_name = op_dir + str(epoch).zfill(4) + '_' + str(class_of_interest).zfill(4) + '.jpg'
        plt.imsave(op_file_name, 1-grid_pred, cmap='afmhot', vmin=0, vmax=1)


    if params['use_date_feats']:
        print('\nGenerating predictions for each month of the year.')
        if not os.path.isdir(op_dir + 'time/'):
            os.makedirs(op_dir + 'time/')
        for ii, tm in enumerate(np.linspace(0,1,13)):
           grid_pred = gp.dense_prediction(model, class_of_interest, tm)
           op_file_name = op_dir + 'time/' + str(class_of_interest).zfill(4) + '_' + str(ii) + '.jpg'
           plt.imsave(op_file_name, 1-grid_pred, cmap='afmhot', vmin=0, vmax=1)


    # save trained model
    print('Saving output model to ' + params['model_file_name'])
    op_state = {'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'params' : params}
    torch.save(op_state, params['model_file_name'])


if __name__== "__main__":
    main()