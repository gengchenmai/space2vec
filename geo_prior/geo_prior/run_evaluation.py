"""
Script to evaluate different spatio-temporal priors.
"""

import numpy as np
import json
from scipy import sparse
import torch
import math
import pandas as pd
import os
from sklearn.neighbors import BallTree, DistanceMetric

from paths import get_paths
import utils as ut
import datasets as dt
import baselines as bl
import models


def compute_acc(val_preds, val_classes, val_split, val_feats=None, train_classes=None,
                train_feats=None, prior_type='no_prior', prior=None, hyper_params=None):
    '''
    Computes accuracy on held out set with a specified prior. Not very efficient
    as it loops though each example one at a time.
    Args:
        val_preds: CNN pretrained model's image prediction of class
        val_classes: [batch_size, 1], the list of image category id
        val_split: for bridsnap, np.ones() (batch_size)
        val_feats: the inpit location features, shape [batch_size, x]
        train_classes:
        train_feats:
        prior_type: 'geo_net'
        prior: the model itself
    Return:
        pred_classes: (batch_size), the list of joint predicted image category
    '''
    

    top_k_acc = {}
    for kk in [1, 3, 5, 10]:
        top_k_acc[kk] = np.zeros(len(val_classes))
    max_class = np.max(list(top_k_acc.keys()))
    pred_classes = [] # the list of joint predicted image category

    for ind in range(len(val_classes)):

        # select the type of prior to be used
        if prior_type == 'no_prior':
            pred = val_preds[ind, :]

        elif prior_type == 'train_freq':
            pred = val_preds[ind, :]*prior

        elif prior_type == 'nn_dist':
            geo_prior = bl.compute_neighbor_prior(train_classes, val_preds.shape[1],
                        val_feats[ind, :], prior, hyper_params, ptype='distance')
            pred = val_preds[ind, :]*geo_prior

        elif prior_type == 'nn_knn':
            geo_prior = bl.compute_neighbor_prior(train_classes, val_preds.shape[1],
                           val_feats[ind, :], prior, hyper_params, ptype='knn')
            pred = val_preds[ind, :]*geo_prior

        elif prior_type == 'kde':
            geo_prior = bl.kde_prior(train_classes, train_feats, val_preds.shape[1],
                           val_locs[ind, :], prior, hyper_params)
            pred = val_preds[ind, :]*geo_prior

        elif prior_type == 'grid':
            geo_prior = prior.eval(val_feats[ind, :])
            pred = val_preds[ind, :]*geo_prior

        elif prior_type in ['geo_net'] + ut.get_spa_enc_list():
            # if there is no location info won't use prior
            # pred: the pretrained CNN image class prediction distribution
            pred = val_preds[ind, :]
            with torch.no_grad():
                # if all image have location infor
                if torch.isnan(val_feats[ind, 0]).item() == 0:
                    # net_prior: (1, num_classes), the spa_enc model image class prediction distribution
                    net_prior = prior(val_feats[ind, :].unsqueeze(0))
                    net_prior = net_prior.cpu().data.numpy()[0, :].astype(np.float64)
                    #net_prior /= net_prior.sum()  # does not matter for argmax
                    pred = pred*net_prior

        elif prior_type == 'tang_et_al':
            # if there is no location info won't use prior
            pred = val_preds[ind, :]
            with torch.no_grad():
                if torch.isnan(val_feats['val_locs'][ind, 0]).item() == 0:
                    # takes location and network features as input
                    pred = prior(val_feats['val_locs'][ind, :].unsqueeze(0),
                                      val_feats['val_feats'][ind, :].unsqueeze(0))
                    pred = pred.cpu().data.numpy()[0, :].astype(np.float64)


        # store accuracy of prediction
        pred_classes.append(np.argmax(pred))
        top_N = np.argsort(pred)[-max_class:]
        for kk in top_k_acc.keys():
            if val_classes[ind] in top_N[-kk:]:
                top_k_acc[kk][ind] = 1

    # print final accuracy
    # some datasets have mutiple splits. These are represented by integers for each example in val_split
    for ii, split in enumerate(np.unique(val_split)):
        print(' Split ID: {}'.format(ii))
        inds = np.where(val_split == split)[0]
        for kk in np.sort(list(top_k_acc.keys())):
            print(' Top {}\tacc (%):   {}'.format(kk, round(top_k_acc[kk][inds].mean()*100, 2)))

    return pred_classes


def get_cross_val_hyper_params(eval_params):

    hyper_params = {}
    if eval_params['dataset'] == 'inat_2018':
        hyper_params['num_neighbors'] = 1500
        hyper_params['dist_type'] = 'euclidean'  # euclidean, haversine
        hyper_params['dist_thresh'] = 2.0  # kms if haversine - divide by radius earth
        hyper_params['gp_size'] = [180, 60]
        hyper_params['pseudo_count'] = 2
        hyper_params['kde_dist_type'] = 'euclidean'  # for KDE
        hyper_params['kde_quant'] = 5.0  # for KDE
        hyper_params['kde_nb'] = 700  # for KDE

    elif eval_params['dataset'] == 'inat_2017':
        hyper_params['num_neighbors'] = 1450
        hyper_params['dist_type'] = 'euclidean'
        hyper_params['dist_thresh'] = 5.0
        hyper_params['gp_size'] = [45, 30]
        hyper_params['pseudo_count'] = 2
        hyper_params['kde_dist_type'] = 'euclidean'
        hyper_params['kde_quant'] = 5.0
        hyper_params['kde_nb'] = 700

    elif eval_params['dataset'] == 'birdsnap' and eval_params['meta_type'] == 'ebird_meta':
        hyper_params['num_neighbors'] = 700
        hyper_params['dist_type'] = 'euclidean'
        hyper_params['dist_thresh'] = 5.0
        hyper_params['gp_size'] = [30, 30]
        hyper_params['pseudo_count'] = 2
        hyper_params['kde_dist_type'] = 'euclidean'
        hyper_params['kde_quant'] = 0.001
        hyper_params['kde_nb'] = 500

    elif eval_params['dataset'] == 'birdsnap' and eval_params['meta_type'] == 'orig_meta':
        hyper_params['num_neighbors'] = 100
        hyper_params['dist_type'] = 'euclidean'
        hyper_params['dist_thresh'] = 9.0
        hyper_params['gp_size'] = [225, 60]
        hyper_params['pseudo_count'] = 2
        hyper_params['kde_dist_type'] = 'euclidean'
        hyper_params['kde_quant'] = 0.001
        hyper_params['kde_nb'] = 600

    elif eval_params['dataset'] == 'nabirds':
        hyper_params['num_neighbors'] = 500
        hyper_params['dist_type'] = 'euclidean'
        hyper_params['dist_thresh'] = 6.0
        hyper_params['gp_size'] = [45, 60]
        hyper_params['pseudo_count'] = 2
        hyper_params['kde_dist_type'] = 'euclidean'
        hyper_params['kde_quant'] = 0.001
        hyper_params['kde_nb'] = 600

    elif eval_params['dataset'] == 'yfcc':
        hyper_params['num_neighbors'] = 75
        hyper_params['dist_type'] = 'haversine'
        hyper_params['dist_thresh'] = 2.0/6371.4
        hyper_params['gp_size'] = [540, 150]
        hyper_params['pseudo_count'] = 3
        hyper_params['kde_dist_type'] = 'euclidean'
        hyper_params['kde_quant'] = 0.001
        hyper_params['kde_nb'] = 300

    return hyper_params


if __name__ == "__main__":

    eval_params = {}
    eval_params['dataset'] = 'birdsnap'  # inat_2018, inat_2017, birdsnap, nabirds, yfcc
    eval_params['eval_split'] = 'test'  # train, val, test
    eval_params['inat2018_resolution'] = 'standard' # 'standard' or 'high_res' - only valid for inat_2018
    eval_params['meta_type'] = 'ebird_meta'  # orig_meta, ebird_meta - only for nabirds, birdsnap
    eval_params['model_type'] = '' # '_full_final', '_no_date_final', '_no_photographer_final', '_no_encode_final'
    eval_params['trained_models_root'] = '../models/'  # location where trained models are stored
    eval_params['save_op'] = False

    eval_params['spa_enc'] = "rbf"

    # specify which algorithms to evaluate. Ours is 'geo_net'.
    #eval_params['algs'] = ['no_prior', 'train_freq', 'geo_net', 'tang_et_al', 'grid', 'nn_knn', 'nn_dist', 'kde']
    eval_params['algs'] = ['no_prior', eval_params['spa_enc']]

    # if torch.cuda.is_available():
    #     device = torch.device("cuda:1")
    #     eval_params['device'] = device
    # else:
    #     eval_params['device'] = 'cpu'
    eval_params['device'] = 'cpu'

    # path to trained models
    meta_str = ''
    if eval_params['dataset'] in ['birdsnap', 'nabirds']:
        meta_str = '_' + eval_params['meta_type']
    nn_model_path = "{}model_{}{}_{}{}.pth.tar".format(
        eval_params['trained_models_root'],
        eval_params['dataset'],
        meta_str,
        eval_params['spa_enc'],
        eval_params['model_type']) 
    nn_model_path_tang = "{}bl_tang_{}{}_gps.pth.tar".format(
        eval_params['trained_models_root'],
        eval_params['dataset'],
        meta_str
        )
    # if eval_params['spa_enc'] == "rbf":
    #     rbf_anchor_pt_ids_file_name = "{}rbf_pts_{}{}_{}{}.pkl".format(
    #         eval_params['trained_models_root'],
    #         eval_params['dataset'],
    #         meta_str,
    #         eval_params['spa_enc'],
    #         eval_params['model_type'])
    #     rbf_anchor_pt_ids = ut.pickle_load(rbf_anchor_pt_ids_file_name)

    # nn_model_path_tang = eval_params['trained_models_root'] + 'bl_tang_'+eval_params['dataset']+meta_str+'_gps.pth.tar'

    print('Dataset    \t' + eval_params['dataset'])
    print('Eval split \t' + eval_params['eval_split'])

    # load data and features
    if 'tang_et_al' in eval_params['algs']:
        op = dt.load_dataset(eval_params, eval_params['eval_split'], True, False, True, True, False)
    else:
        op = dt.load_dataset(eval_params, eval_params['eval_split'], True, False, True, False, False)

    train_locs = op['train_locs']
    train_classes = op['train_classes']
    train_users = op['train_users']
    train_dates = op['train_dates']
    val_locs = op['val_locs']
    val_classes = op['val_classes']
    val_users = op['val_users']
    val_dates = op['val_dates']
    class_of_interest = op['class_of_interest']
    classes = op['classes']
    num_classes = op['num_classes']
    val_preds = op['val_preds']
    val_split = op['val_split']

    # these hyper parameters have been cross validated for the baseline methods
    hyper_params = get_cross_val_hyper_params(eval_params)


    #
    # no prior
    #
    if 'no_prior' in eval_params['algs']:
        print('\nNo prior')
        pred_no_prior = compute_acc(val_preds, val_classes, val_split, prior_type='no_prior')


    #
    # overall training frequency prior
    #
    if 'train_freq' in eval_params['algs']:
        print('\nTrain frequency prior')
        # weight the eval predictions by the overall frequency of each class at train time
        cls_id, cls_cnt = np.unique(train_classes, return_counts=True)
        train_prior = np.ones(num_classes)
        train_prior[cls_id] += cls_cnt
        train_prior /= train_prior.sum()
        compute_acc(val_preds, val_classes, val_split, prior_type='train_freq', prior=train_prior)


    #
    # neural network spatio-temporal prior
    #
    # if 'geo_net' in eval_params['algs']:
    #     print('\nNeural net prior')
    #     print(' Model :\t' + os.path.basename(nn_model_path))
    #     net_params = torch.load(nn_model_path)
    #     params = net_params['params']

    #     # construct features
    #     val_locs_scaled = ut.convert_loc_to_tensor(val_locs)
    #     val_dates_scaled = torch.from_numpy(val_dates.astype(np.float32)*2 - 1)
    #     val_feats_net = ut.encode_loc_time(val_locs_scaled, val_dates_scaled, concat_dim=1, params=params)

    #     model = models.FCNet(params['num_feats'], params['num_classes'], params['num_filts'], params['num_users'])
    #     model.load_state_dict(net_params['state_dict'])
    #     model.eval()
    #     pred_geo_net = compute_acc(val_preds, val_classes, val_split, val_feats=val_feats_net, prior_type='geo_net', prior=model)

    #
    # spatial encoder neural network spatio-temporal prior
    #
    spa_enc_algs = set(ut.get_spa_enc_list() + ['geo_net'])
    spa_enc_algs = set(eval_params['algs']).intersection(spa_enc_algs)
    if len(spa_enc_algs) == 1:
        spa_enc_type = list(spa_enc_algs)[0]
        
        print('\n{}'.format(spa_enc_type))
        print(' Model :\t' + os.path.basename(nn_model_path))

        net_params = torch.load(nn_model_path)
        params = net_params['params']

        # construct features
        # val_feats_net: shape [batch_size, 2], torch.tensor
        val_feats_net = ut.generate_model_input_feats(
                spa_enc_type = params['spa_enc_type'], 
                locs = val_locs, 
                dates = val_dates, 
                params = params,
                device = eval_params['device'])

        model = ut.get_model(
            train_locs = train_locs,
            params = params, 
            spa_enc_type = params['spa_enc_type'], 
            num_inputs = params['num_feats'], 
            num_classes = params['num_classes'], 
            num_filts = params['num_filts'], 
            num_users = params['num_users'], 
            device = eval_params['device'])

        model.load_state_dict(net_params['state_dict'])
        model.eval()
        pred_geo_net = compute_acc(val_preds, val_classes, val_split, val_feats=val_feats_net, prior_type=spa_enc_type, prior=model)

    #
    # Tang et al ICCV 2015, Improving Image Classification with Location Context
    #
    if 'tang_et_al' in eval_params['algs']:
        print('\nTang et al. prior')
        print('  using model :\t' + os.path.basename(nn_model_path_tang))
        net_params = torch.load(nn_model_path_tang)
        params = net_params['params']

        # construct features
        val_feats_tang = {}
        val_feats_tang['val_locs']  = ut.convert_loc_to_tensor(val_locs)
        val_feats_tang['val_feats'] = torch.from_numpy(op['val_feats'])
        assert params['loc_encoding'] == 'gps'

        model = models.TangNet(params['loc_feat_size'], params['net_feats_dim'],
                               params['embedding_dim'], params['num_classes'], params['use_loc'])
        model.load_state_dict(net_params['state_dict'])
        model.eval()
        compute_acc(val_preds, val_classes, val_split, val_feats=val_feats_tang, prior_type='tang_et_al', prior=model)
        del val_feats_tang  # save memory

    #
    # discretized grid prior
    #
    if 'grid' in eval_params['algs']:
        print('\nDiscrete grid prior')
        gp = bl.GridPrior(train_locs, train_classes, num_classes, hyper_params)
        compute_acc(val_preds, val_classes, val_split, val_feats=val_locs, prior_type='grid', prior=gp,
                    hyper_params=hyper_params)


    #
    # setup look up tree for NN lookup based methods
    #
    if ('nn_knn' in eval_params['algs']) or ('nn_dist' in eval_params['algs']):
        if hyper_params['dist_type'] == 'haversine':
            nn_tree = BallTree(np.deg2rad(train_locs)[:,::-1], metric='haversine')
            val_locs_n = np.deg2rad(val_locs)
        else:
            nn_tree = BallTree(train_locs[:,::-1], metric='euclidean')
            val_locs_n = val_locs


    #
    # nearest neighbor prior - based on KNN
    #
    if 'nn_knn' in eval_params['algs']:
        print('\nNearest neighbor KNN prior')
        compute_acc(val_preds, val_classes, val_split, val_feats=val_locs_n, train_classes=train_classes,
                    prior_type='nn_knn', prior=nn_tree, hyper_params=hyper_params)


    #
    # nearest neighbor prior - based on distance
    #
    if 'nn_dist' in eval_params['algs']:
        print('\nNearest neighbor distance prior')
        compute_acc(val_preds, val_classes, val_split, val_feats=val_locs_n, train_classes=train_classes,
                    prior_type='nn_dist', prior=nn_tree, hyper_params=hyper_params)


    #
    # kernel density estimate e.g. BirdSnap CVPR 2014
    #
    if 'kde' in eval_params['algs']:
        print('\nKernel density estimate prior')
        kde_params = {}
        train_classes_kde, train_locs_kde, kde_params['counts'] = bl.create_kde_grid(train_classes, train_locs, hyper_params)
        if hyper_params['kde_dist_type'] == 'haversine':
            train_locs_kde = np.deg2rad(train_locs_kde)
            val_locs_kde = np.deg2rad(val_locs)
            kde_params['nn_tree_kde'] = BallTree(train_locs_kde[:, ::-1], metric='haversine')
        else:
            val_locs_kde = val_locs
            kde_params['nn_tree_kde'] = BallTree(train_locs_kde[:, ::-1], metric='euclidean')

        compute_acc(val_preds, val_classes, val_split, val_feats=val_locs_kde, train_classes=train_classes_kde,
                    train_feats=train_locs_kde, prior_type='kde', prior=kde_params, hyper_params=hyper_params)


    if eval_params['save_op']:
        np.savez('model_preds', val_classes=val_classes, pred_geo_net=pred_geo_net,
            pred_no_prior=pred_no_prior, dataset=eval_params['dataset'],
            split=eval_params['eval_split'], model_type=eval_params['model_type'])
