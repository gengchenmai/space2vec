import numpy as np
import torch
import json
import os
import math
import pickle
from torch.utils.data.sampler import Sampler

from SpatialRelationEncoder import *
from module import *
import models

def encode_loc_time(loc_ip, date_ip, concat_dim=1, params=None):
    '''
    Args:
        loc_ip: shape [batch_size, 2], torch.tensor, 2 means (lon, lat)
        date_ip: shape [batch_size]
    Return:
        feat: shape [batch_size, x]
    '''
    # assumes inputs location and date features are in range -1 to 1
    # location is lon, lat

    if params['loc_encode'] == 'encode_cos_sin':
        feats = torch.cat((torch.sin(math.pi*loc_ip), torch.cos(math.pi*loc_ip)), concat_dim)

    elif params['loc_encode'] == 'encode_3D':
        # X, Y, Z in 3D space
        if concat_dim == 1:
            cos_lon = torch.cos(math.pi*loc_ip[:, 0]).unsqueeze(-1)
            sin_lon = torch.sin(math.pi*loc_ip[:, 0]).unsqueeze(-1)
            cos_lat = torch.cos(math.pi*loc_ip[:, 1]).unsqueeze(-1)
            sin_lat = torch.sin(math.pi*loc_ip[:, 1]).unsqueeze(-1)
        if concat_dim == 2:
            cos_lon = torch.cos(math.pi*loc_ip[:, :, 0]).unsqueeze(-1)
            sin_lon = torch.sin(math.pi*loc_ip[:, :, 0]).unsqueeze(-1)
            cos_lat = torch.cos(math.pi*loc_ip[:, :, 1]).unsqueeze(-1)
            sin_lat = torch.sin(math.pi*loc_ip[:, :, 1]).unsqueeze(-1)
        feats = torch.cat((cos_lon*cos_lat, sin_lon*cos_lat, sin_lat), concat_dim)

    elif params['loc_encode'] == 'encode_none':
        feats = loc_ip

    else:
        print('error - no loc feat type defined')


    if params['use_date_feats']:
        if params['date_encode'] == 'encode_cos_sin':
            feats_date = torch.cat((torch.sin(math.pi*date_ip.unsqueeze(-1)),
                                    torch.cos(math.pi*date_ip.unsqueeze(-1))), concat_dim)
        elif params['date_encode'] == 'encode_none':
            feats_date = date_ip.unsqueeze(-1)
        else:
            print('error - no date feat type defined')
        feats = torch.cat((feats, feats_date), concat_dim)

    return feats


class BalancedSampler(Sampler):
    # sample "evenly" from each from class
    def __init__(self, classes, num_per_class, use_replace=False, multi_label=False):
        '''
        Args:
            classes: list(), [batch_size], the list of image category id
            num_per_class: the max number of sample per class
            use_replace: whether or not do sample with replacement
        '''
        self.class_dict = {}
        self.num_per_class = num_per_class
        self.use_replace = use_replace
        self.multi_label = multi_label

        if self.multi_label:
            self.class_dict = classes
        else:
            # standard classification
            un_classes = np.unique(classes)
            for cc in un_classes:
                self.class_dict[cc] = []

            for ii in range(len(classes)):
                self.class_dict[classes[ii]].append(ii)
            '''
            class_dict: dict()
                key: the class id
                value: a list of image sample index who belong to this class
            '''
        if self.use_replace:
            self.num_exs = self.num_per_class*len(un_classes)
        else:
            self.num_exs = 0
            for cc in self.class_dict.keys():
                self.num_exs += np.minimum(len(self.class_dict[cc]), self.num_per_class)


    def __iter__(self):
        indices = []
        for cc in self.class_dict:
            if self.use_replace:
                indices.extend(np.random.choice(self.class_dict[cc], self.num_per_class).tolist())
            else:
                indices.extend(np.random.choice(self.class_dict[cc], np.minimum(len(self.class_dict[cc]),
                                                self.num_per_class), replace=False).tolist())
        # in the multi label setting there will be duplictes at training time
        np.random.shuffle(indices)  # will remain a list
        return iter(indices)

    def __len__(self):
        return self.num_exs


def convert_loc_to_tensor(x, device=None):
    '''
    Args:
        x: shape [batch_size, 2], 2 means (lon, lat)
    Return:
        xt: shape [batch_size, 2], torch.tensor
    '''
    # intput is in lon {-180, 180}, lat {90, -90}
    xt = x.astype(np.float32)
    xt[:,0] /= 180.0
    xt[:,1] /= 90.0
    xt = torch.from_numpy(xt)
    if device is not None:
        xt = xt.to(device)
    return xt


def distance_pw_euclidean(xx, yy):
    # equivalent to scipy.spatial.distance.cdist
    dist = np.sqrt((xx**2).sum(1)[:, np.newaxis] - 2*xx.dot(yy.transpose()) + ((yy**2).sum(1)[np.newaxis, :]))
    return dist


def distance_pw_haversine(xx, yy, radius=6372.8):
    # input should be in radians
    # output is in km's if radius = 6372.8

    d_lon = xx[:, 0][..., np.newaxis] - yy[:, 0][np.newaxis, ...]
    d_lat = xx[:, 1][..., np.newaxis] - yy[:, 1][np.newaxis, ...]

    cos_term = np.cos(xx[:,1])[..., np.newaxis]*np.cos(yy[:, 1])[np.newaxis, ...]
    dist = np.sin(d_lat/2.0)**2 + cos_term*np.sin(d_lon/2.0)**2
    dist = 2 * radius * np.arcsin(np.sqrt(dist))
    return dist


def euclidean_distance(xx, yy):
    return np.sqrt(((xx - yy)**2).sum(1))


def haversine_distance(xx, yy, radius=6371.4):
    # assumes shape N x 2, where col 0 is lat, and col 1 is lon
    # input should be in radians
    # output is in km's if radius = 6371.4
    # note that SKLearns haversine distance is [latitude, longitude] not [longitude, latitude]

    d_lon = xx[:, 0] - yy[0]
    d_lat = xx[:, 1] - yy[1]

    cos_term = np.cos(xx[:,1])*np.cos(yy[1])
    dist = np.sin(d_lat/2.0)**2 + cos_term*np.sin(d_lon/2.0)**2
    dist = 2 * radius * np.arcsin(np.sqrt(dist + 1e-16))

    return dist


def bilinear_interpolate(loc_ip, data, remove_nans=False):
    # loc is N x 2 vector, where each row is [lon,lat] entry
    #   each entry spans range [-1,1]
    # data is H x W x C, height x width x channel data matrix
    # op will be N x C matrix of interpolated features

    # map to [0,1], then scale to data size
    loc = (loc_ip.clone() + 1) / 2.0
    loc[:,1] = 1 - loc[:,1]   # this is because latitude goes from +90 on top to bottom while
                              # longitude goes from -90 to 90 left to right
    if remove_nans:
        loc[torch.isnan(loc)] = 0.5
    loc[:, 0] *= (data.shape[1]-1)
    loc[:, 1] *= (data.shape[0]-1)

    loc_int = torch.floor(loc).long()  # integer pixel coordinates
    xx = loc_int[:, 0]
    yy = loc_int[:, 1]
    xx_plus = xx + 1
    xx_plus[xx_plus > (data.shape[1]-1)] = data.shape[1]-1
    yy_plus = yy + 1
    yy_plus[yy_plus > (data.shape[0]-1)] = data.shape[0]-1

    loc_delta = loc - torch.floor(loc)   # delta values
    dx = loc_delta[:, 0].unsqueeze(1)
    dy = loc_delta[:, 1].unsqueeze(1)
    interp_val = data[yy, xx, :]*(1-dx)*(1-dy) + data[yy, xx_plus, :]*dx*(1-dy) + \
                 data[yy_plus, xx, :]*(1-dx)*dy   + data[yy_plus, xx_plus, :]*dx*dy

    return interp_val


class AverageMeter:
    # computes and stores the average and current value

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / float(self.count)




############################# added new code ##########################
def pickle_dump(obj, pickle_filepath):
    with open(pickle_filepath, "wb") as f:
        pickle.dump(obj, f, protocol=2)

def pickle_load(pickle_filepath):
    with open(pickle_filepath, "rb") as f:
        obj = pickle.load(f)
    return obj

def get_spa_enc_list():
    return ["gridcell", "hexagridcell", "theory", "theorydiag", "naive", "rbf"]

def generate_feats(locs, dates, params, device):
    '''
    Args:
        locs: numpy.array, shape [batch_size, 2], 2 means (lon, lat)
        dates: numpy.array, shape [batch_size], dates
        params:
    Return:
        feats: the encoded input features including lon, lat, date, [batch_size, input_feat_dim]
    '''
    # x_locs: shape [batch_size, 2], torch.tensor
    x_locs = convert_loc_to_tensor(locs, device)
    # x_dates: shape [batch_size], torch.tensor
    x_dates = torch.from_numpy(dates.astype(np.float32)*2 - 1).to(device)
    # feats: shape [batch_size, 2] or [batch_size, 3] (use_date_feats=True), torch.tensor
    feats = encode_loc_time(x_locs, x_dates, concat_dim=1, params=params)
    return feats

def generate_model_input_feats(spa_enc_type, locs, dates, params, device):
    '''
    We rewrite the function, make the input features tensor
    Args:
        spa_enc_type: 
        locs: numpy.array, shape [batch_size, 2], 2 means (lon, lat)
        dates: numpy.array, shape [batch_size], dates
        params:
        device: "cuda" or "cpu"
    Return:
        feats: torch.tensor, shape [batch_size, 2] or [batch_size, 3]
                the encoded input features including lon, lat, date, [batch_size, input_feat_dim]
    '''
    if spa_enc_type == "geo_net":
        # train_feats: shape [batch_size, 2] or [batch_size, 3] (use_date_feats=True), torch.tensor
        feats = generate_feats(locs, dates, params, device)
    elif spa_enc_type in get_spa_enc_list():
        # train_feats: shape [batch_size, 2], torch.tensor
        feats = convert_loc_to_tensor_no_normalize(
            x=locs, device=device)
    else:
        raise Exception("spa_enc not defined for loc normalization!!!")
    return feats

def convert_loc_to_tensor_no_normalize(x, device=None):
    '''
    Args:
        x: shape [batch_size, 2], 2 means (lon, lat)
    Return:
        xt: shape [batch_size, 2], torch.tensor
    '''
    # intput is in lon {-180, 180}, lat {90, -90}
    xt = x.astype(np.float32)
    # xt[:,0] /= 180.0
    # xt[:,1] /= 90.0
    xt = torch.from_numpy(xt)
    if device is not None:
        xt = xt.to(device)
    return xt

def get_ffn(params, input_dim, output_dim, f_act, context_str = ""):
    
    return MultiLayerFeedForwardNN(
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden_layers=params['num_hidden_layer'],
            dropout_rate=params['dropout'],
            hidden_dim=params['hidden_dim'],
            activation=f_act,
            use_layernormalize=params['use_layn'],
            skip_connection = params['skip_connection'],
            context_str = context_str)


def get_spa_encoder(train_locs, params, spa_enc_type, spa_embed_dim, extent, coord_dim = 2,
                    frequency_num = 16, 
                    max_radius = 10000, min_radius = 1, 
                    f_act = "sigmoid", freq_init = "geometric",
                    num_rbf_anchor_pts = 100, rbf_kernal_size = 10e2,  
                    use_postmat = True,
                    device = "cuda"):
    if spa_enc_type == "gridcell":
        ffn = get_ffn(params,
            input_dim=int(4 * frequency_num),
            output_dim=spa_embed_dim,
            f_act = f_act,
            context_str = "GridCellSpatialRelationEncoder")
        spa_enc = GridCellSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "hexagridcell":
        spa_enc = HexagonGridCellSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            dropout = params['dropout'], 
            f_act= f_act,
            device=device)
    elif spa_enc_type == "theory":
        ffn = get_ffn(params,
            input_dim=int(6 * frequency_num),
            output_dim=spa_embed_dim,
            f_act = f_act,
            context_str = "TheoryGridCellSpatialRelationEncoder")
        spa_enc = TheoryGridCellSpatialRelationEncoder(
            spa_embed_dim,
            coord_dim = coord_dim,
            frequency_num = frequency_num,
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "theorydiag":
        spa_enc = TheoryDiagGridCellSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius, 
            min_radius = min_radius,
            dropout = params['dropout'], 
            f_act= f_act, 
            freq_init = freq_init, 
            use_layn = params['use_layn'], 
            use_post_mat = use_postmat,
            device=device)
    elif spa_enc_type == "naive":
        ffn = get_ffn(params,
            input_dim=2,
            output_dim=spa_embed_dim,
            f_act = f_act,
            context_str = "NaiveSpatialRelationEncoder")
        spa_enc = NaiveSpatialRelationEncoder(
            spa_embed_dim, 
            extent = extent, 
            coord_dim = coord_dim, 
            ffn = ffn,
            device=device)
    elif spa_enc_type == "rbf":
        ffn = get_ffn(params,
            input_dim=num_rbf_anchor_pts,
            output_dim=spa_embed_dim,
            f_act = f_act,
            context_str = "RBFSpatialRelationEncoder")
        spa_enc = RBFSpatialRelationEncoder(
            model_type = "global", 
            train_locs = train_locs,
            spa_embed_dim = spa_embed_dim,
            coord_dim = coord_dim, 
            num_rbf_anchor_pts = num_rbf_anchor_pts,
            rbf_kernal_size = rbf_kernal_size,
            rbf_kernal_size_ratio = 0,
            max_radius = max_radius,
            ffn=ffn,
            rbf_anchor_pt_ids = params['rbf_anchor_pt_ids'],
            device = device)
    # elif spa_enc_type == "aodha":
    #     extent = get_spatial_context(model_type, pointset, max_radius)
    #     spa_enc = AodhaSpatialRelationEncoder(
    #         spa_embed_dim, 
    #         extent = extent, 
    #         coord_dim = coord_dim,
    #         num_hidden_layers = params['num_hidden_layer'],
    #         hidden_dim = params['hidden_dim'],
    #         use_post_mat=use_postmat,
    #         f_act=f_act)
    # elif spa_enc_type == "none":
    #     assert spa_embed_dim == 0
    #     spa_enc = None
    else:
        raise Exception("Space encoder function no support!")
    return spa_enc


def get_model(train_locs, params, spa_enc_type, num_inputs, num_classes, num_filts, num_users, device):
    '''
    Make the location encoder model
    '''
    if spa_enc_type == "geo_net":
        return models.FCNet(
                num_inputs=num_inputs, 
                num_classes=num_classes,
                num_filts=num_filts, 
                num_users=num_users).to(device)
    elif spa_enc_type in get_spa_enc_list():
        spa_enc = get_spa_encoder(
                train_locs = train_locs,
                params = params, 
                spa_enc_type = spa_enc_type, 
                spa_embed_dim = num_filts, 
                extent = params['map_range'], 
                coord_dim = num_inputs,
                frequency_num = params['frequency_num'], 
                max_radius = params['max_radius'], 
                min_radius = params['min_radius'], 
                f_act = params['spa_f_act'], 
                freq_init = params['freq_init'],
                num_rbf_anchor_pts = params['num_rbf_anchor_pts'], 
                rbf_kernal_size = params['rbf_kernal_size'],  
                use_postmat = params['spa_enc_use_postmat'], 
                device = device).to(device)

        return models.LocationEncoder(
                spa_enc = spa_enc, 
                num_inputs = num_inputs, 
                num_classes = num_classes, 
                num_filts = num_filts, 
                num_users = num_users).to(device)
    else:
        raise Exception("spa_enc not defined!!!")