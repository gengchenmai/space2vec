import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import logging
import random
import time
import numpy as np


from spacegraph_codebase.encoder import *
from spacegraph_codebase.SpatialRelationEncoder import *
from spacegraph_codebase.decoder import *
from spacegraph_codebase.model import *

def cudify(feature_embedding):
    '''
    Make the features function with cuda mode
    Args:
        feature_embedding: a dict of embedding matrix by node type, each embed matrix shape: [num_ent_by_type + 2, embed_dim]
        
    Return:
        features(pt_types): a embeddin look up function 
            pt_types: a lists of point type ids
            
    '''
    feature_embed_lookup = lambda pt_types: feature_embedding(
            torch.autograd.Variable(torch.LongTensor(pt_types).cuda()))
    return feature_embed_lookup

def setup_console():
    logging.getLogger('').handlers = []
    console = logging.StreamHandler()
    # optional, set the logging level
    console.setLevel(logging.INFO)
    # set a format which is the same for console use
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    
def setup_logging(log_file, console=True, filemode='w'):
    #logging.getLogger('').handlers = []
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode=filemode)
    if console:
        #logging.getLogger('').handlers = []
        console = logging.StreamHandler()
        # optional, set the logging level
        console.setLevel(logging.INFO)
        # set a format which is the same for console use
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    return logging


def get_encoder(feature_embed_lookup, feature_embedding, pointset, enc_agg):
    if enc_agg == "mean":
        enc = PointFeatureEncoder(feature_embed_lookup, feature_embedding, pointset, agg_func=torch.mean)
    elif enc_agg == "min":
        enc = PointFeatureEncoder(feature_embed_lookup, feature_embedding, pointset, agg_func=torch.min)
    elif enc_agg == "max":
        enc = PointFeatureEncoder(feature_embed_lookup, feature_embedding, pointset, agg_func=torch.max)
    else:
        raise Exception("Aggregation function no support!")
    return enc

def get_ffn(args, input_dim, f_act, context_str = ""):
    # print("Create 3 FeedForward NN!!!!!!!!!!")
    if args.use_layn == "T":
        use_layn = True
    else:
        use_layn = False
    if args.skip_connection == "T":
        skip_connection = True
    else:
        skip_connection = False
#     if args.use_post_mat == "T":
#         use_post_mat = True
#     else:
#         use_post_mat = False
    return MultiLayerFeedForwardNN(
            input_dim=input_dim,
            output_dim=args.spa_embed_dim,
            num_hidden_layers=args.num_hidden_layer,
            dropout_rate=args.dropout,
            hidden_dim=args.hidden_dim,
            activation=f_act,
            use_layernormalize=use_layn,
            skip_connection = skip_connection,
            context_str = context_str)

def get_spatial_context(model_type, pointset, max_radius):
    if model_type == "global":
        # extent = pointset.make_spatial_extent()
        extent = (-1710000, -1690000+1000, 1610000, 1640000+1000)
    elif model_type == "relative":
        extent = (-max_radius-500, max_radius+500, -max_radius-500, max_radius+500)
    return extent

def get_spa_encoder(args, model_type, spa_enc_type, pointset, spa_embed_dim, coord_dim = 2,
                    num_rbf_anchor_pts = 100, rbf_kernal_size = 10e2, frequency_num = 16, 
                    max_radius = 10000, min_radius = 1, f_act = "sigmoid", freq_init = "geometric", use_postmat = "T"):
    if args.use_layn == "T":
        use_layn = True
    else:
        use_layn = False
    
    if use_postmat == "T":
        use_post_mat = True
    else:
        use_post_mat = False
    if spa_enc_type == "gridcell":
        ffn = get_ffn(args,
            input_dim=int(4 * frequency_num),
            f_act = f_act,
            context_str = "GridCellSpatialRelationEncoder")
        spa_enc = GridCellSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn)
    elif spa_enc_type == "hexagridcell":
        spa_enc = HexagonGridCellSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            dropout = args.dropout, 
            f_act= f_act)
    elif spa_enc_type == "theory":
        ffn = get_ffn(args,
            input_dim=int(6 * frequency_num),
            f_act = f_act,
            context_str = "TheoryGridCellSpatialRelationEncoder")
        spa_enc = TheoryGridCellSpatialRelationEncoder(
            spa_embed_dim,
            coord_dim = coord_dim,
            frequency_num = frequency_num,
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn)
    elif spa_enc_type == "theorydiag":
        spa_enc = TheoryDiagGridCellSpatialRelationEncoder(
            spa_embed_dim, coord_dim = coord_dim, frequency_num = frequency_num, max_radius = max_radius, min_radius = min_radius,
            dropout = args.dropout, f_act= f_act, freq_init = freq_init, use_layn = use_layn, use_post_mat = use_post_mat)
    elif spa_enc_type == "naive":
        extent = get_spatial_context(model_type, pointset, max_radius)
        ffn = get_ffn(args,
            input_dim=2,
            f_act = f_act,
            context_str = "NaiveSpatialRelationEncoder")
        spa_enc = NaiveSpatialRelationEncoder(spa_embed_dim, extent = extent, coord_dim = coord_dim, ffn = ffn)
    elif spa_enc_type == "polar":
        ffn = get_ffn(args,
            input_dim=2,
            f_act = f_act,
            context_str = "PolarCoordSpatialRelationEncoder")
        spa_enc = PolarCoordSpatialRelationEncoder(spa_embed_dim, coord_dim = coord_dim, ffn = ffn)
    elif spa_enc_type == "polardist":
        ffn = get_ffn(args,
            input_dim=1,
            f_act = f_act,
            context_str = "PolarDistCoordSpatialRelationEncoder")
        spa_enc = PolarDistCoordSpatialRelationEncoder(spa_embed_dim, coord_dim = coord_dim, ffn = ffn)
    elif spa_enc_type == "polargrid":
        ffn = get_ffn(args,
            input_dim=int(2 * frequency_num),
            f_act = f_act,
            context_str = "PolarGridCoordSpatialRelationEncoder")
        spa_enc = PolarGridCoordSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num,
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn)
    elif spa_enc_type == "rbf":
        ffn = get_ffn(args,
            input_dim=num_rbf_anchor_pts,
            f_act = f_act,
            context_str = "RBFSpatialRelationEncoder")
        spa_enc = RBFSpatialRelationEncoder(
            model_type, pointset,
            spa_embed_dim,
            coord_dim = coord_dim, 
            num_rbf_anchor_pts = num_rbf_anchor_pts,
            rbf_kernal_size = rbf_kernal_size,
            rbf_kernal_size_ratio = args.rbf_kernal_size_ratio,
            max_radius = max_radius,
            ffn=ffn)
    elif spa_enc_type == "distrbf":
        spa_enc = DistRBFSpatialRelationEncoder(
            spa_embed_dim, coord_dim = coord_dim,
            num_rbf_anchor_pts = num_rbf_anchor_pts, rbf_kernal_size = rbf_kernal_size, max_radius = max_radius,
            dropout = dropout, f_act = f_act)
    elif spa_enc_type == "gridlookup":
        ffn = get_ffn(args,
            input_dim=args.spa_embed_dim,
            f_act = f_act,
            context_str = "GridLookupSpatialRelationEncoder")

        extent = get_spatial_context(model_type, pointset, max_radius)
        
        spa_enc = GridLookupSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            interval = min_radius, 
            extent = extent, 
            ffn = ffn)
    elif spa_enc_type == "gridlookupnoffn":
        extent = get_spatial_context(model_type, pointset, max_radius)

        spa_enc = GridLookupSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            interval = min_radius, 
            extent = extent, 
            ffn = None)
    elif spa_enc_type == "polargridlookup":
        assert model_type == "relative"
        ffn = get_ffn(args,
            input_dim=args.spa_embed_dim,
            f_act = f_act,
            context_str = "PolarGridLookupSpatialRelationEncoder")
        spa_enc = PolarGridLookupSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            max_radius = max_radius, 
            frequency_num = frequency_num, 
            ffn = ffn)
    elif spa_enc_type == "aodha":
        extent = get_spatial_context(model_type, pointset, max_radius)
        spa_enc = AodhaSpatialRelationEncoder(
            spa_embed_dim, 
            extent = extent, 
            coord_dim = coord_dim,
            num_hidden_layers = args.num_hidden_layer,
            hidden_dim = args.hidden_dim,
            use_post_mat=use_post_mat,
            f_act=f_act)
    elif spa_enc_type == "none":
        assert spa_embed_dim == 0
        spa_enc = None
    else:
        raise Exception("Space encoder function no support!")
    return spa_enc


def get_context_decoder(dec_type, query_dim, key_dim, spa_embed_dim, g_spa_embed_dim, have_query_embed = True,
                        num_attn = 1,  activation = "leakyrelu", f_activation = "sigmoid", 
                        layn = "T", use_postmat = "T", dropout = 0.5):
    if layn == "T":
        layernorm = True
    else:
        layernorm = False

    if use_postmat == "T":
        use_post_mat = True
    else:
        use_post_mat = False

    if dec_type == "concat":
        dec = IntersectConcatAttention(query_dim, key_dim, spa_embed_dim, have_query_embed = have_query_embed, num_attn = num_attn, 
        activation = activation, f_activation = f_activation, 
        layernorm = layernorm, use_post_mat = use_post_mat, dropout = dropout)
    elif dec_type == "g_pos_concat":
        dec = GolbalPositionIntersectConcatAttention(query_dim, key_dim, spa_embed_dim, g_spa_embed_dim, have_query_embed = have_query_embed, num_attn = num_attn, 
        activation = activation, f_activation = f_activation, 
        layernorm = layernorm, use_post_mat = use_post_mat, dropout = dropout)
    else:
        raise Exception("decoder type not support!")

    return dec

def get_enc_dec(model_type, pointset, enc, spa_enc = None, 
    g_spa_enc = None, g_spa_dec = None, init_dec=None, dec=None, joint_dec=None, 
    activation = "sigmoid", num_context_sample = 10, num_neg_resample = 10):
    if model_type == "relative": # relative position encoding only
        enc_dec = NeighGraphEncoderDecoder(pointset=pointset, 
                    enc=enc, 
                    spa_enc=spa_enc, 
                    init_dec=init_dec, 
                    dec=dec, 
                    activation = activation,
                    num_context_sample = num_context_sample, 
                    num_neg_resample = num_neg_resample)
    elif model_type == "global": # global position encoding only
        enc_dec = GlobalPositionEncoderDecoder(pointset=pointset, 
                    enc = enc, 
                    g_spa_enc = g_spa_enc, 
                    g_spa_dec = g_spa_dec, 
                    activation = activation, 
                    num_neg_resample = num_neg_resample)
    elif model_type == "join": # joint relative and global position encoding
        enc_dec = JointRelativeGlobalEncoderDecoder(pointset=pointset, 
                    enc = enc, 
                    spa_enc = spa_enc, 
                    g_spa_enc = g_spa_enc, 
                    g_spa_dec = g_spa_dec,
                    init_dec = init_dec, 
                    dec = dec, 
                    joint_dec = joint_dec, 
                    activation = activation, 
                    num_context_sample = num_context_sample, 
                    num_neg_resample = num_neg_resample)
    elif model_type == "together": # use global position of center point in context prediction
        enc_dec = GlobalPositionNeighGraphEncoderDecoder(pointset=pointset, 
                    enc = enc, 
                    spa_enc = spa_enc, 
                    g_spa_enc = g_spa_enc, 
                    init_dec = init_dec, 
                    dec = dec,
                    activation = activation, 
                    num_context_sample = num_context_sample, 
                    num_neg_resample = num_neg_resample)
    else:
        raise Exception("Unknow Model Type")

    return enc_dec

