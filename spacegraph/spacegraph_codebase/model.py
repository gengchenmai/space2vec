import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import os
from collections import OrderedDict, defaultdict
import random
import json

from spacegraph_codebase.module import get_activation_function
from spacegraph_codebase.data import PointSet, NeighborGraph, Point

class NeighGraphEncoderDecoder(nn.Module):
    """
    Combine the encoder, decoder and set up the training process
    """
    def __init__(self, pointset, enc, spa_enc, init_dec, dec, activation = "sigmoid", num_context_sample = 10, num_neg_resample = 10):
        super(NeighGraphEncoderDecoder, self).__init__()
        self.pointset = pointset
        self.enc = enc
        self.init_dec = init_dec
        self.dec = dec
        self.spa_enc = spa_enc
        self.num_context_sample = num_context_sample
        self.num_neg_resample = num_neg_resample # given 100 negative sample, we sample 10

        self.activation = get_activation_function(activation, "NeighGraphEncoderDecoder")
        

    def sample_context_pts(self, ng_list):
        for ng in ng_list:
            ng.sample_neighbor(self.num_context_sample)

    def sample_neg_pts(self, ng_list):
        for ng in ng_list:
            ng.sample_neg(self.num_neg_resample)

    def forward(self, ng_list, do_full_eval = True):
        '''
        Given a list of NeighborGraph(), 
            1. Compute the predicted center point feature embedding,
            2. Get the ground truth feature embedding for center points
            3. Get the N negative sampled center points feature embedding
        Args:
            do_full_eval: do we use the full negative samples to do evaluation
            ng_list: a list of NeighborGraph()
        Return:
            center_pred_embed: the predicted feature embedding for center points by using context points
                    shape (batch_size, embed_dim)
            center_embed: the ground truth feature embedding for center points
                    shape (batch_size, embed_dim)
            neg_embeds: the N negative sampled center points feature embedding
                    shape (batch_size, num_neg_sample, embed_dim)
        '''
        if do_full_eval == False:
            # random sample each context points in NeighborGraph()
            self.sample_neg_pts(ng_list)

        center_pred_embed = self.predict(ng_list)

        # center_embed: shape (batch_size, embed_dim)
        center_embed = self.get_center_pt_embed(ng_list)

        # neg_embeds: shape (batch_size, num_neg_sample, embed_dim)
        neg_embeds = self.get_neg_pt_embed(ng_list, do_full_eval)

        return center_pred_embed, center_embed, neg_embeds
    
    def predict(self, ng_list):
        # random sample each context points in NeighborGraph()
        self.sample_context_pts(ng_list)

        if self.spa_enc != None:
            # get all context-center point (deltaX, deltaY) list
            # coords: shape (batch_size, num_context_sample, 2)
            coords = self.get_spa_coords(ng_list)
            # key_spa_embeds: shape (batch_size, num_context_sample, spa_embed_dim)
            key_spa_embeds = self.spa_enc(coords)
        else:
            key_spa_embeds = torch.FloatTensor([])

        # get the feature embedding of the context points
        # key_embeds: shape (batch_size, num_context_sample, embed_dim)
        key_embeds = self.get_context_pt_embed(ng_list)

        # init_query_embed: shape (batch_size, embed_dim)
        init_query_embed = self.init_dec(key_embeds, key_spa_embeds, query_embed = None)

        if self.dec != None:
            # center_pred_embed: shape (batch_size, embed_dim)
            center_pred_embed = self.dec(key_embeds, key_spa_embeds, query_embed = init_query_embed)
        else:
            center_pred_embed = init_query_embed
        return center_pred_embed

    def get_batch_scores(self, ng_list, do_full_eval = True):
        '''
        Given a list of NeighborGraph(), 
        Return:
            pos: the dot product score between ground truth center point embedding and presented embedding
                (batch_size)
            neg: the dot product score between neg sampled center point embedding and presented embedding
                (batch_size, num_neg_sample)
        '''
        # center_pred_embed: shape (batch_size, embed_dim)
        # center_embed: shape (batch_size, embed_dim)
        # neg_embeds: shape (batch_size, num_neg_sample, embed_dim)
        center_pred_embed, center_embed, neg_embeds = self.forward(ng_list, do_full_eval)

        # positive score
        # pos: (batch_size)
        pos = torch.sum(center_embed * center_pred_embed, dim=1, keepdim=False)
        

        # negative sampling
        # center_pred_embed_: shape (batch_size, num_neg_sample, embed_dim)
        center_pred_embed_ = center_pred_embed.unsqueeze(1).expand_as(neg_embeds)
        # neg: (batch_size, num_neg_sample)
        neg = torch.sum(neg_embeds * center_pred_embed_, dim=2, keepdim=False)
        # neg: (batch_size)
   
        return pos, neg

    def softmax_loss(self, ng_list, do_full_eval = True):
        # pos: (batch_size)
        # neg: (batch_size, num_neg_sample)
        pos, neg = self.get_batch_scores(ng_list, do_full_eval)

        num_neg_sample = neg.size()[1]

        # pos: (batch_size)
        pos = torch.log(self.activation(pos))


        # neg: (batch_size)
        neg = torch.sum(torch.log(self.activation(-neg)), dim=1, keepdim=False)/num_neg_sample

        losses = -(pos + neg)
        loss = losses.mean()

        return loss




    def get_context_pt_embed(self, ng_list):
        '''
        Given a list of NeighborGraph(), get the feature embedding of the context points
        Return:
            key_embeds: shape (batch_size, num_context_sample, embed_dim)
        '''
        # pt_list: shape (batch_size*num_context_sample)
        pt_list = []
        for ng in ng_list:
            pt_list += list(ng.sample_context_pts)

        # key_embeds: shape (batch_size*num_context_sample, embed_dim)
        key_embeds = self.enc(pt_list)
        # key_embeds: shape (batch_size, num_context_sample, embed_dim)
        key_embeds = key_embeds.view(len(ng_list), self.num_context_sample, -1)
        return key_embeds

    def get_neg_pt_embed(self, ng_list, do_full_eval = True):
        '''
        Given a list of NeighborGraph(), get the feature embedding of the negative sampled center points
        Return:
            key_embeds: shape (batch_size, num_neg_sample, embed_dim)
        '''
        if do_full_eval == True:
            num_neg_sample = len(ng_list[0].neg_samples)
            # pt_list: shape (batch_size*num_neg_sample)
            pt_list = []
            for ng in ng_list:
                pt_list += list(ng.neg_samples)

            # key_embeds: shape (batch_size*num_neg_sample, embed_dim)
            key_embeds = self.enc(pt_list)
            # key_embeds: shape (batch_size, num_neg_sample, embed_dim)
            key_embeds = key_embeds.view(len(ng_list), num_neg_sample, -1)
        else:
            # pt_list: shape (batch_size*num_neg_resample)
            pt_list = []
            for ng in ng_list:
                pt_list += list(ng.sample_neg_pts)

            # key_embeds: shape (batch_size*num_neg_resample, embed_dim)
            key_embeds = self.enc(pt_list)
            # key_embeds: shape (batch_size, num_neg_resample, embed_dim)
            key_embeds = key_embeds.view(len(ng_list), self.num_neg_resample, -1)
        return key_embeds

    def get_center_pt_embed(self, ng_list):
        '''
        Given a list of NeighborGraph(), get the feature embedding of the center points
        Return:
            query_embed: shape (batch_size, embed_dim)
        '''
        pt_list = [ng.center_pt for ng in ng_list]

        # query_embed: shape (batch_size, embed_dim)
        query_embed = self.enc(pt_list)
        return query_embed

    
    def get_spa_coords(self, ng_list):
        '''
        Given a list of NeighborGraph(), get their (deltaX, deltaY) list
        '''
        coords = []
        for ng in ng_list:
            cur_coords = []
            center_coord = self.pointset.pt_dict[ng.center_pt].coord
            for i in range(len(ng.sample_context_pts)):
                coord = self.pointset.pt_dict[ng.sample_context_pts[i]].coord
                cur_coords.append([coord[0]-center_coord[0], coord[1]-center_coord[1]])
            coords.append(cur_coords)
        # coords: shape (batch_size, num_context_sample, 2)
        return coords


class GlobalPositionNeighGraphEncoderDecoder(nn.Module):
    """
    add the global position embedding of the center point in the decoder
    Combine the encoder, decoder and set up the training process
    """
    def __init__(self, pointset, enc, spa_enc, g_spa_enc, init_dec, dec, activation = "sigmoid", num_context_sample = 10, num_neg_resample = 10):
        super(GlobalPositionNeighGraphEncoderDecoder, self).__init__()
        self.pointset = pointset
        self.enc = enc
        self.init_dec = init_dec
        self.dec = dec
        self.spa_enc = spa_enc
        self.g_spa_enc = g_spa_enc
        self.num_context_sample = num_context_sample
        self.num_neg_resample = num_neg_resample # given 100 negative sample, we sample 10

        self.activation = get_activation_function(activation, "GlobalPositionNeighGraphEncoderDecoder")
        

        

    def sample_context_pts(self, ng_list):
        for ng in ng_list:
            ng.sample_neighbor(self.num_context_sample)

    def sample_neg_pts(self, ng_list):
        for ng in ng_list:
            ng.sample_neg(self.num_neg_resample)

    def forward(self, ng_list, do_full_eval = True):
        '''
        Given a list of NeighborGraph(), 
            1. Compute the predicted center point feature embedding,
            2. Get the ground truth feature embedding for center points
            3. Get the N negative sampled center points feature embedding
        Args:
            do_full_eval: do we use the full negative samples to do evaluation
            ng_list: a list of NeighborGraph()
        Return:
            center_pred_embed: the predicted feature embedding for center points by using context points
                    shape (batch_size, embed_dim)
            center_embed: the ground truth feature embedding for center points
                    shape (batch_size, embed_dim)
            neg_embeds: the N negative sampled center points feature embedding
                    shape (batch_size, num_neg_sample, embed_dim)
        '''
        if do_full_eval == False:
            # random sample each context points in NeighborGraph()
            self.sample_neg_pts(ng_list)

        center_pred_embed = self.predict(ng_list)

        # 2. get the true center embedding
        # center_embed: shape (batch_size, embed_dim)
        center_embed = self.get_center_pt_embed(ng_list)

        # 3. get the true negative embedding
        # neg_embeds: shape (batch_size, num_neg_sample, embed_dim)
        neg_embeds = self.get_neg_pt_embed(ng_list, do_full_eval)

        return center_pred_embed, center_embed, neg_embeds

    def predict(self, ng_list):
        # random sample each context points in NeighborGraph()
        self.sample_context_pts(ng_list)

        # 1. predict the center pt feature embedding from context points
        if self.spa_enc != None:
            # get all context-center point (deltaX, deltaY) list
            # coords: shape (batch_size, num_context_sample, 2)
            coords = self.get_spa_coords(ng_list)
            # key_spa_embeds: shape (batch_size, num_context_sample, spa_embed_dim)
            key_spa_embeds = self.spa_enc(coords)
        else:
            key_spa_embeds = torch.FloatTensor([])

        # 1.1 get the feature embedding of the context points
        # key_embeds: shape (batch_size, num_context_sample, embed_dim)
        key_embeds = self.get_context_pt_embed(ng_list)

        # 1.2 get center pt position embedding
        if self.g_spa_enc != None:
            # coords: shape (batch_size, 1, 2)
            coords = self.get_center_pt_spa_coords(ng_list)
            # center_g_spa_embeds: shape (batch_size, 1, g_spa_embed_dim)
            center_g_spa_embeds = self.g_spa_enc(coords)
            # center_g_spa_embeds: shape (batch_size, g_spa_embed_dim)
            center_g_spa_embeds = center_g_spa_embeds.squeeze(1)
        else:
            center_g_spa_embeds = torch.FloatTensor([])

        # init_query_embed: shape (batch_size, embed_dim)
        init_query_embed = self.init_dec(key_embeds, key_spa_embeds, center_g_spa_embeds, query_embed = None)

        if self.dec != None:
            # center_pred_embed: shape (batch_size, embed_dim)
            center_pred_embed = self.dec(key_embeds, key_spa_embeds, center_g_spa_embeds, query_embed = init_query_embed)
        else:
            center_pred_embed = init_query_embed
        return center_pred_embed

    def get_batch_scores(self, ng_list, do_full_eval = True):
        '''
        Given a list of NeighborGraph(), 
        Return:
            pos: the dot product score between ground truth center point embedding and presented embedding
                (batch_size)
            neg: the dot product score between neg sampled center point embedding and presented embedding
                (batch_size, num_neg_sample)
        '''
        # center_pred_embed: shape (batch_size, embed_dim)
        # center_embed: shape (batch_size, embed_dim)
        # neg_embeds: shape (batch_size, num_neg_sample, embed_dim)
        center_pred_embed, center_embed, neg_embeds = self.forward(ng_list, do_full_eval)

        # positive score
        # pos: (batch_size)
        pos = torch.sum(center_embed * center_pred_embed, dim=1, keepdim=False)
        

        # negative sampling
        # center_pred_embed_: shape (batch_size, num_neg_sample, embed_dim)
        center_pred_embed_ = center_pred_embed.unsqueeze(1).expand_as(neg_embeds)
        # neg: (batch_size, num_neg_sample)
        neg = torch.sum(neg_embeds * center_pred_embed_, dim=2, keepdim=False)
        # neg: (batch_size)
   
        return pos, neg

    def softmax_loss(self, ng_list, do_full_eval = True):
        # pos: (batch_size)
        # neg: (batch_size, num_neg_sample)
        pos, neg = self.get_batch_scores(ng_list, do_full_eval)

        num_neg_sample = neg.size()[1]

        # pos: (batch_size)
        pos = torch.log(self.activation(pos))


        # neg: (batch_size)
        neg = torch.sum(torch.log(self.activation(-neg)), dim=1, keepdim=False)/num_neg_sample

        losses = -(pos + neg)
        loss = losses.mean()

        return loss




    def get_context_pt_embed(self, ng_list):
        '''
        Given a list of NeighborGraph(), get the feature embedding of the context points
        Return:
            key_embeds: shape (batch_size, num_context_sample, embed_dim)
        '''
        # pt_list: shape (batch_size*num_context_sample)
        pt_list = []
        for ng in ng_list:
            pt_list += list(ng.sample_context_pts)

        # key_embeds: shape (batch_size*num_context_sample, embed_dim)
        key_embeds = self.enc(pt_list)
        # key_embeds: shape (batch_size, num_context_sample, embed_dim)
        key_embeds = key_embeds.view(len(ng_list), self.num_context_sample, -1)
        return key_embeds

    def get_neg_pt_embed(self, ng_list, do_full_eval = True):
        '''
        Given a list of NeighborGraph(), get the feature embedding of the negative sampled center points
        Return:
            key_embeds: shape (batch_size, num_neg_sample, embed_dim)
        '''
        if do_full_eval == True:
            num_neg_sample = len(ng_list[0].neg_samples)
            # pt_list: shape (batch_size*num_neg_sample)
            pt_list = []
            for ng in ng_list:
                pt_list += list(ng.neg_samples)

            # key_embeds: shape (batch_size*num_neg_sample, embed_dim)
            key_embeds = self.enc(pt_list)
            # key_embeds: shape (batch_size, num_neg_sample, embed_dim)
            key_embeds = key_embeds.view(len(ng_list), num_neg_sample, -1)
        else:
            # pt_list: shape (batch_size*num_neg_resample)
            pt_list = []
            for ng in ng_list:
                pt_list += list(ng.sample_neg_pts)

            # key_embeds: shape (batch_size*num_neg_resample, embed_dim)
            key_embeds = self.enc(pt_list)
            # key_embeds: shape (batch_size, num_neg_resample, embed_dim)
            key_embeds = key_embeds.view(len(ng_list), self.num_neg_resample, -1)
        return key_embeds

    def get_center_pt_embed(self, ng_list):
        '''
        Given a list of NeighborGraph(), get the feature embedding of the center points
        Return:
            query_embed: shape (batch_size, embed_dim)
        '''
        pt_list = [ng.center_pt for ng in ng_list]

        # query_embed: shape (batch_size, embed_dim)
        query_embed = self.enc(pt_list)
        return query_embed

    
    def get_spa_coords(self, ng_list):
        '''
        Given a list of NeighborGraph(), get their (deltaX, deltaY) list
        '''
        coords = []
        for ng in ng_list:
            cur_coords = []
            center_coord = self.pointset.pt_dict[ng.center_pt].coord
            for i in range(len(ng.sample_context_pts)):
                coord = self.pointset.pt_dict[ng.sample_context_pts[i]].coord
                cur_coords.append([coord[0]-center_coord[0], coord[1]-center_coord[1]])
            coords.append(cur_coords)
        # coords: shape (batch_size, num_context_sample, 2)
        return coords

    def get_center_pt_spa_coords(self, ng_list):
        '''
        Given a list of NeighborGraph(), get their center point (X, Y) list
        '''
        coords = []
        for ng in ng_list:
            cur_coords = []
            center_coord = self.pointset.pt_dict[ng.center_pt].coord
            cur_coords.append(center_coord)
            coords.append(cur_coords)
        # coords: shape (batch_size, 1, 2)
        return coords

    def freeze_param_except_join_dec(self):
        # freeze all parameter except the parameters of join_dec
        
        self.freeze_param(self.enc)
        self.freeze_param(self.init_dec)
        # self.freeze_param(self.joint_dec)
        self.freeze_param(self.spa_enc)
        self.freeze_param(self.g_spa_enc)
        self.freeze_param(self.g_spa_dec)

    def freeze_param(self, module):
        for param in module.parameters():
            param.requires_grad = False

class GlobalPositionEncoderDecoder(nn.Module):
    """
    encode the position of point and directly decode to its feature embedding
    """
    def __init__(self, pointset, enc, g_spa_enc, g_spa_dec, activation = "sigmoid", num_neg_resample = 10):
        super(GlobalPositionEncoderDecoder, self).__init__()
        self.pointset = pointset
        self.enc = enc             # point feature embedding encoder
        self.g_spa_enc = g_spa_enc # one of the SpatialRelationEncoder
        self.g_spa_dec = g_spa_dec # DirectPositionEmbeddingDecoder()

        self.activation = get_activation_function(activation, "GlobalPositionEncoderDecoder")
        

        self.num_neg_resample = num_neg_resample # given 100 negative sample, we sample 10


    def sample_neg_pts(self, ng_list):
        for ng in ng_list:
            ng.sample_neg(self.num_neg_resample)

    def forward(self, ng_list, do_full_eval = True):
        '''
        Given a list of NeighborGraph(), 
            1. Compute the predicted center point feature embedding,
            2. Get the ground truth feature embedding for center points
            3. Get the N negative sampled center points feature embedding
        Args:
            do_full_eval: do we use the full negative samples to do evaluation
            ng_list: a list of NeighborGraph()
        Return:
            center_pred_embed: the predicted feature embedding for center points by using context points
                    shape (batch_size, embed_dim)
            center_embed: the ground truth feature embedding for center points
                    shape (batch_size, embed_dim)
            neg_embeds: the N negative sampled center points feature embedding
                    shape (batch_size, num_neg_sample, embed_dim)
        '''

        if do_full_eval == False:
            # random sample each context points in NeighborGraph()
            self.sample_neg_pts(ng_list)

        # coords: shape (batch_size, 1, 2)
        # coords = self.get_center_pt_spa_coords(ng_list)
        # # center_g_spa_embeds: shape (batch_size, 1, g_spa_embed_dim)
        # center_g_spa_embeds = self.g_spa_enc(coords)
        # # center_g_spa_embeds: shape (batch_size, g_spa_embed_dim)
        # center_g_spa_embeds = center_g_spa_embeds.squeeze(1)
        # # center_pred_embed: shape (batch_size, embed_dim)
        # center_pred_embed = self.g_spa_dec(center_g_spa_embeds)
        center_pred_embed = self.predict(ng_list)

        # center_embed: shape (batch_size, embed_dim)
        center_embed = self.get_center_pt_embed(ng_list)

        # neg_embeds: shape (batch_size, num_neg_sample, embed_dim)
        neg_embeds = self.get_neg_pt_embed(ng_list, do_full_eval)

        return center_pred_embed, center_embed, neg_embeds

    def predict(self, ng_list):
        coords = self.get_center_pt_spa_coords(ng_list)
        # center_g_spa_embeds: shape (batch_size, 1, g_spa_embed_dim)
        center_g_spa_embeds = self.g_spa_enc(coords)
        # center_g_spa_embeds: shape (batch_size, g_spa_embed_dim)
        center_g_spa_embeds = center_g_spa_embeds.squeeze(1)
        # center_pred_embed: shape (batch_size, embed_dim)
        center_pred_embed = self.g_spa_dec(center_g_spa_embeds)
        return center_pred_embed

    def get_pred_embed_from_coords(self, coords):
        '''
        Args:
            coords: a list of coordinates, (y_batch_size, x_batch_size, 2)
        '''
        # center_g_spa_embeds: shape (y_batch_size, x_batch_size, g_spa_embed_dim)
        center_g_spa_embeds = self.g_spa_enc(coords)
        y_batch_size, x_batch_size, g_spa_embed_dim = center_g_spa_embeds.size()
        # center_g_spa_embeds: shape (y_batch_size * x_batch_size, g_spa_embed_dim)
        center_g_spa_embeds = center_g_spa_embeds.view(y_batch_size * x_batch_size, g_spa_embed_dim)
        # center_pred_embed: shape (y_batch_size * x_batch_size, embed_dim)
        center_pred_embed = self.g_spa_dec(center_g_spa_embeds)
        return center_pred_embed


    def get_batch_scores(self, ng_list, do_full_eval = True):
        '''
        Given a list of NeighborGraph(), 
        Return:
            pos: the dot product score between ground truth center point embedding and presented embedding
                (batch_size)
            neg: the dot product score between neg sampled center point embedding and presented embedding
                (batch_size, num_neg_sample)
        '''
        # center_pred_embed: shape (batch_size, embed_dim)
        # center_embed: shape (batch_size, embed_dim)
        # neg_embeds: shape (batch_size, num_neg_sample, embed_dim)
        center_pred_embed, center_embed, neg_embeds = self.forward(ng_list, do_full_eval)

        # positive score
        # pos: (batch_size)
        pos = torch.sum(center_embed * center_pred_embed, dim=1, keepdim=False)
        

        # negative sampling
        # center_pred_embed_: shape (batch_size, num_neg_sample, embed_dim)
        center_pred_embed_ = center_pred_embed.unsqueeze(1).expand_as(neg_embeds)
        # neg: (batch_size, num_neg_sample)
        neg = torch.sum(neg_embeds * center_pred_embed_, dim=2, keepdim=False)
        # neg: (batch_size)
   
        return pos, neg

    def softmax_loss(self, ng_list, do_full_eval = True):
        # pos: (batch_size)
        # neg: (batch_size, num_neg_sample)
        pos, neg = self.get_batch_scores(ng_list, do_full_eval)

        num_neg_sample = neg.size()[1]

        # pos: (batch_size)
        pos = torch.log(self.activation(pos))


        # neg: (batch_size)
        neg = torch.sum(torch.log(self.activation(-neg)), dim=1, keepdim=False)/num_neg_sample

        losses = -(pos + neg)
        loss = losses.mean()

        return loss

    def get_center_pt_spa_coords(self, ng_list):
        '''
        Given a list of NeighborGraph(), get their center point (X, Y) list
        '''
        coords = []
        for ng in ng_list:
            cur_coords = []
            center_coord = self.pointset.pt_dict[ng.center_pt].coord
            cur_coords.append(center_coord)
            coords.append(cur_coords)
        # coords: shape (batch_size, 1, 2)
        return coords

    def get_neg_pt_embed(self, ng_list, do_full_eval = True):
        '''
        Given a list of NeighborGraph(), get the feature embedding of the negative sampled center points
        Return:
            key_embeds: shape (batch_size, num_neg_sample, embed_dim)
        '''
        if do_full_eval == True:
            num_neg_sample = len(ng_list[0].neg_samples)
            # pt_list: shape (batch_size*num_neg_sample)
            pt_list = []
            for ng in ng_list:
                pt_list += list(ng.neg_samples)

            # key_embeds: shape (batch_size*num_neg_sample, embed_dim)
            key_embeds = self.enc(pt_list)
            # key_embeds: shape (batch_size, num_neg_sample, embed_dim)
            key_embeds = key_embeds.view(len(ng_list), num_neg_sample, -1)
        else:
            # pt_list: shape (batch_size*num_neg_resample)
            pt_list = []
            for ng in ng_list:
                pt_list += list(ng.sample_neg_pts)

            # key_embeds: shape (batch_size*num_neg_resample, embed_dim)
            key_embeds = self.enc(pt_list)
            # key_embeds: shape (batch_size, num_neg_resample, embed_dim)
            key_embeds = key_embeds.view(len(ng_list), self.num_neg_resample, -1)
        return key_embeds

    def get_center_pt_embed(self, ng_list):
        '''
        Given a list of NeighborGraph(), get the feature embedding of the center points
        Return:
            query_embed: shape (batch_size, embed_dim)
        '''
        pt_list = [ng.center_pt for ng in ng_list]

        # query_embed: shape (batch_size, embed_dim)
        query_embed = self.enc(pt_list)
        return query_embed



class JointRelativeGlobalEncoderDecoder(nn.Module):
    """
    Combine the encoder, decoder and set up the training process
    """
    def __init__(self, pointset, enc, spa_enc, g_spa_enc, g_spa_dec, init_dec, dec, joint_dec, activation = "sigmoid", num_context_sample = 10, num_neg_resample = 10):
        super(JointRelativeGlobalEncoderDecoder, self).__init__()
        self.pointset = pointset
        self.enc = enc
        self.init_dec = init_dec
        self.dec = dec
        self.joint_dec = joint_dec
        self.spa_enc = spa_enc
        self.g_spa_enc = g_spa_enc
        self.g_spa_dec = g_spa_dec
        self.num_context_sample = num_context_sample
        self.num_neg_resample = num_neg_resample # given 100 negative sample, we sample 10

        self.activation = get_activation_function(activation, "JointRelativeGlobalEncoderDecoder")
        

        

    def sample_context_pts(self, ng_list):
        for ng in ng_list:
            ng.sample_neighbor(self.num_context_sample)

    def sample_neg_pts(self, ng_list):
        for ng in ng_list:
            ng.sample_neg(self.num_neg_resample)

    def forward(self, ng_list, do_full_eval = True):
        '''
        Given a list of NeighborGraph(), 
            1. Compute the predicted center point feature embedding,
            2. Get the ground truth feature embedding for center points
            3. Get the N negative sampled center points feature embedding
        Args:
            do_full_eval: do we use the full negative samples to do evaluation
            ng_list: a list of NeighborGraph()
        Return:
            center_pred_embed: the predicted feature embedding for center points by using context points
                    shape (batch_size, embed_dim)
            center_embed: the ground truth feature embedding for center points
                    shape (batch_size, embed_dim)
            neg_embeds: the N negative sampled center points feature embedding
                    shape (batch_size, num_neg_sample, embed_dim)
        '''
        if do_full_eval == False:
            # random sample each context points in NeighborGraph()
            self.sample_neg_pts(ng_list)

        center_pred_embed = self.predict(ng_list)

        # 4. get the true center embedding
        # center_embed: shape (batch_size, embed_dim)
        center_embed = self.get_center_pt_embed(ng_list)

        # 5. get the true negative embedding
        # neg_embeds: shape (batch_size, num_neg_sample, embed_dim)
        neg_embeds = self.get_neg_pt_embed(ng_list, do_full_eval)

        return center_pred_embed, center_embed, neg_embeds

    def predict(self, ng_list):
        # random sample each context points in NeighborGraph()
        self.sample_context_pts(ng_list)

        # 1. predict the center pt feature embedding from context points
        if self.spa_enc != None:
            # get all context-center point (deltaX, deltaY) list
            # coords: shape (batch_size, num_context_sample, 2)
            coords = self.get_spa_coords(ng_list)
            # key_spa_embeds: shape (batch_size, num_context_sample, spa_embed_dim)
            key_spa_embeds = self.spa_enc(coords)
        else:
            key_spa_embeds = torch.FloatTensor([])

        # get the feature embedding of the context points
        # key_embeds: shape (batch_size, num_context_sample, embed_dim)
        key_embeds = self.get_context_pt_embed(ng_list)

        # init_query_embed: shape (batch_size, embed_dim)
        init_query_embed = self.init_dec(key_embeds, key_spa_embeds, query_embed = None)

        if self.dec != None:
            # center_pred_embed_1: shape (batch_size, embed_dim)
            center_pred_embed_1 = self.dec(key_embeds, key_spa_embeds, query_embed = init_query_embed)
        else:
            center_pred_embed_1 = init_query_embed

        # 2. predict center feature embedding from point location
        # coords: shape (batch_size, 1, 2)
        coords = self.get_center_pt_spa_coords(ng_list)
        # center_g_spa_embeds: shape (batch_size, 1, g_spa_embed_dim)
        center_g_spa_embeds = self.g_spa_enc(coords)
        # center_g_spa_embeds: shape (batch_size, g_spa_embed_dim)
        center_g_spa_embeds = center_g_spa_embeds.squeeze(1)
        # center_pred_embed_2: shape (batch_size, embed_dim)
        center_pred_embed_2 = self.g_spa_dec(center_g_spa_embeds)


        # 3. Given the predict center embedding from context, and the center position embedding
        # predict final feature embedding
        center_pred_embed = self.joint_dec(center_pred_embed_1, center_pred_embed_2)
        return center_pred_embed

    def get_batch_scores(self, ng_list, do_full_eval = True):
        '''
        Given a list of NeighborGraph(), 
        Return:
            pos: the dot product score between ground truth center point embedding and presented embedding
                (batch_size)
            neg: the dot product score between neg sampled center point embedding and presented embedding
                (batch_size, num_neg_sample)
        '''
        # center_pred_embed: shape (batch_size, embed_dim)
        # center_embed: shape (batch_size, embed_dim)
        # neg_embeds: shape (batch_size, num_neg_sample, embed_dim)
        center_pred_embed, center_embed, neg_embeds = self.forward(ng_list, do_full_eval)

        # positive score
        # pos: (batch_size)
        pos = torch.sum(center_embed * center_pred_embed, dim=1, keepdim=False)
        

        # negative sampling
        # center_pred_embed_: shape (batch_size, num_neg_sample, embed_dim)
        center_pred_embed_ = center_pred_embed.unsqueeze(1).expand_as(neg_embeds)
        # neg: (batch_size, num_neg_sample)
        neg = torch.sum(neg_embeds * center_pred_embed_, dim=2, keepdim=False)
        # neg: (batch_size)
   
        return pos, neg

    def softmax_loss(self, ng_list, do_full_eval = True):
        # pos: (batch_size)
        # neg: (batch_size, num_neg_sample)
        pos, neg = self.get_batch_scores(ng_list, do_full_eval)

        num_neg_sample = neg.size()[1]

        # pos: (batch_size)
        pos = torch.log(self.activation(pos))


        # neg: (batch_size)
        neg = torch.sum(torch.log(self.activation(-neg)), dim=1, keepdim=False)/num_neg_sample

        losses = -(pos + neg)
        loss = losses.mean()

        return loss




    def get_context_pt_embed(self, ng_list):
        '''
        Given a list of NeighborGraph(), get the feature embedding of the context points
        Return:
            key_embeds: shape (batch_size, num_context_sample, embed_dim)
        '''
        # pt_list: shape (batch_size*num_context_sample)
        pt_list = []
        for ng in ng_list:
            pt_list += list(ng.sample_context_pts)

        # key_embeds: shape (batch_size*num_context_sample, embed_dim)
        key_embeds = self.enc(pt_list)
        # key_embeds: shape (batch_size, num_context_sample, embed_dim)
        key_embeds = key_embeds.view(len(ng_list), self.num_context_sample, -1)
        return key_embeds

    def get_neg_pt_embed(self, ng_list, do_full_eval = True):
        '''
        Given a list of NeighborGraph(), get the feature embedding of the negative sampled center points
        Return:
            key_embeds: shape (batch_size, num_neg_sample, embed_dim)
        '''
        if do_full_eval == True:
            num_neg_sample = len(ng_list[0].neg_samples)
            # pt_list: shape (batch_size*num_neg_sample)
            pt_list = []
            for ng in ng_list:
                pt_list += list(ng.neg_samples)

            # key_embeds: shape (batch_size*num_neg_sample, embed_dim)
            key_embeds = self.enc(pt_list)
            # key_embeds: shape (batch_size, num_neg_sample, embed_dim)
            key_embeds = key_embeds.view(len(ng_list), num_neg_sample, -1)
        else:
            # pt_list: shape (batch_size*num_neg_resample)
            pt_list = []
            for ng in ng_list:
                pt_list += list(ng.sample_neg_pts)

            # key_embeds: shape (batch_size*num_neg_resample, embed_dim)
            key_embeds = self.enc(pt_list)
            # key_embeds: shape (batch_size, num_neg_resample, embed_dim)
            key_embeds = key_embeds.view(len(ng_list), self.num_neg_resample, -1)
        return key_embeds

    def get_center_pt_embed(self, ng_list):
        '''
        Given a list of NeighborGraph(), get the feature embedding of the center points
        Return:
            query_embed: shape (batch_size, embed_dim)
        '''
        pt_list = [ng.center_pt for ng in ng_list]

        # query_embed: shape (batch_size, embed_dim)
        query_embed = self.enc(pt_list)
        return query_embed

    
    def get_spa_coords(self, ng_list):
        '''
        Given a list of NeighborGraph(), get their (deltaX, deltaY) list
        '''
        coords = []
        for ng in ng_list:
            cur_coords = []
            center_coord = self.pointset.pt_dict[ng.center_pt].coord
            for i in range(len(ng.sample_context_pts)):
                coord = self.pointset.pt_dict[ng.sample_context_pts[i]].coord
                cur_coords.append([coord[0]-center_coord[0], coord[1]-center_coord[1]])
            coords.append(cur_coords)
        # coords: shape (batch_size, num_context_sample, 2)
        return coords

    def get_center_pt_spa_coords(self, ng_list):
        '''
        Given a list of NeighborGraph(), get their center point (X, Y) list
        '''
        coords = []
        for ng in ng_list:
            cur_coords = []
            center_coord = self.pointset.pt_dict[ng.center_pt].coord
            cur_coords.append(center_coord)
            coords.append(cur_coords)
        # coords: shape (batch_size, 1, 2)
        return coords

    def freeze_param_except_join_dec(self):
        # freeze all parameter except the parameters of join_dec
        
        self.freeze_param(self.enc)
        self.freeze_param(self.init_dec)
        # self.freeze_param(self.joint_dec)
        self.freeze_param(self.spa_enc)
        self.freeze_param(self.g_spa_enc)
        self.freeze_param(self.g_spa_dec)

    def freeze_param(self, module):
        for param in module.parameters():
            param.requires_grad = False