import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np
import torch.nn.functional as F

from spacegraph_codebase.module import get_activation_function
from spacegraph_codebase.module import LayerNorm


'''
The Decoder of PointSet, Use N context Point to predict the center point
'''
class IntersectConcatAttention(nn.Module):
    def __init__(self, query_dim, key_dim, spa_embed_dim, have_query_embed = True, num_attn = 1, 
        activation = "leakyrelu", f_activation = "sigmoid", 
        layernorm = False, use_post_mat = False, dropout = 0.5):
        '''
        The attention method used by Graph Attention network (LeakyReLU)
        Args:
            query_dim: the center point feature embedding dimention
            key_dim: the N context point feature embedding dimention
            spa_embed_dim: the spatial relation embedding dimention
            have_query_embed: Trua/False, do we use query embedding in the attention
            num_attn: number of attention head
            activation: the activation function to atten_vecs * torch.cat(query_embed, key_embed), see GAT paper Equ 3
            f_activation: the final activation function applied to get the final result, see GAT paper Equ 6
        ''' 
        super(IntersectConcatAttention, self).__init__()
        

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.spa_embed_dim = spa_embed_dim
        self.num_attn = num_attn
        self.have_query_embed = have_query_embed

        self.activation = get_activation_function(activation, "IntersectConcatAttention middle")

        self.f_activation = get_activation_function(f_activation, "IntersectConcatAttention final")
        
        self.softmax = nn.Softmax(dim=1)


        self.layernorm = layernorm
        self.use_post_mat = use_post_mat
        if self.have_query_embed:
            assert key_dim == query_dim


            # define the layer normalization
            if self.layernorm:
                self.pre_ln = LayerNorm(query_dim)
                self.add_module("attn_preln", self.pre_ln)

            if self.use_post_mat:
                self.post_linear = nn.Linear(query_dim, query_dim)
                self.dropout = nn.Dropout(p=dropout)
                # self.register_parameter("attn_PostLinear", self.post_linear)

                # self.post_W = nn.Parameter(torch.FloatTensor(query_dim, query_dim))
                # init.xavier_uniform_(self.post_W)
                # self.register_parameter("attn_PostW", self.post_W)

                # self.post_B = nn.Parameter(torch.FloatTensor(1,query_dim))
                # init.xavier_uniform_(self.post_B)
                # self.register_parameter("attn_PostB",self.post_B)
                if self.layernorm:
                    self.post_ln = LayerNorm(query_dim)
                    self.add_module("attn_Postln", self.post_ln)
      
            # each column represent an attention vector for one attention head: [embed_dim*2, num_attn]
            self.atten_vecs = nn.Parameter(torch.FloatTensor(query_dim+key_dim+spa_embed_dim, self.num_attn))
            init.xavier_uniform_(self.atten_vecs)
            self.register_parameter("attn_attenvecs", self.atten_vecs)
        else:
            # if we do not use query embedding in the attention, this means
            # We just compute the initial query embedding 
            # define the layer normalization
            if self.layernorm:
                self.pre_ln = LayerNorm(key_dim)
                self.add_module("attn_nq_preln", self.pre_ln)

            if self.use_post_mat:
                self.post_linear = nn.Linear(key_dim, key_dim)
                self.dropout = nn.Dropout(p=dropout)
                # self.register_parameter("attn_PostLinear", self.post_linear)

                # self.post_W = nn.Parameter(torch.FloatTensor(key_dim, key_dim))
                # init.xavier_uniform_(self.post_W)
                # self.register_parameter("attn_nq_PostW", self.post_W)

                # self.post_B = nn.Parameter(torch.FloatTensor(1,key_dim))
                # init.xavier_uniform_(self.post_B)
                # self.register_parameter("attn_nq_PostB",self.post_B)
                if self.layernorm:
                    self.post_ln = LayerNorm(key_dim)
                    self.add_module("attn_nq_Postln", self.post_ln)
            
            # In the initial query embedding computing, we just use key embeddings and spatial relation embeddings
            # each column represent an attention vector for one attention head: [embed_dim*2, num_attn]
            self.atten_vecs = nn.Parameter(torch.FloatTensor(key_dim+spa_embed_dim, self.num_attn))
            init.xavier_uniform_(self.atten_vecs)
            self.register_parameter("attn_nq_attenvecs", self.atten_vecs)
        




    def forward(self, key_embeds, key_spa_embeds, query_embed = None):
        '''
        Args:
            
            key_embeds: a list of feature embeddings computed from different context point, 
                        [batch_size, num_context_pt, key_dim]
            key_spa_embeds: a list of spatial relation embeddings computed from differnet context point, 
                        [batch_size, num_context_pt, spa_embed_dim]
            query_embed: the pre-computed variable embeddings, 
                        [batch_size, query_dim]
                        have_query_embed (True):
                        have_query_embed (False): None
        Return:
            combined: the multi-head attention based embeddings for center pt [batch_size, key_dim]
        '''
        tensor_size = key_embeds.size()
        num_context_pt = tensor_size[1]
        batch_size = tensor_size[0]
        if key_spa_embeds.size()[0] > 0: # we have key_spa_embeds
            assert num_context_pt == key_spa_embeds.size()[1]
            assert batch_size == key_spa_embeds.size()[0]

        if self.have_query_embed:
            # We use the pre-computed query_embed to do attention
            # assert query_embed != None
            assert batch_size == query_embed.size()[0] 
            query_dim = query_embed.size()[1]
            assert query_dim == self.query_dim == self.key_dim
            # query_embed_expand: [batch_size, num_context_pt, query_dim]
            query_embed_expand = query_embed.unsqueeze(1).expand(batch_size, num_context_pt, query_dim)
            if key_spa_embeds.size()[0] > 0: # we have key_spa_embeds 
                # concat: [batch_size, num_context_pt, query_dim+key_dim+spa_embed_dim]
                concat = torch.cat((query_embed_expand, key_embeds, key_spa_embeds), dim=2)
            else:
                # concat: [batch_size, num_context_pt, query_dim+key_dim]
                concat = torch.cat((query_embed_expand, key_embeds), dim=2)
        else:
            # We just use the context feature embedding (key_embeds) and spatial embedding to compute initial center point embedding
            assert query_embed == None

            if key_spa_embeds.size()[0] > 0: # we have key_spa_embeds 
                # concat: [batch_size, num_context_pt, key_dim+spa_embed_dim]
                concat = torch.cat((key_embeds, key_spa_embeds), dim=2)
            else:
                concat = key_embeds
        
        # 1. compute the attention score for each key embeddings
        # attn: [batch_size, num_context_pt, num_attn]
        attn = torch.einsum("bnd,dk->bnk", (concat, self.atten_vecs))
        # attn: [batch_size, num_context_pt, num_attn]
        attn = self.softmax(self.activation(attn))
        # attn: [batch_size, num_attn, num_context_pt]
        attn = attn.transpose(1,2)
        # 2. using the attention score to compute the weighted average of the key embeddings
        # combined: [batch_size, num_attn, key_dim]
        combined = torch.einsum("bkn,bnd->bkd", (attn, key_embeds))
        # combined: [batch_size, key_dim]
        combined = torch.sum(combined, dim=1,keepdim=False) * (1.0/self.num_attn)
        # combined: [batch_size, key_dim]
        combined =  self.f_activation(combined)

        # Note that query_dim == key_dim
        if self.layernorm:
            if self.have_query_embed: 
                # residual connection
                combined = combined + query_embed
            combined = self.pre_ln(combined)

        if self.use_post_mat:
            # linear: [batch_size, query_dim]
            # linear = combined.mm(self.post_W) + self.post_B
            linear = self.dropout(self.post_linear(combined))
            if self.layernorm:
                linear = linear + combined
                linear = self.post_ln(linear)
            return linear

        
        return combined

'''
The Decoder of PointSet, Use N context Point to predict the center point
Include the global position encoding in the decoder
'''
class GolbalPositionIntersectConcatAttention(nn.Module):
    def __init__(self, query_dim, key_dim, spa_embed_dim, g_spa_embed_dim, have_query_embed = True, num_attn = 1, 
        activation = "leakyrelu", f_activation = "sigmoid", 
        layernorm = False, use_post_mat = False, dropout = 0.5):
        '''
        The attention method used by Graph Attention network (LeakyReLU)
        Args:
            query_dim: the center point feature embedding dimention
            key_dim: the N context point feature embedding dimention
            spa_embed_dim: the spatial relation embedding dimention
            have_query_embed: Trua/False, do we use query embedding in the attention
            num_attn: number of attention head
            activation: the activation function to atten_vecs * torch.cat(query_embed, key_embed), see GAT paper Equ 3
            f_activation: the final activation function applied to get the final result, see GAT paper Equ 6
        ''' 
        super(GolbalPositionIntersectConcatAttention, self).__init__()
        

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.spa_embed_dim = spa_embed_dim

        self.g_spa_embed_dim = g_spa_embed_dim

        self.num_attn = num_attn
        self.have_query_embed = have_query_embed

        self.activation = get_activation_function(activation, "GolbalPositionIntersectConcatAttention middle")

        self.f_activation = get_activation_function(f_activation, "GolbalPositionIntersectConcatAttention final")
        
        self.softmax = nn.Softmax(dim=1)


        self.layernorm = layernorm
        self.use_post_mat = use_post_mat
        if self.have_query_embed:
            assert key_dim == query_dim


            # define the layer normalization
            if self.layernorm:
                self.pre_ln = LayerNorm(query_dim)
                self.add_module("attn_preln", self.pre_ln)

            if self.use_post_mat:
                self.post_linear = nn.Linear(query_dim, query_dim)
                self.dropout = nn.Dropout(p=dropout)
                # self.register_parameter("attn_PostLinear", self.post_linear)

                # self.post_W = nn.Parameter(torch.FloatTensor(query_dim, query_dim))
                # init.xavier_uniform_(self.post_W)
                # self.register_parameter("attn_PostW", self.post_W)

                # self.post_B = nn.Parameter(torch.FloatTensor(1,query_dim))
                # init.xavier_uniform_(self.post_B)
                # self.register_parameter("attn_PostB",self.post_B)
                if self.layernorm:
                    self.post_ln = LayerNorm(query_dim)
                    self.add_module("attn_Postln", self.post_ln)
      
            # each column represent an attention vector for one attention head: [embed_dim*2, num_attn]
            self.atten_vecs = nn.Parameter(torch.FloatTensor(query_dim+key_dim+spa_embed_dim+g_spa_embed_dim, self.num_attn))
            init.xavier_uniform_(self.atten_vecs)
            self.register_parameter("attn_attenvecs", self.atten_vecs)
        else:
            # if we do not use query embedding in the attention, this means
            # We just compute the initial query embedding 
            # define the layer normalization
            if self.layernorm:
                self.pre_ln = LayerNorm(key_dim)
                self.add_module("attn_nq_preln", self.pre_ln)

            if self.use_post_mat:
                self.post_linear = nn.Linear(key_dim, key_dim)
                self.dropout = nn.Dropout(p=dropout)
                # self.register_parameter("attn_PostLinear", self.post_linear)

                # self.post_W = nn.Parameter(torch.FloatTensor(key_dim, key_dim))
                # init.xavier_uniform_(self.post_W)
                # self.register_parameter("attn_nq_PostW", self.post_W)

                # self.post_B = nn.Parameter(torch.FloatTensor(1,key_dim))
                # init.xavier_uniform_(self.post_B)
                # self.register_parameter("attn_nq_PostB",self.post_B)
                if self.layernorm:
                    self.post_ln = LayerNorm(key_dim)
                    self.add_module("attn_nq_Postln", self.post_ln)
            
            # In the initial query embedding computing, we just use key embeddings and spatial relation embeddings
            # each column represent an attention vector for one attention head: [embed_dim*2, num_attn]
            self.atten_vecs = nn.Parameter(torch.FloatTensor(key_dim+spa_embed_dim+g_spa_embed_dim, self.num_attn))
            init.xavier_uniform_(self.atten_vecs)
            self.register_parameter("attn_nq_attenvecs", self.atten_vecs)
        




    def forward(self, key_embeds, key_spa_embeds, query_g_spa_embeds, query_embed = None):
        '''
        Args:
            
            key_embeds: a list of feature embeddings computed from different context point, 
                        [batch_size, num_context_pt, key_dim]
            key_spa_embeds: a list of spatial relation embeddings computed from differnet context point, 
                        [batch_size, num_context_pt, spa_embed_dim]
            key_spa_embeds: a list of global spatial embeddingsof the center point
                        [batch_size, g_spa_embed_dim]
            query_embed: the pre-computed variable embeddings, 
                        [batch_size, query_dim]
                        have_query_embed (True):
                        have_query_embed (False): None
        Return:
            combined: the multi-head attention based embeddings for center pt [batch_size, key_dim]
        '''
        tensor_size = key_embeds.size()
        num_context_pt = tensor_size[1]
        batch_size = tensor_size[0]
        if key_spa_embeds.size()[0] > 0: # we have key_spa_embeds
            assert num_context_pt == key_spa_embeds.size()[1]
            assert batch_size == key_spa_embeds.size()[0]
        
        if query_g_spa_embeds.size()[0] > 0: # we have query_g_spa_embeds
            assert batch_size == query_g_spa_embeds.size()[0]
            # query_g_spa_embeds_expand: [batch_size, num_context_pt, g_spa_embed_dim]
            query_g_spa_embeds_expand = query_g_spa_embeds.unsqueeze(1).expand(batch_size, num_context_pt, self.g_spa_embed_dim)

        if self.have_query_embed:
            # We use the pre-computed query_embed to do attention
            # assert query_embed != None
            assert batch_size == query_embed.size()[0] 
            query_dim = query_embed.size()[1]
            assert query_dim == self.query_dim == self.key_dim
            # query_embed_expand: [batch_size, num_context_pt, query_dim]
            query_embed_expand = query_embed.unsqueeze(1).expand(batch_size, num_context_pt, query_dim)
            if key_spa_embeds.size()[0] > 0: # we have key_spa_embeds
                if query_g_spa_embeds.size()[0] > 0: # we have query_g_spa_embeds
                    # concat: [batch_size, num_context_pt, query_dim+key_dim+spa_embed_dim+g_spa_embed_dim]
                    concat = torch.cat((query_embed_expand, key_embeds, key_spa_embeds, query_g_spa_embeds_expand), dim=2)
                else:
                    # concat: [batch_size, num_context_pt, query_dim+key_dim+spa_embed_dim]
                    concat = torch.cat((query_embed_expand, key_embeds, key_spa_embeds), dim=2)
            else:
                # concat: [batch_size, num_context_pt, query_dim+key_dim]
                concat = torch.cat((query_embed_expand, key_embeds), dim=2)
        else:
            # We just use the context feature embedding (key_embeds) and spatial embedding to compute initial center point embedding
            assert query_embed == None

            if key_spa_embeds.size()[0] > 0: # we have key_spa_embeds 
                if query_g_spa_embeds.size()[0] > 0: # we have query_g_spa_embeds
                    # concat: [batch_size, num_context_pt, key_dim+spa_embed_dim+g_spa_embed_dim]
                    concat = torch.cat((key_embeds, key_spa_embeds, query_g_spa_embeds_expand), dim=2)
                else:
                    # concat: [batch_size, num_context_pt, key_dim+spa_embed_dim]
                    concat = torch.cat((key_embeds, key_spa_embeds), dim=2)
            else:
                concat = key_embeds
        
        # 1. compute the attention score for each key embeddings
        # attn: [batch_size, num_context_pt, num_attn]
        attn = torch.einsum("bnd,dk->bnk", (concat, self.atten_vecs))
        # attn: [batch_size, num_context_pt, num_attn]
        attn = self.softmax(self.activation(attn))
        # attn: [batch_size, num_attn, num_context_pt]
        attn = attn.transpose(1,2)
        # 2. using the attention score to compute the weighted average of the key embeddings
        # combined: [batch_size, num_attn, key_dim]
        combined = torch.einsum("bkn,bnd->bkd", (attn, key_embeds))
        # combined: [batch_size, key_dim]
        combined = torch.sum(combined, dim=1,keepdim=False) * (1.0/self.num_attn)
        # combined: [batch_size, key_dim]
        combined =  self.f_activation(combined)

        # Note that query_dim == key_dim
        if self.layernorm:
            if self.have_query_embed: 
                # residual connection
                combined = combined + query_embed
            combined = self.pre_ln(combined)

        if self.use_post_mat:
            # linear: [batch_size, query_dim]
            # linear = combined.mm(self.post_W) + self.post_B
            linear = self.dropout(self.post_linear(combined))
            if self.layernorm:
                linear = linear + combined
                linear = self.post_ln(linear)
            return linear

        
        return combined



'''
The Decoder of Location, given a position embedding, decode to point feature embedding
'''

class DirectPositionEmbeddingDecoder(nn.Module):
    def __init__(self, g_spa_embed_dim, feature_embed_dim, 
        f_act = "sigmoid", dropout = 0.5):
        '''
        
        Args:
            g_spa_embed_dim: the global position embedding dimention
            feature_embed_dim: the feature embedding dimention
            f_act: the final activation function applied to get the final result
        '''
        super(DirectPositionEmbeddingDecoder, self).__init__()
        self.g_spa_embed_dim = g_spa_embed_dim
        self.feature_embed_dim = feature_embed_dim

        self.post_linear = nn.Linear(self.g_spa_embed_dim, self.feature_embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.f_act = get_activation_function(f_act, "DirectPositionEmbeddingDecoder")
        

    def forward(self, g_spa_embeds):
        '''
        Args:
            g_spa_embeds: a batch of global position embedding of a list of point
                (batch_size, g_spa_embed_dim)
        Return:
            embeds: the predicted feature embeddings
                (batch_size, feature_embed_dim)
        '''
        embeds = self.f_act(self.dropout(self.post_linear(g_spa_embeds)))
        return embeds


'''
The Decoder of Location, 
given two predicted feature embedding:
1) from the context points, 
2) from the center point location
decode the point feature embedding
'''
class JointRelativeGlobalDecoder(nn.Module):
    def __init__(self, feature_embed_dim, 
        f_act = "sigmoid", dropout = 0.5, join_type = "cat"):
        '''
        
        Args:
            
            feature_embed_dim: the feature embedding dimention
            f_act: the final activation function applied to get the final result
        '''
        super(JointRelativeGlobalDecoder, self).__init__()
        self.feature_embed_dim = feature_embed_dim
        self.join_type = join_type

        if self.join_type == "cat":
            self.post_linear = nn.Linear(self.feature_embed_dim*2, self.feature_embed_dim)
        else:
            self.post_linear = nn.Linear(self.feature_embed_dim, self.feature_embed_dim)

        self.dropout = nn.Dropout(p=dropout)

        self.f_act = get_activation_function(f_act, "JointRelativeGlobalDecoder")
        

    def forward(self, context_feature_embeds, spa_feature_embeds):
        '''
        Args:
            context_feature_embeds: a batch of the predicted center point feature embedding from context points
                (batch_size, feature_embed_dim)
            spa_feature_embeds: a batch of the predicted center point feature embedding from global position of center pts
                (batch_size, feature_embed_dim)
        Return:
            embeds: the predicted feature embeddings
                (batch_size, feature_embed_dim)
        '''
        
        if self.join_type == "cat":
            aggs = torch.cat((context_feature_embeds, spa_feature_embeds), dim=1)
        else:
            # aggs: shape (2, batch_size, feature_embed_dim)
            aggs = torch.stack([context_feature_embeds, spa_feature_embeds])
            if self.join_type == "mean":
                aggs = torch.mean(aggs, dim=0, keepdim=False)
            elif self.join_type == "min":
                aggs = torch.min(aggs, dim=0, keepdim=False)
            elif self.join_type == "max":
                aggs = torch.max(aggs, dim=0, keepdim=False)
            if type(aggs) == tuple:
                # For torch.min/torch.max, the result is a tuple (min_value/max_value, index_tensor), we just get the 1st
                # For torch.mean, the result is just mean_value
                # so we need to check the result type
                aggs = aggs[0]

        embeds = self.f_act(self.dropout(self.post_linear(aggs)))
        return embeds

