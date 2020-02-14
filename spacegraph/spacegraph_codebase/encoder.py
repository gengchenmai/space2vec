from sets import Set

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from spacegraph_codebase.data import PointSet, NeighborGraph,Point


class PointFeatureEncoder(nn.Module):
	'''
	Given a list of point id, encode their feature embeddings based on POI type embedding
	'''
	def __init__(self, feature_embed_lookup, feature_embedding, pointset, agg_func=torch.mean):
		'''
		Args:
			feature_embed_lookup: The POI Type embedding lookup function 
			feature_embedding: POI Type embedding matrix
		'''
		super(PointFeatureEncoder, self).__init__()
		self.add_module("feat-embed", feature_embedding)
		self.feature_embed_lookup = feature_embed_lookup
		self.pointset = pointset
		self.num_feature_sample = pointset.num_feature_sample
		self.agg_func = agg_func

	def forward(self, pt_list):
		'''
		Args:
			pt_list: a list of point id, shape (batch_size)
		Return:
			aggs: a list of feature embeddings of each point, shape (batch_size, embed_dim)
		'''
		feature_list = []
		for pid in pt_list:
			feature_list.append(list(self.pointset.pt_dict[pid].features))
		# feature_list: shape (batch_size, num_feature_sample)

		# embeds: shape (batch_size, num_feature_sample, embed_dim)
		embeds = self.feature_embed_lookup(feature_list)
		# norm: shape (batch_size, num_feature_sample, 1)
		norm = embeds.norm(p=2, dim=2, keepdim=True)
		# normalize the embedding vectors
		# embeds_norm: shape (batch_size, num_feature_sample, embed_dim)
		embeds_norm = embeds.div(norm.expand_as(embeds))
		aggs = self.agg_func(embeds_norm, dim=1, keepdim=False)
		if type(aggs) == tuple:
			# For torch.min/torch.max, the result is a tuple (min_value/max_value, index_tensor), we just get the 1st
            # For torch.mean, the result is just mean_value
            # so we need to check the result type
			aggs = aggs[0]
		# aggs: shape (batch_size, embed_dim)

		# normalize the point feature vectors
		# aggs_norm: shape (batch_size, 1)
		aggs_norm = aggs.norm(p=2, dim=1, keepdim=True)
		aggs_normalize = aggs.div(aggs_norm.expand_as(aggs))

		return aggs_normalize



