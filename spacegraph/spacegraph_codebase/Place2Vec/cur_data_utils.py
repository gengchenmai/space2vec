import os
import pickle
import torch
from collections import OrderedDict, defaultdict
from multiprocessing import Process
import random
import json

from spacegraph_codebase.data import PointSet, NeighborGraph,Point


def load_pointset(data_dir, point_data_path = "/pointset.pkl", num_feature_sample = 3, embed_dim = 10, do_feature_sampling = False):
    '''
    Args:
        point_data_path: the pointset data (num_poi_type, point_list)
            num_poi_type: total number of poi type
            point_list: a list of point tuple (PT): (id, (X, Y), (Type1, Type2,...TypeM), training/validation/test)
        num_feature_sample: each POI have different num of POI Type, we resample a fix number of POI Types for each POI
        embed_dim: embedding dimention
    '''
    num_poi_type, point_list = pickle.load(open(data_dir+point_data_path, "rb"), encoding='latin1')

    feature_dim = embed_dim
    feature_embedding = torch.nn.Embedding(num_poi_type, embed_dim)
    feature_embedding.weight.data.normal_(0, 1./embed_dim)

    # The POI Type embedding lookup function, given a list of POI type id, get their embedding
    feature_embed_lookup = lambda pt_types: feature_embedding(
            torch.autograd.Variable(torch.LongTensor(pt_types)))

    pointset = PointSet(point_list, num_poi_type, feature_embed_lookup, feature_dim, "TYPE", num_feature_sample, do_feature_sampling = do_feature_sampling)
    return pointset, feature_embedding


def make_data_samples(data_mode, pointset, data_dir, neighbor_tuple_path, neg_sample_num = 10):
    
    # print("Load PointSet....")
    # pointset = load_pointset(data_dir, point_data_path)

    print("Load {} neighbor_tuple_list".format(data_mode))
    neighbor_tuple_list = pickle.load(open(data_dir + "/" + neighbor_tuple_path, "rb"))

    print("Do negative sampling and get NeighborGraph")
    ng_list = pointset.get_data_samples(neighbor_tuple_list, neg_sample_num, data_mode)

    print("Dump {} NeighborGraph()".format(data_mode))
    pickle.dump([ng.serialize() for ng in ng_list], open(data_dir+"/neighborgraphs_{}.pkl".format(data_mode), "w"), protocol=pickle.HIGHEST_PROTOCOL)


def poi_type_pre_sampling(data_dir, out_file = "/pointset.pkl"):
    print("Load the raw PointSet file")
    pointset, feature_embedding = load_pointset(data_dir, point_data_path="/pointset_raw.pkl", do_feature_sampling = True)
    info = pointset.serialize()

    print("Dump Sampled PointSet")
    pickle.dump(info, open(data_dir+out_file, "w"), protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':

    data_dir="/home/gengchen/Position_Encoding/data_collection/Place2Vec_center/"
    poi_type_pre_sampling(data_dir, out_file = "/pointset.pkl")

    print("Load new data")
    pointset, feature_embedding = load_pointset(data_dir, point_data_path="/pointset.pkl", do_feature_sampling=False)
    
    print("Train: neg {}".format(10))
    data_mode = "training"
    neighbor_tuple_path = "/neighbortuple_{}.pkl".format(data_mode)
    make_data_samples(data_mode, pointset, data_dir, neighbor_tuple_path, neg_sample_num = 10)

    print("Valid: neg {}".format(100))
    data_mode = "validation"
    neighbor_tuple_path = "/neighbortuple_{}.pkl".format(data_mode)
    make_data_samples(data_mode, pointset, data_dir, neighbor_tuple_path, neg_sample_num = 100)

    print("Test: neg {}".format(100))
    data_mode = "test"
    neighbor_tuple_path = "/neighbortuple_{}.pkl".format(data_mode)
    make_data_samples(data_mode, pointset, data_dir, neighbor_tuple_path, neg_sample_num = 100)

