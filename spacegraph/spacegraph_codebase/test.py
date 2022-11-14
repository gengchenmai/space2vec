import argparse
import os
import json
import pickle
import numpy as np
import torch
from argparse import Namespace

from spacegraph_codebase.Place2Vec.cur_data_utils import load_pointset
from spacegraph_codebase.trainer import make_enc_dec, make_args_combine
from spacegraph_codebase.utils import *
from spacegraph_codebase.data import NeighborGraph
from spacegraph_codebase.data import PointSet


def embed_points(model_dir, neighborgraphs, point_list, num_poi_type):
    # load config and model
    with open(os.path.join(model_dir, "config.json"), "r") as infile:
        config = json.load(infile)
    config["load_model"] = True
    args = Namespace(**config)

    feature_embed_lookup = init_feature_embedding(args.embed_dim, num_poi_type)

    test_ng_list = [NeighborGraph.deserialize(info) for info in neighborgraphs]
    pointset = PointSet(
        point_list,
        num_poi_type,
        feature_embed_lookup,
        10,
        "TYPE",
        args.embed_dim,
        do_feature_sampling=False
    )
    enc_dec = make_enc_dec(args, pointset=pointset, feature_embedding=None)
    enc_dec.load_state_dict(
        torch.load(
            os.path.join(model_dir + str(make_args_combine(args)) + ".pth")
        )
    )
    enc_dec.eval()
    final_embedding = enc_dec.predict(test_ng_list)
    return final_embedding


def init_feature_embedding(embed_dim, num_poi_type):
    feature_embedding = torch.nn.Embedding(num_poi_type, embed_dim)
    feature_embedding.weight.data.normal_(0, 1. / embed_dim)

    # The POI Type embedding lookup function, given a list of POI type id,
    # get their embedding
    feature_embed_lookup = lambda pt_types: feature_embedding(
        torch.autograd.Variable(torch.LongTensor(pt_types))
    )
    return feature_embed_lookup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_path",
        default="data_collection/example_poi_data",
        type=str
    )
    parser.add_argument(
        "-m",
        "--model_path",
        default="model_dir/global_example_data",
        type=str
    )
    args = parser.parse_args()

    model_dir = args.model_path
    data_dir = args.data_path

    neighborgraphs = pickle.load(
        open(data_dir + "/neighborgraphs_test.pkl", "rb"), encoding='latin1'
    )
    num_poi_type, point_list = pickle.load(
        open(os.path.join(data_dir, "pointset.pkl"), "rb"), encoding='latin1'
    )
    final_embedding = embed_points(
        model_dir, neighborgraphs, point_list, num_poi_type
    )
    print("Output is your embedding with shape", final_embedding.shape)
