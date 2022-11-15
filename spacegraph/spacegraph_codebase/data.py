from collections import OrderedDict, defaultdict
import random
import numpy as np
import re

def _random_sampling(item_tuple, num_sample):
    '''
    poi_type_tuple: (Type1, Type2,...TypeM)
    '''

    type_list = list(item_tuple)
    if len(type_list) > num_sample:
        return tuple(np.random.choice(type_list, num_sample, replace=False))
    elif len(type_list) == num_sample:
        return item_tuple
    else:
        return tuple(np.random.choice(type_list, num_sample, replace=True))



class Point():
    '''
    define the attribute of Point
    '''
    def __init__(self, id, coord, features, data_mode, feature_func=None, num_sample=None):
        '''
        A point_tuple (PT): (id, (X, Y), (Type1, Type2,...TypeM), training/validation/test)
        '''
        self.id = id
        self.coord = tuple([coord[i] for i in range(len(coord))])
        self.coord_dim = len(coord)
        if feature_func is not None and num_sample is not None:
            self.features = feature_func(features, num_sample)
        else:
            self.features = features # the other feature of this point, tuple
        self.data_mode = data_mode # training/validation/test

    def __hash__(self):
        return hash((self.id, self.coord, self.features))

    def __eq__(self, other):
        return self.id == other.id

    def __neq__(self, other):
        return self.id != other.id

    def __str__(self):
        return "{}: coord: ({}) features: ({})".format(self.id, " ".join(list(self.coord)), " ".join(list(self.features)))

    def serialize(self):
        '''
        serialize the current Point()
        as (id, (X, Y), (Type1, Type2,...TypeM), training/validation/test)
        '''
        return (self.id, self.coord, self.features, self.data_mode)

class NeighborGraph():
    def __init__(self, neighbor_tuple, neg_samples): 
        '''
        Note we only use id to indicate points
        A neighbor_tuple: [CenterPT, [ContextPT1, ContextPT2, ..., ContextPTN], training/validation/test]
        A full tuple in data sample: (neighbor_tuple, NegativeSampleNum, (NegPT1, NegPT2, ...,NegPTJ))
        '''
        self.center_pt = neighbor_tuple[0]
        self.context_pts = tuple(neighbor_tuple[1]) # a tuple of context points id
        self.data_mode = neighbor_tuple[2] # training/validation/test
        if neg_samples is not None:
            self.neg_samples = neg_samples # a tuple of negative samples id

        self.sample_context_pts = None
        self.sample_neg_pts = None

    def sample_neighbor(self, num_sample):
        self.sample_context_pts = _random_sampling(self.context_pts, num_sample)

    def sample_neg(self, num_neg_sample):
        self.sample_neg_pts = list(_random_sampling(self.neg_samples, num_neg_sample))

    def __hash__(self):
         return hash((self.center_pt, self.context_pts, self.data_mode, self.neg_samples))

    def __eq__(self, other):
        '''
        The euqavalence between two neighborgraph
        '''
        return (self.center_pt, self.context_pts, self.data_mode, self.neg_samples) == (other.center_pt, other.context_pts, other.data_mode, other.neg_samples)

    def __neq__(self, other):
        return self.__hash__() != other.__hash__()

    def serialize(self):
        '''
        Serialize the current NeighborGraph() object as an entry for train/val/test data samples
        '''
        return (self.center_pt, self.context_pts, self.data_mode, self.neg_samples)

    @staticmethod
    def deserialize(serial_info):
        '''
        Given a entry (serial_info) in train/val/test data samples
        parse it as NeighborGraph() object
        '''
        return NeighborGraph(serial_info[:3], serial_info[3])

class PointSet():
    """
    This act as the knowledge base for 
    """
    def __init__(self, point_list, num_feature_type, feature_embed_lookup, 
            feature_dim, feature_mode, num_feature_sample, do_feature_sampling = False):
        '''
        Args:
            point_list: a list of point tuple (PT): (id, (X, Y), (Type1, Type2,...TypeM), training/validation/test)
            num_feature_type: total number of POI type
            feature_embed_lookup: lookup function for the embedding matrix of features (POI Type)
            feature_dim: the embedding dimention
            feature_mode: the mode of feature, "TYPE" (POI Type)
            num_feature_sample: each POI have different num of POI Type, we resample a fix number of POI Types for each POI
            do_feature_sampling: whether we sample the POI type
            
        '''
        self.num_feature_type = num_feature_type
        self.do_feature_sampling = do_feature_sampling
        self.feature_embed_lookup = feature_embed_lookup
        self.feature_dim = feature_dim
        self.feature_mode = feature_mode
        self.num_feature_sample = num_feature_sample

        self.pt_dict = defaultdict()
        self.pt_mode = defaultdict()
        self.pt_mode["training"] = set()
        self.pt_mode["validation"] = set()
        self.pt_mode["test"] = set()
        
        _, _, features, _ = point_list[0]
        init_num_feature = len(features)
        for point_tuple in point_list:
            id, coord, features, data_mode = point_tuple
            if self.feature_mode == "TYPE":
                if self.do_feature_sampling:
                    self.pt_dict[id] = Point(id, coord, features, data_mode,
                            feature_func=_random_sampling, num_sample=num_feature_sample)
                else:
                    assert init_num_feature == len(features)
                    self.pt_dict[id] = Point(id, coord, features, data_mode,
                            feature_func=None, num_sample=None)
            else:
                self.pt_dict[id] = Point(id, coord, features, data_mode,
                        feature_func=None, num_sample=None)
            self.pt_mode[data_mode].add(id)

        self.num_rbf_anchor_pts = None
        self.rbf_anchor_pts = None

    def make_spatial_extent(self, eps = 1.0):
        '''
        Get the spatial extent of the current Pointset
        Args:
            eps: make the extent a little bit larger
        Return:
            extent: (left, right, bottom, top)
        '''
        x_list = []
        y_list = []
        for id in self.pt_dict:
            x_list.append(self.pt_dict[id].coord[0])
            y_list.append(self.pt_dict[id].coord[1])

        left = np.floor(np.min(x_list) - eps)
        right = np.ceil(np.max(x_list) + eps)
        bottom = np.floor(np.min(y_list) - eps)
        top = np.ceil(np.max(y_list) + eps)
        return (left, right, bottom, top)

    def sample_RBF_anchor_points(self, num_rbf_anchor_pts = 100):
        '''
        Args:
            num_rbf_anchor_pts: the number of RBF anchor points we sample
        '''
        self.num_rbf_anchor_pts = num_rbf_anchor_pts
        self.rbf_anchor_pts = list(_random_sampling(self.pt_mode["training"], num_rbf_anchor_pts))
        # return rbf_anchor_pts

    def remove_points(self, remove_point_list):
        '''
        Given a list of point tuple. we remove these point from the point set
        Args:
            remove_point_list: a list of point tuple (PT): (id, (X, Y), (Type1, Type2,...TypeM))
        '''
        for point_tuple in remove_point_list:
            id, coord, features = point_tuple
            try:
                self.pt_dict.remove(id)
            except Exception:
                continue

    def get_negative_point_sample(self, neighbor_tuple, neg_sample_num):
        '''
        Given a neighbor_tuple, get N negative sample from the training/validation/test pointset
        Args:
            A neighbor_tuple: [CenterPT, [ContextPT1, ContextPT2, ..., ContextPTN], training/validation/test]
        Return:
            a list of negative samples id
        '''
        data_mode = neighbor_tuple[2]
        pt_list = list(self.pt_mode[data_mode]-set([neighbor_tuple[0]]+list(neighbor_tuple[1])))
        if len(pt_list) >= neg_sample_num:
            return list(np.random.choice(pt_list, neg_sample_num, replace=False))
        else:
            return list(np.random.choice(pt_list, neg_sample_num, replace=True))


    def get_data_samples(self, neighbor_tuple_list, neg_sample_num, data_mode):
        '''
        Given a list of neighbor_tuple from certain data_mode (training/validation/test) which coresponses, 
        This should contain each point in training/validation/test pointset as the center point
        get all negative sample and make final training/validation/test samples
        Args:
            neighbor_tuple_list: a list of neighbor_tuple
            neg_sample_num: number of negative sampling
            data_mode: training/validation/test
        Return:
            ng_list: a list of NeighborGraph()
        '''
        ng_list = []
        for neighbor_tuple in neighbor_tuple_list:
            assert neighbor_tuple[-1] == data_mode
            neg_samples = self.get_negative_point_sample(neighbor_tuple, neg_sample_num)
            ng = NeighborGraph(neighbor_tuple, neg_samples)
            ng_list.append(ng)

        return ng_list

    def serialize(self):
        '''
        Serialize the pointset
        '''
        pt_list = []
        for id in self.pt_dict:
            pt_list.append(self.pt_dict[id].serialize())

        return (self.num_feature_type, pt_list)





