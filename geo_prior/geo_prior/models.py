import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import math


class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out


class FCNet(nn.Module):
    def __init__(self, num_inputs, num_classes, num_filts, num_users=1):
        '''
        Args:
            num_inputs: input embedding dimention
            num_classes: number of categories we want to classify
            num_filts: hidden embedding dimention
        '''
        super(FCNet, self).__init__()
        self.inc_bias = False
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        self.user_emb = nn.Linear(num_filts, num_users, bias=self.inc_bias)

        self.feats = nn.Sequential(nn.Linear(num_inputs, num_filts),
                                    nn.ReLU(inplace=True),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts))

    def forward(self, x, class_of_interest=None, return_feats=False):
        '''
        Args:
            x: torch.FloatTensor(), input location features (batch_size, input_loc_dim = 2 or 3 or ...)
            class_of_interest: the class id we want to extract
            return_feats: whether or not just return location embedding
        '''
        loc_emb = self.feats(x)
        if return_feats:
            # loc_emb: (batch_size, num_filts)
            return loc_emb
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)

        # return (batch_size, num_classes)
        return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        '''
        Return:
            shape (batch_size)
        '''
        # note: self.class_emb.weight shape (num_classes, num_filts)
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :]) + self.class_emb.bias[class_of_interest]
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])


class TangNet(nn.Module):
    def __init__(self, ip_loc_dim, feats_dim, loc_dim, num_classes, use_loc):
        super(TangNet, self).__init__()
        self.use_loc  = use_loc
        self.fc_loc   = nn.Linear(ip_loc_dim, loc_dim)
        if self.use_loc:
            self.fc_class = nn.Linear(feats_dim+loc_dim, num_classes)
        else:
            self.fc_class = nn.Linear(feats_dim, num_classes)

    def forward(self, loc, net_feat):
        if self.use_loc:
            x = torch.sigmoid(self.fc_loc(loc))
            x = self.fc_class(torch.cat((x, net_feat), 1))
        else:
            x = self.fc_class(net_feat)
        return F.log_softmax(x, dim=1)



class LocationEncoder(nn.Module):
    def __init__(self, spa_enc, num_inputs, num_classes, num_filts, num_users=1):
        '''
        Args:
            spa_enc: the spatial encoder
            num_inputs: input embedding dimention
            num_classes: number of categories we want to classify
            num_filts: hidden embedding dimention
        '''
        super(LocationEncoder, self).__init__()
        self.spa_enc = spa_enc
        self.inc_bias = False
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        self.user_emb = nn.Linear(num_filts, num_users, bias=self.inc_bias)


    def forward(self, x, class_of_interest=None, return_feats=False):
        '''
        Args:
            x: torch.FloatTensor(), input location features (batch_size, input_loc_dim = 2)
            class_of_interest: the class id we want to extract
            return_feats: whether or not just return location embedding
        '''
        # loc_feat: (batch_size, 1, input_loc_dim = 2)
        loc_feat = torch.unsqueeze(x, dim=1)
        loc_feat = loc_feat.cpu().data.numpy()
        
        # loc_embed: torch.Tensor(), (batch_size, 1, spa_embed_dim = num_filts)
        loc_embed = self.spa_enc(loc_feat)
        # loc_emb: torch.Tensor(), (batch_size, spa_embed_dim = num_filts)
        loc_emb = loc_embed.squeeze(1)
        if return_feats:
            # loc_emb: (batch_size, num_filts)
            return loc_emb
        if class_of_interest is None:
            # class_pred: (batch_size, num_classes)
            class_pred = self.class_emb(loc_emb)
        else:
            # class_pred: shape (batch_size)
            class_pred = self.eval_single_class(loc_emb, class_of_interest)

        return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        '''
        Args:
            x: (batch_size, num_filts)
        Return:
            shape (batch_size)
        '''
        # note: self.class_emb.weight shape (num_classes, num_filts)
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :]) + self.class_emb.bias[class_of_interest]
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])