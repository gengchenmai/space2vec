import torch
import utils as ut
import math


def bce_loss(pred):
    return -torch.log(pred + 1e-5)


def rand_samples(batch_size, params, rand_type='uniform'):
    '''
    randomly sample background locations, generate (lon, lat, date) and put into pre loc encoder
    Note that the generated (lon, lat) are between [-1, 1] for geo_net
    But for our spa_enc, they generate real (lat, lon)
    Return:
        rand_feats: shape (batch_size, input_feat_dim)
    '''
    spa_enc_type = params['spa_enc_type']

    # randomly sample background locations and date
    # the generated location and date from [-1, 1]
    rand_feats_orig = torch.rand(batch_size, 3).to(params['device'])*2 -1

    if rand_type == 'spherical':
        # theta is between (0, 2*pi), computed based on latitude
        theta = ((rand_feats_orig[:,1].unsqueeze(1)+1) / 2.0)*(2*math.pi)
        r_lon = torch.sqrt(1.0 - rand_feats_orig[:,0].unsqueeze(1)**2) * torch.cos(theta)
        r_lat = torch.sqrt(1.0 - rand_feats_orig[:,0].unsqueeze(1)**2) * torch.sin(theta)
        # rand_feats_orig: (batch_size, 3)
        rand_feats_orig = torch.cat((r_lon, r_lat, rand_feats_orig[:,2].unsqueeze(1)), 1)

    if spa_enc_type == "geo_net":
        rand_feats = ut.encode_loc_time(rand_feats_orig[:,:2], rand_feats_orig[:,2], concat_dim=1, params=params)
    
    elif spa_enc_type in ut.get_spa_enc_list():
        lon = torch.unsqueeze(rand_feats_orig[:,0] * 180, dim = 1)
        lat = torch.unsqueeze(rand_feats_orig[:,1] * 90, dim = 1)
        # rand_feats: shape (batch_size, input_feat_dim = 2)
        rand_feats = torch.cat((lon, lat), 1).to(params["device"])
    else:
        raise Exception("spa_enc not defined!!!")

    return rand_feats


def embedding_loss(model, params, loc_feat, loc_class, user_ids, inds):
    '''
    Args:
        model:
        param:
        loc_feat: shape (batch_size, input_feat_dim)
        loc_class: shape (batch_size)
        user_ids: shape (batch_size)
        inds: tensor, [0,1,2,...,batch_size-1]
    '''

    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # create random background samples
    # loc_feat_rand: (batch_size, input_feat_dim)
    loc_feat_rand = rand_samples(batch_size, params, rand_type='spherical')

    # get location embeddings
    # loc_cat: (2*batch_size, input_feat_dim)
    loc_cat = torch.cat((loc_feat, loc_feat_rand), 0)
    loc_emb_cat = model(loc_cat, return_feats=True)
    # the location embedding for training samples, (batch_size, num_filts)
    loc_emb = loc_emb_cat[:batch_size, :]
    # the location embedding for random selected samples, (batch_size, num_filts)
    loc_emb_rand = loc_emb_cat[batch_size:, :]

    # the prediction distribution for training samples, (batch_size, num_classes)
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    # the prediction distribution for random selected samples, (batch_size, num_classes)
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))

    # data loss
    # see equation 7 in paper https://arxiv.org/abs/1906.05272
    pos_weight = params['num_classes']
    # loss_pos: (batch_size, num_classes)
    loss_pos = bce_loss(1.0 - loc_pred)  # neg
    # update probability at the training sample's correct class
    loss_pos[inds[:batch_size], loc_class] = pos_weight*bce_loss(loc_pred[inds[:batch_size], loc_class])  # pos
    loss_bg = bce_loss(1.0 - loc_pred_rand)

    if 'user' in params['train_loss']:

        # user location loss
        # see equation 8 in paper https://arxiv.org/abs/1906.05272

        # note: self.user_emb.weight shape (num_users, num_filts)
        # get the user embedding for each data sample
        # user: (batch_size, num_filts)
        user = model.user_emb.weight[user_ids, :]
        # p_u_given_l/p_u_given_randl:  (batch_size)
        p_u_given_l = torch.sigmoid((user*loc_emb).sum(1))
        p_u_given_randl = torch.sigmoid((user*loc_emb_rand).sum(1))

        # user_loc_pos_loss/user_loc_neg_loss: (batch_size)
        user_loc_pos_loss = bce_loss(p_u_given_l)
        user_loc_neg_loss = bce_loss(1.0 - p_u_given_randl)

        # user class loss
        # see equation 9 in paper https://arxiv.org/abs/1906.05272
        # p_c_given_u: (batch_size, num_classes)
        p_c_given_u = torch.sigmoid(torch.matmul(user, model.class_emb.weight.transpose(0,1)))
        # user_class_loss: (batch_size, num_classes)
        user_class_loss = bce_loss(1.0 - p_c_given_u)
        user_class_loss[inds[:batch_size], loc_class] = pos_weight*bce_loss(p_c_given_u[inds[:batch_size], loc_class])

        # total loss
        loss = loss_pos.mean() + loss_bg.mean() + user_loc_pos_loss.mean() + \
               user_loc_neg_loss.mean() + user_class_loss.mean()

    else:

        # total loss
        loss = loss_pos.mean() + loss_bg.mean()

    return loss
