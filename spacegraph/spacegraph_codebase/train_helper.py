import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import random
import numpy as np
from collections import defaultdict

from spacegraph_codebase.model import NeighGraphEncoderDecoder

def check_conv(vals, window=2, tol=1e-6):
    '''
    Check the convergence of mode based on the evaluation score:
    Args:
        vals: a list of evaluation score
        window: the average window size
        tol: the threshold for convergence
    '''
    if len(vals) < 2 * window:
        return False
    conv = np.mean(vals[-window:]) - np.mean(vals[-2*window:-window]) 
    return conv < tol

def update_loss(loss, losses, ema_loss, ema_alpha=0.01):
    losses.append(loss)
    if ema_loss is None:
        ema_loss = loss
    else:
        ema_loss = (1-ema_alpha)*ema_loss + ema_alpha*loss
    return losses, ema_loss

def run_eval(model, ng_list, iteration, logger, batch_size=1000, do_full_eval = True):
    '''
    Given a list of NeighborGraph(), evaluate MRR and HIT@1,5,10 by negative sampling
    Args:
        model: the NeighGraphEncoderDecoder() model
        ng_list: a list of NeighborGraph()
    Return:
        
    '''
    if model is not None:
        # switch to a evaluation mode
        model.eval()

    # a list of ranks of all ng testing samples
    full_ranks = []
    offset = 0
    # split the ng_list into batches
    while offset < len(ng_list):
        max_index = min(offset+batch_size, len(ng_list))
        batch_ng_list = ng_list[offset:max_index]
        ranks = get_batch_ranks(model, batch_ng_list, do_full_eval)
        full_ranks += ranks

        offset += batch_size

    mrr, hit1, hit5, hit10 = eval_stat(full_ranks)

    return mrr, hit1, hit5, hit10

def run_eval_per_type(model, pointset, ng_list, iteration, logger, typeid2root = None, batch_size=1000, do_full_eval = True):
    '''
    Given a list of NeighborGraph(), evaluate MRR and HIT@1,5,10 by negative sampling
    Args:
        model: the NeighGraphEncoderDecoder() model
        pointset: PointSet()
        ng_list: a list of NeighborGraph()
    Return:
        
    '''
    if model is not None:
        # switch to a evaluation mode
        model.eval()

    # a list of ranks of all ng testing samples
    full_ranks = []
    offset = 0
    # split the ng_list into batches
    while offset < len(ng_list):
        max_index = min(offset+batch_size, len(ng_list))
        batch_ng_list = ng_list[offset:max_index]
        ranks = get_batch_ranks(model, batch_ng_list, do_full_eval)
        full_ranks += ranks

        offset += batch_size

    num_sample = len(full_ranks)

    type2rank = dict()
    for i, ng in enumerate(ng_list):
        type_list = list(pointset.pt_dict[ng.center_pt].features)
        if typeid2root is not None:
            type_list = list(set([typeid2root[typeid] for typeid in type_list]))
        for pt_type in type_list:
            if pt_type not in type2rank:
                type2rank[pt_type] = []
            type2rank[pt_type].append(full_ranks[i])
            
    type2mrr   = dict()
    type2hit1  = dict()
    type2hit5  = dict()
    type2hit10 = dict()
    for pt_type in type2rank:
        type2mrr[pt_type], type2hit1[pt_type], type2hit5[pt_type], type2hit10[pt_type] = eval_stat(type2rank[pt_type])


    return type2mrr, type2hit1, type2hit5, type2hit10



def eval_stat(full_ranks):
    num_sample = len(full_ranks)

    # compute MRR, HIT@1,5,10
    mrr = 0.0
    hit1 = 0.0
    hit5 = 0.0
    hit10 = 0.0

    for rank in full_ranks:
        mrr += 1.0/rank
        if rank <= 1:
            hit1 += 1.0
        if rank <= 5:
            hit5 += 1.0
        if rank <= 10:
            hit10 += 1.0

    mrr /= num_sample
    hit1 /= num_sample
    hit5 /= num_sample
    hit10 /= num_sample

    return mrr, hit1, hit5, hit10

def get_batch_ranks(model, ng_list, do_full_eval = True):
    '''
    Given a list of NeighborGraph()
    Args:
        model: the NeighGraphEncoderDecoder() model
        ng_list: (batch_size), a list of NeighborGraph()
    Return:
        ranks: (batch_size), a list of rank, each rank indicate the rank of the positive prediction among all pos+neg samples
    '''
    # pos: (batch_size)
    # neg: (batch_size, num_neg_sample)
    if model:
        pos, neg = model.get_batch_scores(ng_list, do_full_eval)

        # scores: (batch_size, num_neg_sample+1)
        # note for each sample, the 1st is the positive one
        scores = torch.cat((pos.unsqueeze(1), neg), dim=1)
        scores = np.asarray(scores.data.tolist())
    else:
        # if model == None, we just do random guess
        batch_size = len(ng_list)
        if do_full_eval:
            num_neg_sample = 100
        else:
            num_neg_sample = 10
        pos = np.random.randn(batch_size, 1)
        neg = np.random.randn(batch_size, num_neg_sample)
        scores = np.concatenate((pos, neg), axis=1)

    batch_size, num_pt = scores.shape

    # argsort(): sort the dot product scores for each prediction list, then each rank position is their original index
    # argmin(): get the index of min(), get the rank-1 of the 1st item (pos pair) in the ranking list
    # num_pt - (): Note argsort() do sort in ascending order, dot product should use descending order
    #              we use num_pt - () to get the rank of pos pair in the descending order list
    ranks = num_pt - np.argmin(np.argsort(scores, axis = -1), axis = -1)
    # ranks: (batch_size)

    return list(ranks)

def run_train(model, optimizer, train_ng_list, val_ng_list, test_ng_list, logger,
        max_iter=int(10e7), batch_size=512, log_every=100, val_every=1000, tol=1e-6,
        model_file=None):
    ema_loss = None
    vals = []
    losses = []

    ema_loss_val = None
    losses_val = []

    if model is not None:
        random.shuffle(train_ng_list)
        for i in range(max_iter):
            # switch to training mode
            model.train()
            optimizer.zero_grad()
            loss = run_batch(train_ng_list, model, i, batch_size, do_full_eval=True) # use all 10 neg samples
            losses, ema_loss = update_loss(loss.item(), losses, ema_loss)
            loss.backward()
            optimizer.step()

            loss_val = run_batch(val_ng_list, model, i, batch_size, do_full_eval=False) # sample 10 neg samples
            losses_val, ema_loss_val = update_loss(loss_val.item(), losses_val, ema_loss_val)

            if i % log_every == 0:
                logger.info("Iter: {:d}; Train ema_loss: {:f}".format(i, ema_loss))
                logger.info("Iter: {:d}; Validate ema_loss: {:f}".format(i, ema_loss_val))


            if i >= val_every and i % val_every == 0:
                mrr, hit1, hit5, hit10 = run_eval(model, random.sample(train_ng_list, len(val_ng_list)), i, logger, do_full_eval = True)
                logger.info("Iter: {:d}; 10 Neg, Train MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))


                mrr, hit1, hit5, hit10 = run_eval(model, val_ng_list, i, logger, do_full_eval = False)
                logger.info("Iter: {:d}; 10 Neg, Validate MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))

                mrr, hit1, hit5, hit10 = run_eval(model, val_ng_list, i, logger, do_full_eval = True)
                logger.info("Iter: {:d}; 100 Neg, Validate MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))
                
                vals.append(mrr)
                if not model_file is None:
                    torch.save(model.state_dict(), model_file)

            # if check_conv(vals, tol=tol):
            #     logger.info("Fully converged at iteration {:d}".format(i))
            #     break
    else:
        i = 0
    
    mrr, hit1, hit5, hit10 = run_eval(model, random.sample(train_ng_list, len(val_ng_list)), i, logger, do_full_eval = True)
    logger.info("Iter: {:d}; 10 Neg, Train MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))

    mrr, hit1, hit5, hit10 = run_eval(model, val_ng_list, i, logger, do_full_eval = False)
    logger.info("Iter: {:d}; 10 Neg, Validate MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))

    mrr, hit1, hit5, hit10 = run_eval(model, val_ng_list, i, logger, do_full_eval = True)
    logger.info("Iter: {:d}; 100 Neg, Validate MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))
    
    mrr_, hit1_, hit5_, hit10_ = run_eval(model, test_ng_list, i, logger, do_full_eval = False)
    logger.info("Iter: {:d}; 10 Neg, Test MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr_, hit1_, hit5_, hit10_))

    mrr_, hit1_, hit5_, hit10_ = run_eval(model, test_ng_list, i, logger, do_full_eval = True)
    logger.info("Iter: {:d}; 100 Neg, Test MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr_, hit1_, hit5_, hit10_))
    

def run_joint_train(global_model, relative_model, join_model, 
        global_optimizer, relative_optimizer, join_optimizer,
        train_ng_list, val_ng_list, test_ng_list, logger,
        max_iter=int(10e7), max_burn_in =10000, batch_size=512, log_every=100, val_every=1000, tol=1e-6,
        model_file=None, relative_conv=False, global_conv=False):
    relative_max_burn_in = 500
    global_max_burn_in = 2000
    
    vals = []
    ema_loss = None
    losses = []

    ema_loss_val = None
    losses_val = []

    conv_rel_mrr = None
    conv_rel_hit1 = None
    conv_rel_hit5 = None
    conv_rel_hit10 = None

    conv_glb_mrr = None
    conv_glb_hit1 = None
    conv_glb_hit5 = None
    conv_glb_hit10 = None

    random.shuffle(train_ng_list)
    for i in xrange(max_iter):
        
        if not relative_conv:
            # we need to train relative_model
            model = relative_model
            optimizer = relative_optimizer
        elif not global_conv:
            # we need to train global_model
            model = global_model
            optimizer = global_optimizer
        else:
            # we train join_model
            model = join_model
            optimizer = join_optimizer

        # switch to training mode
        model.train()
        optimizer.zero_grad()

        loss = run_batch(train_ng_list, model, i, batch_size, do_full_eval=True) # use all 10 neg samples
        losses, ema_loss = update_loss(loss.item(), losses, ema_loss)
        loss.backward()
        optimizer.step()

        loss_val = run_batch(val_ng_list, model, i, batch_size, do_full_eval=False) # sample 10 neg samples
        losses_val, ema_loss_val = update_loss(loss_val.item(), losses_val, ema_loss_val)

        


        if i % log_every == 0:
            logger.info("Iter: {:d}; Train ema_loss: {:f}".format(i, ema_loss))
            logger.info("Iter: {:d}; Validate ema_loss: {:f}".format(i, ema_loss_val))


        if i >= val_every and i % val_every == 0:
            mrr, hit1, hit5, hit10 = run_eval(model, random.sample(train_ng_list, len(val_ng_list)), i, logger, do_full_eval = True)
            logger.info("Iter: {:d}; 10 Neg, Train MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))


            mrr, hit1, hit5, hit10 = run_eval(model, val_ng_list, i, logger, do_full_eval = False)
            logger.info("Iter: {:d}; 10 Neg, Validate MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))

            mrr, hit1, hit5, hit10 = run_eval(model, val_ng_list, i, logger, do_full_eval = True)
            logger.info("Iter: {:d}; 100 Neg, Validate MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))
            
            vals.append(mrr)
            if not model_file is None:
                if not relative_conv:
                    torch.save(join_model.state_dict(), model_file.replace(".pth", "--relative_conv.pth"))
                elif not global_conv:
                    torch.save(join_model.state_dict(), model_file.replace(".pth", "--global_conv.pth"))
                else:
                    torch.save(join_model.state_dict(), model_file)
    

        # if not relative_conv and (check_conv(vals, tol=tol) or len(losses) >= max_burn_in):
        if not relative_conv and ( len(losses) >= relative_max_burn_in):
            logger.info("Relative converged at iteration {:d}".format(i-1))
            # logger.info("Testing at relative conv...")
            conv_rel_mrr, conv_rel_hit1, conv_rel_hit5, conv_rel_hit10 = run_eval(model, test_ng_list, i, logger, do_full_eval = True)
            logger.info("Testing at relative conv... Iter: {:d}; 100 Neg, Test MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, conv_rel_mrr, conv_rel_hit1, conv_rel_hit5, conv_rel_hit10))
            relative_conv = True
            vals = []
            ema_loss = None
            losses = []
            ema_loss_val = None
            losses_val = []
            if not relative_conv:
                torch.save(join_model.state_dict(), model_file.replace(".pth", "--relative_conv.pth"))

        # if relative_conv and not global_conv and (check_conv(vals, tol=tol) or len(losses) >= max_burn_in):
        if relative_conv and not global_conv and ( len(losses) >= global_max_burn_in):
            logger.info("Global converged at iteration {:d}".format(i-1))
            # logger.info("Testing at global conv...")
            conv_glb_mrr, conv_glb_hit1, conv_glb_hit5, conv_glb_hit10 = run_eval(model, test_ng_list, i, logger, do_full_eval = True)
            logger.info("Testing at global conv... Iter: {:d}; 100 Neg, Test MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, conv_glb_mrr, conv_glb_hit1, conv_glb_hit5, conv_glb_hit10))
            global_conv = True
            vals = []
            ema_loss = None
            losses = []
            ema_loss_val = None
            losses_val = []
            # freeze all parameters except the join_dec
            join_model.freeze_param_except_join_dec()
            if not global_conv:
                torch.save(join_model.state_dict(), model_file.replace(".pth", "--global_conv.pth"))

        # if relative_conv and global_conv:
        #     if check_conv(vals, tol=tol):
        #         logger.info("Fully converged at iteration {:d}".format(i))
        #         break

    model = join_model

    mrr, hit1, hit5, hit10 = run_eval(model, random.sample(train_ng_list, len(val_ng_list)), i, logger, do_full_eval = True)
    logger.info("Fully conv Iter: {:d}; 10 Neg, Train MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))

    mrr, hit1, hit5, hit10 = run_eval(model, val_ng_list, i, logger, do_full_eval = False)
    logger.info("Fully conv Iter: {:d}; 10 Neg, Validate MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))

    mrr, hit1, hit5, hit10 = run_eval(model, val_ng_list, i, logger, do_full_eval = True)
    logger.info("Fully conv Iter: {:d}; 100 Neg, Validate MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))
    
    mrr_, hit1_, hit5_, hit10_ = run_eval(model, test_ng_list, i, logger, do_full_eval = False)
    logger.info("Fully conv Iter: {:d}; 10 Neg, Test MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr_, hit1_, hit5_, hit10_))

    mrr_, hit1_, hit5_, hit10_ = run_eval(model, test_ng_list, i, logger, do_full_eval = True)
    logger.info("Fully conv Iter: {:d}; 100 Neg, Test MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr_, hit1_, hit5_, hit10_))
    if conv_rel_mrr is not None:
        logger.info("Improv. from relative conv: MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(mrr_ - conv_rel_mrr, 
                                                                                                            hit1_ - conv_rel_hit1, 
                                                                                                            hit5_ - conv_rel_hit5, 
                                                                                                            hit10_ - conv_rel_hit10))
    if conv_glb_mrr is not None:
        logger.info("Improv. from global conv: MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(mrr_ - conv_glb_mrr, 
                                                                                                            hit1_ - conv_glb_hit1, 
                                                                                                            hit5_ - conv_glb_hit5, 
                                                                                                            hit10_ - conv_glb_hit10))                                                                                               


def run_batch(train_ng_list, enc_dec, iter_count, batch_size, do_full_eval):
    '''
    Given the training NeighborGraph() list and the iterator num, find the batch and train encoder-decoder
    Args:
        train_ng_list: a list of training NeighborGraph()
        enc_dec: encoder-decoder model
        iter_count: scaler, iterator num
        batch_size: 
        do_full_eval: whether to use full negative to do eval
        
    '''
    
    n = len(train_ng_list)
    start = (iter_count * batch_size) % n
    end = min(((iter_count+1) * batch_size) % n, n)
    end = n if end <= start else end
    # print("start: {:d}\tend: {:d}".format(start, end))
    ng_list = train_ng_list[start:end]
    loss = enc_dec.softmax_loss(ng_list, do_full_eval)
    return loss