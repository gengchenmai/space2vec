# Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py

import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models
from scipy import sparse

from inception import *
import birdsnap_loader

class Params:
    workers = 6
    epochs = 30
    start_epoch = 0
    batch_size = 32  # might want to make smaller
    lr = 0.0045
    lr_decay = 0.94
    epoch_decay = 4
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 100

    resume = ''  # set this to path of model to resume training
    num_classes = 500
    meta_data = '/media/macaodha/Data/datasets/birdsnap/birdsnap_with_loc_2019.json'
    data_root = '/media/macaodha/Data/datasets/birdsnap/images_sm/'
    split_name = 'test'

    # if AlexNet
    use_alexnet = False
    if use_alexnet:
        im_size_crop = 224
        im_size_resize = 256
        model_path = '/media/macaodha/Data/datasets/birdsnap/models/alexnet/'
        features_path = '/media/macaodha/Data/datasets/birdsnap/models/alexnet/'
    else:
        im_size_crop = 299
        im_size_resize = 342
        model_path = '/media/macaodha/Data/datasets/birdsnap/models/inception/'
        features_path = '/media/macaodha/Data/datasets/birdsnap/models/inception/'

    # set evaluate to True to run the test set
    evaluate = True
    save_preds = True  # if set to True will save model predictions
    if evaluate == True:
        resume = model_path + 'model_best.pth.tar'
        op_file_name = features_path + 'birdsnap_' + split_name + '_net_feats'

    best_prec1 = 0.0  # store current best top 1


def main():
    global args
    args = Params()

    if args.use_alexnet:
        print("Using pre-trained alexnet")
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, args.num_classes)
    else:
        print("Using pre-trained inception_v3")
        # inception is changed to accept variable size inputs
        model = inception_v3(pretrained=True)
        model.fc = nn.Linear(2048, args.num_classes)
        model.aux_logits = False

    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # data loading code
    train_dataset = birdsnap_loader.BS(args.data_root, args.meta_data,
                     split_name='train', im_size_crop=args.im_size_crop,
                     im_size_resize=args.im_size_resize, is_train=True)
    val_dataset = birdsnap_loader.BS(args.data_root, args.meta_data,
                     split_name=args.split_name, im_size_crop=args.im_size_crop,
                     im_size_resize=args.im_size_resize, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                   shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                  batch_size=args.batch_size, shuffle=False,
                  num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        prec1, preds, im_ids, feats = validate(val_loader, model, criterion, True, True)
        # write predictions to file
        if args.save_preds:
            # save dense
            np.save(args.op_file_name, feats)

            # save sparse
            #feats[feats<0.000001] = 0.0
            #sp = sparse.csr_matrix(feats)
            #sparse.save_npz(args.op_file_name + '_sparse', sp)

            # with open(args.op_file_name, 'w') as opfile:
            #     opfile.write('id,predicted\n')
            #     for ii in range(len(im_ids)):
            #         opfile.write(str(im_ids[ii]) + ',' + ' '.join(str(x) for x in preds[ii,:])+'\n')
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, False)
        is_better = prec1 > args.best_prec1
        # remember best Acc@1 and save checkpoint
        args.best_prec1 = max(prec1, args.best_prec1)
        model_state = {
            'epoch': epoch + 1,
            #'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': args.best_prec1,
            'optimizer' : optimizer.state_dict()}

        torch.save(model_state, args.model_path + 'checkpoint.pth.tar')
        if is_better:
            print('\t* Saving new best model')
            torch.save(model_state, args.model_path + 'model_best.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    print('\nEpoch:{0}'.format(epoch))
    print('Itr\t\tTime\t\tData\t\tLoss\t\tAcc@1\t\tAcc@5')
    for i, (input, im_id, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = input.cuda()
        target_var = target.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                '{data_time.val:.2f} ({data_time.avg:.2f})\t'
                '{loss.val:.3f} ({loss.avg:.3f})\t'
                '{top1.val:.2f} ({top1.avg:.2f})\t'
                '{top5.val:.2f} ({top5.avg:.2f})'.format(
                i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, save_preds=False, save_feats=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    pred = []
    im_ids = []
    feats = []
    with torch.no_grad():

        print('Validate:\tTime\t\tLoss\t\tAcc@1\t\tAcc@5')
        for i, (input, im_id, target) in enumerate(val_loader):

            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output, net_feats = model(input_var, return_feats=True)
            loss = criterion(output, target_var)

            if save_preds:
                # store the top K classes for the prediction
                im_ids.extend(im_id)
                _, pred_inds = output.data.topk(5,1,True,True)
                pred.append(pred_inds.cpu().numpy().astype(np.int))

            if save_feats:
                #sm = torch.nn.functional.softmax(output, 1)
                #feats.append(sm.cpu().data.numpy())
                feats.append(net_feats.cpu().data.numpy())

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('[{0}/{1}]\t'
                      '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                      '{loss.val:.3f} ({loss.avg:.3f})\t'
                      '{top1.val:.2f} ({top1.avg:.2f})\t'
                      '{top5.val:.2f} ({top5.avg:.2f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    print('\t* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    if save_preds:
        return top1.avg, np.vstack(pred), np.hstack(im_ids), np.vstack(feats)
    else:
        return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
