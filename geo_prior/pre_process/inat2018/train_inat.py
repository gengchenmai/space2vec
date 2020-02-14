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
#import torchvision.models as models
from inception import *

import inat2018_loader

class Params:
    # arch = 'inception_v3'
    num_classes = 8142
    workers = 6
    epochs = 100
    start_epoch = 0
    batch_size = 64  # might want to make smaller
    lr = 0.0045
    lr_decay = 0.94
    epoch_decay = 4
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 100

    #resume = ''  # set this to path of model to resume training
    resume = '/home/macaodha/Projects/inat2018/code/model_best.pth.tar'
    train_file = '/media/Storage_1/Datasets/inat2018/train2018.json'
    val_file = '/media/Storage_1/Datasets/inat2018/val2018.json'
    data_root = '/media/Storage_1/Datasets/inat2018/images/'

    # set evaluate to True to run the test set
    evaluate = True
    save_preds = True
    op_file_name = 'inat2018_test_preds_new.csv' # submission file
    if evaluate == True:
        val_file = '/media/Storage_1/Datasets/inat2018/test2018.json'

best_prec3 = 0.0  # store current best top 3


def main():
    global args, best_prec3
    args = Params()

    # load pretrained model
    # use this line if instead if you want to train another model
    #model = models.__dict__[args.arch](pretrained=True)
    model = inception_v3(pretrained=True)
    model.fc = nn.Linear(2048, args.num_classes)
    model.aux_logits = False
    #model = torch.nn.DataParallel(model).cuda()
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
            best_prec3 = checkpoint['best_prec3']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # data loading code
    train_dataset = inat2018_loader.INAT(args.data_root, args.train_file,
                     is_train=True)
    val_dataset = inat2018_loader.INAT(args.data_root, args.val_file,
                     is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                   shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                  batch_size=args.batch_size, shuffle=False,
                  num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        prec3, preds, im_ids = validate(val_loader, model, criterion, True)
        # write predictions to file
        if args.save_preds:
            with open(args.op_file_name, 'w') as opfile:
                opfile.write('id,predicted\n')
                for ii in range(len(im_ids)):
                    opfile.write(str(im_ids[ii]) + ',' + ' '.join(str(x) for x in preds[ii,:])+'\n')
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec3 = validate(val_loader, model, criterion, False)

        # remember best prec@1 and save checkpoint
        is_best = prec3 > best_prec3
        best_prec3 = max(prec3, best_prec3)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec3': best_prec3,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    print('Epoch:{0}'.format(epoch))
    print('Itr\t\tTime\t\tData\t\tLoss\t\tPrec@1\t\tPrec@3')
    for i, (input, im_id, target, tax_ids) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

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
                '{top3.val:.2f} ({top3.avg:.2f})'.format(
                i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top3=top3))


def validate(val_loader, model, criterion, save_preds=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    pred = []
    im_ids = []

    print('Validate:\tTime\t\tLoss\t\tPrec@1\t\tPrec@3')
    for i, (input, im_id, target, tax_ids) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        if save_preds:
            # store the top K classes for the prediction
            im_ids.append(im_id.cpu().numpy().astype(np.int))
            _, pred_inds = output.data.topk(3,1,True,True)
            pred.append(pred_inds.cpu().numpy().astype(np.int))

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  '{loss.val:.3f} ({loss.avg:.3f})\t'
                  '{top1.val:.2f} ({top1.avg:.2f})\t'
                  '{top3.val:.2f} ({top3.avg:.2f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))

    if save_preds:
        return top3.avg, np.vstack(pred), np.hstack(im_ids)
    else:
        return top3.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("\tSaving new best model")
        shutil.copyfile(filename, 'model_best.pth.tar')


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
