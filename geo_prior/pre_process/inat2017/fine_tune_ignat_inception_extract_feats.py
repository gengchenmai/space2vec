# iNatularist training and testing code adapted from
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import ignat_loader
import numpy as np
import inception


# training and testing hyper parameters
class Params:
    arch = 'inception_v3'
    num_classes = 5089
    im_size_train = 299
    im_size_test = 299
    workers = 6
    epochs = 64
    start_epoch = 0
    batch_size = 32
    lr = 0.0045
    lr_decay = 0.94
    epoch_decay = 4
    momentum = 0.9
    weight_decay = 0.0
    print_freq = 100
    evaluate = False
    resume = ''

    rootdir = '/media/macaodha/ssd_data/inat_2017/'                             # path to train images
    train_file = '/media/macaodha/ssd_data/inat_2017/inat2017_anns/train2017.json'              # path to train file
    val_file = '/media/macaodha/ssd_data/inat_2017/inat2017_anns/val2017.json'                  # path to val file

    # uncomment if want to evaluate the test set and save a submission file
    evaluate = True
    save_preds = True
    resume = '/home/macaodha/Downloads/iNat_2017_InceptionV3.pth.tar'

    #split_name = 'train'
    #rootdir = '/media/macaodha/ssd_data/inat_2017/'
    split_name = 'val'
    rootdir = '/media/macaodha/ssd_data/inat_2017/'
    #split_name = 'test'
    #rootdir = '/media/macaodha/Data/inat2017_test_images/'

    val_file = '/media/macaodha/ssd_data/inat_2017/inat2017_anns/' + split_name + '2017.json'
    op_file_name = '/media/macaodha/Data/inat_2017_rebuttal_tmp/inat2017_' + split_name + '_preds'

best_prec1 = 0


def class_ave(gt, pred):
    classes = np.unique(gt)
    correct = (pred == gt[..., np.newaxis]).sum(1)
    cls_ave = 0.0
    for cc in classes:
        cls_ave += correct[np.where(gt==cc)[0]].mean()
    cls_ave /= float(len(classes))

    print('top 5 acc\t' + str(round(correct.mean()*100, 4)))
    print('cls ave acc\t' + str(round(cls_ave*100, 4)))
    return cls_ave*100


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    print('Epoch:{0}'.format(epoch))
    print('Itr\t\tTime\t\tData\t\tLoss\t\tPrec@1\t\tPrec@5')
    for i, (input, target, im_id) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
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


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    gt = []
    pred = []
    im_ids = []
    feats = []
    print('Validate:\tTime\t\tLoss\t\tPrec@1\t\tPrec@5')
    for i, (input, target, im_id) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output, net_feats = model(input_var, return_feats=True)

        # store the top 5 classes for the prediction
        im_ids.append(im_id.cpu().numpy().astype(np.int))
        gt.append(target.cpu().numpy().astype(np.int))
        _, pred_inds = output.data.topk(5,1,True,True)
        pred.append(pred_inds.cpu().numpy().astype(np.int))

        # measure accuracy and record loss
        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        if args.save_preds:
            #sm = torch.nn.functional.softmax(output)
            #feats.append(sm.cpu().data.numpy())
            feats.append(net_feats.cpu().data.numpy())

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

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
          .format(top1=top1, top5=top5))

    # print class average acc
    im_ids = np.hstack(im_ids)
    gt = np.hstack(gt)
    pred = np.vstack(pred)
    cls_avg = class_ave(gt, pred)

    if args.save_preds:
        return top1.avg, cls_avg, pred, im_ids, np.vstack(feats)
    else:
        return top1.avg, cls_avg, pred, im_ids, None

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print(' * Saving new best model')
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
    """Decays the learning rate"""
    lr = args.lr * (args.lr_decay ** (epoch // args.epoch_decay))
    for param_group in optimizer.state_dict()['param_groups']:
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
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main():
    global args, best_prec1
    args = Params()

    # create model
    print("Using pre-trained model '{}'".format(args.arch))
    #model = models.__dict__[args.arch](pretrained=True)
    model = inception.inception_v3(pretrained=True)
    model.fc = nn.Linear(2048, args.num_classes)
    model.aux_logits = False
    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # data loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        ignat_loader.IGNAT_Loader(args.rootdir, args.train_file,
            transforms.Compose([
            transforms.RandomSizedCrop(args.im_size_train),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ignat_loader.IGNAT_Loader(args.rootdir, args.val_file,
            transforms.Compose([
            transforms.Scale(int(args.im_size_test/0.875)),
            transforms.CenterCrop(args.im_size_test),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        _, _, preds, im_ids, feats = validate(val_loader, model, criterion, args)
        # write predictions to file
        if args.save_preds:
            np.save(args.op_file_name, feats)
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
        prec1, cls_avg5, _, _, _ = validate(val_loader, model, criterion, args)

        # remember best prec@1 and save checkpoint
        # could also save based on cls_avg
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


if __name__ == '__main__':
    main()
