#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP

import moco.loader
import moco.builder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=2048, type=int,
                    help='queue size; number of negative keys (default: 2048)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def compute_cat(q_l1, q_l2, q_l3, q_l4, ids, female_ids, male_ids):
    new_batch_joint = []
    new_batch_marginal = []
    new_batch_joint3 = []
    new_batch_marginal3 = []
    new_batch_joint2 = []
    new_batch_marginal2 = []
    new_batch_joint1 = []
    new_batch_marginal1 = []
    for batch in range(q_l4.shape[0]):
        if(ids[batch] in female_ids):
            sensitive = torch.cat((torch.ones([1, 16, 16]), torch.zeros([1, 16, 16])), 0)
            sensitive3 = torch.cat((torch.ones([1, 32, 32]), torch.zeros([1, 32, 32])), 0)
            sensitive2 = torch.cat((torch.ones([1, 64, 64]), torch.zeros([1, 64, 64])), 0)
            sensitive1 = torch.cat((torch.ones([1, 128, 128]), torch.zeros([1, 128, 128])), 0)
        elif(ids[batch] in male_ids):
            sensitive = torch.cat((torch.zeros([1, 16, 16]), torch.ones([1, 16, 16])), 0)
            sensitive3 = torch.cat((torch.zeros([1, 32, 32]), torch.ones([1, 32, 32])), 0)
            sensitive2 = torch.cat((torch.zeros([1, 64, 64]), torch.ones([1, 64, 64])), 0)
            sensitive1 = torch.cat((torch.zeros([1, 128, 128]), torch.ones([1, 128, 128])), 0)

        joint = torch.cat((q_l4[batch], sensitive.cuda()), 0)
        joint3 = torch.cat((q_l3[batch], sensitive3.cuda()), 0)
        joint2 = torch.cat((q_l2[batch], sensitive2.cuda()), 0)
        joint1 = torch.cat((q_l1[batch], sensitive1.cuda()), 0)

        sensitive = torch.permute(F.one_hot(torch.randint(low=0, high=2, size=(16, 16)), num_classes=2), (2,0,1))
        sensitive3 = torch.permute(F.one_hot(torch.randint(low=0, high=2, size=(32, 32)), num_classes=2), (2,0,1))
        sensitive2 = torch.permute(F.one_hot(torch.randint(low=0, high=2, size=(64, 64)), num_classes=2), (2,0,1))
        sensitive1 = torch.permute(F.one_hot(torch.randint(low=0, high=2, size=(128, 128)), num_classes=2), (2,0,1))
        marginal = torch.cat((q_l4[batch], sensitive.type(torch.float32).cuda()), 0)
        marginal3 = torch.cat((q_l3[batch], sensitive3.type(torch.float32).cuda()), 0)
        marginal2 = torch.cat((q_l2[batch], sensitive2.type(torch.float32).cuda()), 0)
        marginal1 = torch.cat((q_l1[batch], sensitive1.type(torch.float32).cuda()), 0)

        new_batch_joint.append(torch.unsqueeze(joint, 0))
        new_batch_marginal.append(torch.unsqueeze(marginal, 0))
        new_batch_joint3.append(torch.unsqueeze(joint3, 0))
        new_batch_marginal3.append(torch.unsqueeze(marginal3, 0))
        new_batch_joint2.append(torch.unsqueeze(joint2, 0))
        new_batch_marginal2.append(torch.unsqueeze(marginal2, 0))
        new_batch_joint1.append(torch.unsqueeze(joint1, 0))
        new_batch_marginal1.append(torch.unsqueeze(marginal1, 0))

    new_batch_joint = torch.cat(new_batch_joint, 0)
    new_batch_marginal = torch.cat(new_batch_marginal, 0)
    new_batch_joint3 = torch.cat(new_batch_joint3, 0)
    new_batch_marginal3 = torch.cat(new_batch_marginal3, 0)
    new_batch_joint2 = torch.cat(new_batch_joint2, 0)
    new_batch_marginal2 = torch.cat(new_batch_marginal2, 0)
    new_batch_joint1 = torch.cat(new_batch_joint1, 0)
    new_batch_marginal1 = torch.cat(new_batch_marginal1, 0)

    new_batch = torch.cat([new_batch_joint, new_batch_marginal], 0)
    new_batch3 = torch.cat([new_batch_joint3, new_batch_marginal3], 0)
    new_batch2 = torch.cat([new_batch_joint2, new_batch_marginal2], 0)
    new_batch1 = torch.cat([new_batch_joint1, new_batch_marginal1], 0)

    return new_batch, new_batch3, new_batch2, new_batch1
    


def main_worker(gpu, ngpus_per_node, args):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # # initialize the process group
    # dist.init_process_group("gloo", rank=0, world_size=1)


    # create model
    print("=> creating model '{}'".format(args.arch))

    resnet50 = models.__dict__[args.arch](pretrained=True)
    resnet50.fc = nn.Linear(2048, args.moco_dim)
    resnet50.train()

    resnet50_2 = models.__dict__[args.arch](pretrained=True)
    resnet50_2.fc = nn.Linear(2048, args.moco_dim)
    resnet50_2.train()

    model = moco.builder.MoCo(
        resnet50, resnet50_2, 
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    # print(model)

    # model = moco.builder.MoCo(
    #     models.__dict__[args.arch],
    #     args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    # print(model)

    urban_ids = set(np.load('/scratch/mz2466/LoveDA_mixedD_moco/urban_ids.npy'))
    rural_ids = set(np.load('/scratch/mz2466/LoveDA_mixedD_moco/rural_ids.npy'))
    female_ids = set(np.load('/scratch/mz2466/Face/female_id.npy'))
    male_ids = set(np.load('/scratch/mz2466/Face/male_id.npy'))
    Mine = moco.builder.Mine(input_size=2050, hidden_size1=2050, hidden_size2=205, output_size=205)
    Mine3 = moco.builder.Mine3(input_size=2050, hidden_size1=2050, hidden_size2=205, output_size=205)
    Mine2 = moco.builder.Mine2(input_size=2050, hidden_size1=2050, hidden_size2=205, output_size=205)
    Mine1 = moco.builder.Mine1(input_size=2050, hidden_size1=2050, hidden_size2=205, output_size=205)


    model.cuda()
    Mine.cuda()
    Mine3.cuda()
    Mine2.cuda()
    Mine1.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_mine = torch.optim.Adam(Mine.parameters(), args.lr/2,
                                weight_decay=args.weight_decay)
    optimizer_mine3 = torch.optim.Adam(Mine3.parameters(), args.lr/2,
                                weight_decay=args.weight_decay)
    optimizer_mine2 = torch.optim.Adam(Mine2.parameters(), args.lr/2,
                                weight_decay=args.weight_decay)
    optimizer_mine1 = torch.optim.Adam(Mine1.parameters(), args.lr/2,
                                weight_decay=args.weight_decay)



    # optionally resume from a checkpoint
    if args.resume:
        ee = '18'
        if args.gpu is None:
            checkpoint = torch.load('checkpoint_mine_00'+ee+'.pth.tar')
            checkpoint3 = torch.load('checkpoint_mine3_00'+ee+'.pth.tar')
            checkpoint2 = torch.load('checkpoint_mine2_00'+ee+'.pth.tar')
            checkpoint1 = torch.load('checkpoint_mine1_00'+ee+'.pth.tar')
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load('checkpoint_mine_00'+ee+'.pth.tar', map_location=loc)
            checkpoint3 = torch.load('checkpoint_mine3_00'+ee+'.pth.tar', map_location=loc)
            checkpoint2 = torch.load('checkpoint_mine2_00'+ee+'.pth.tar', map_location=loc)
            checkpoint1 = torch.load('checkpoint_mine1_00'+ee+'.pth.tar', map_location=loc)
        Mine.load_state_dict(checkpoint['state_dict'])
        Mine3.load_state_dict(checkpoint3['state_dict'])
        Mine2.load_state_dict(checkpoint2['state_dict'])
        Mine1.load_state_dict(checkpoint1['state_dict'])
        optimizer_mine.load_state_dict(checkpoint['optimizer'])
        optimizer_mine3.load_state_dict(checkpoint3['optimizer'])
        optimizer_mine2.load_state_dict(checkpoint2['optimizer'])
        optimizer_mine1.load_state_dict(checkpoint1['optimizer'])


        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, '')
    normalize = transforms.Normalize(mean=[123.675, 116.28, 103.53],
                                     std=[58.395, 57.12, 57.375])
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            # transforms.RandomResizedCrop(1024),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            # transforms.RandomResizedCrop(512),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    train_dataset = moco.loader.TwoCropsTransform(traindir, transforms.Compose(augmentation))
    train_dataset_mine = moco.loader.TwoCropsTransform(traindir, transforms.Compose(augmentation))


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True)

    train_loader_mine = torch.utils.data.DataLoader(
        train_dataset_mine, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, train_loader_mine, train_dataset_mine, model, Mine, Mine3, Mine2, Mine1, criterion, optimizer, optimizer_mine, optimizer_mine3, optimizer_mine2, optimizer_mine1, epoch, urban_ids, rural_ids, args)

        if epoch % 1 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': Mine.state_dict(),
                'optimizer' : optimizer_mine.state_dict(),
            }, is_best=False, filename='checkpoint_mine_{:04d}.pth.tar'.format(epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': Mine3.state_dict(),
                'optimizer' : optimizer_mine3.state_dict(),
            }, is_best=False, filename='checkpoint_mine3_{:04d}.pth.tar'.format(epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': Mine2.state_dict(),
                'optimizer' : optimizer_mine2.state_dict(),
            }, is_best=False, filename='checkpoint_mine2_{:04d}.pth.tar'.format(epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': Mine1.state_dict(),
                'optimizer' : optimizer_mine1.state_dict(),
            }, is_best=False, filename='checkpoint_mine1_{:04d}.pth.tar'.format(epoch))


def train(train_loader, train_loader_mine, train_dataset_mine, model, Mine, Mine3, Mine2, Mine1, criterion, optimizer, optimizer_mine, optimizer_mine3, optimizer_mine2, optimizer_mine1, epoch, urban_ids, rural_ids, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    Mine_train_loss = []
    Mine_loss = []
    SL_loss = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, image_ids) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)


        # Training Mine
        Mine.train()
        Mine3.train()
        Mine2.train()
        Mine1.train()
        model.train()
        running_loss, running_mst = 0, 0
        for j, (images_mine, image_ids_mine) in enumerate(train_loader_mine):
        
            images_mine[0] = images_mine[0].cuda(args.gpu, non_blocking=True)
            images_mine[1] = images_mine[1].cuda(args.gpu, non_blocking=True)
            if (j > 30):
                train_loader_mine = torch.utils.data.DataLoader(train_dataset_mine, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
                break
            q, q_l1, q_l2, q_l3, q_l4 = model(im_q=images_mine[0], im_k=images_mine[1], training_mine=True)
            
            new_batch, new_batch3, new_batch2, new_batch1 = compute_cat(q_l1, q_l2, q_l3, q_l4, image_ids_mine, urban_ids, rural_ids)
            
            loss_m, mst_m = Mine(new_batch)
            loss_m3, mst_m3 = Mine3(new_batch3)
            loss_m2, mst_m2 = Mine2(new_batch2)
            loss_m1, mst_m1 = Mine1(new_batch1)
            #print(str(mst_m + mst_m3 + mst_m2))

            running_loss = running_loss +loss_m.item() +loss_m3.item() +loss_m2.item() +loss_m1.item()
            running_mst = running_mst +mst_m.item() +mst_m3.item() +mst_m2.item() +mst_m1.item()

            optimizer_mine.zero_grad()
            loss_m.backward()
            optimizer_mine.step()

            optimizer_mine3.zero_grad()
            loss_m3.backward()
            optimizer_mine3.step()

            optimizer_mine2.zero_grad()
            loss_m2.backward()
            optimizer_mine2.step()

            optimizer_mine1.zero_grad()
            loss_m1.backward()
            optimizer_mine1.step()

        Mine_train_loss.append(running_mst/30)
        #print("JSD loss from training is: " + str(running_loss/30))
        print("JSD mst from training is: " + str(running_mst/30))


        # compute output
        output, target, backbone_feature, q_l1, q_l2, q_l3, q_l4 = model(im_q=images[0], im_k=images[1], training_mine=False)
        #Mine.eval()
        #Mine3.eval()
        #Mine2.eval()

        new_batch, new_batch3, new_batch2, new_batch1 = compute_cat(q_l1, q_l2, q_l3, q_l4, image_ids, urban_ids, rural_ids)

        loss_mine, mst_mine = Mine(new_batch)
        loss_mine3, mst_mine3 = Mine3(new_batch3)
        loss_mine2, mst_mine2 = Mine2(new_batch2)
        loss_mine1, mst_mine1 = Mine1(new_batch1)

        Mine_loss.append(mst_mine.item()+mst_mine3.item()+mst_mine2.item()+mst_mine1.item())
        print("JSD mst is: " + str(Mine_loss[-1]))

        loss = criterion(output, target)
        SL_loss.append(loss.item())
        print("SL_Moco loss is: " + str(loss))
        loss = loss - loss_mine - loss_mine3 - loss_mine2 - loss_mine1

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    np.save(str(epoch)+'_JSD_train_mst.npy', Mine_train_loss)
    np.save(str(epoch)+'_JSD_mst.npy', Mine_loss)
    np.save(str(epoch)+'_SL_loss.npy', SL_loss)



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
