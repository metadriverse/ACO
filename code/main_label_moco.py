#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
os.environ["NCCL_DEBUG"] = "INFO"
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
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
import numpy as np

import moco.loader
import moco.builder

from utils.dataset import LabelYTBDataset
from utils.loss import LabelMoCoLoss
from utils.sampler import DistributedSamplerWrapper
from utils.visualizer import HtmlPageVisualizer
from tqdm import tqdm
import wandb

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ACO Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
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
parser.add_argument('--log-img-freq', default=1, type=int)
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
parser.add_argument('--ckpt-dir', default=None, type=str)
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--groupname', type=str, default='default')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
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

parser.add_argument('--aug-cf', action='store_true',
                    help='use cropping and flipping in augmentation')
parser.add_argument('--thres', default=0.04, type=float,
                    help='action similarity threshold')

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
        print(ngpus_per_node, args.world_size)
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    wandb.init(project='aco')
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.LabelMoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args.ckpt_dir)
    model.set_train_mode()
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    device_name = 'cuda'
    channel = args.moco_dim
    criterion = LabelMoCoLoss(channel=channel, T=args.moco_t, device=device_name, thres=args.thres)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
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
    traindir = args.data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop((180, 320), scale=(0.6, 1.)) if args.aug_cf else transforms.RandomGrayscale(p=0.0),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip() if args.aug_cf else transforms.RandomGrayscale(p=0.0),
            transforms.ToTensor(),
            normalize
        ]
        test_aug = [
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop((180, 320), scale=(0.2, 1.)) if args.aug_cf else transforms.RandomGrayscale(p=0.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip() if args.aug_cf else transforms.RandomGrayscale(p=0.0),
            transforms.ToTensor(),
            normalize
        ]
        test_aug = [
            transforms.ToTensor(),
            normalize
        ]

    train_dataset = LabelYTBDataset(data_path=traindir, interval=10, phase='train', 
            transform=moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    memory_dataset = LabelYTBDataset(data_path=traindir, interval=10, phase='train', 
            transform=transforms.Compose(test_aug))
    eval_dataset = LabelYTBDataset(data_path=traindir, interval=10, phase='eval', 
            transform=transforms.Compose(test_aug))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        samples_weight = torch.load('label.pt')
        total_len = len(samples_weight)
        train_samples_weight = samples_weight[:int(total_len * 0.7)]
        eval_samples_weight = samples_weight[int(total_len * 0.7):]
        # train_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_samples_weight, len(train_samples_weight))
        memory_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_samples_weight, len(train_samples_weight))
        eval_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(eval_samples_weight, len(eval_samples_weight))
        # train_sampler =  DistributedSamplerWrapper(train_weighted_sampler)
        memory_sampler =  DistributedSamplerWrapper(memory_weighted_sampler)
        eval_sampler =  DistributedSamplerWrapper(eval_weighted_sampler)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=(eval_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=eval_sampler, drop_last=True)

    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=args.batch_size, shuffle=(memory_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=memory_sampler, drop_last=True)

    cnt = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            # train_sampler.set_epoch(epoch)
            memory_sampler.set_epoch(epoch)
            eval_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        cnt += 1
        train(train_loader, model, criterion, optimizer, epoch, args, cnt)
        test(eval_loader, memory_loader, model, epoch, args)
        # evaluate(eval_loader, model, criterion, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))

def get_log_img(images, labels, q):
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    im1, im2 = images
    im1 = invTrans(im1).permute(0,2,3,1).detach().cpu()
    q = q.detach().cpu()
    html = HtmlPageVisualizer(num_rows=10, num_cols=10)
    with torch.no_grad():
        logit = torch.einsum('nk,mk->nm', q, q)
        for i in range(10):
            idx = torch.topk(logit[i], 9).indices
            idx = torch.cat([ torch.tensor([i]), idx])
            for ji, j in enumerate(idx):
                chose_img = im1[j]
                chose_img = np.array(chose_img * 255, dtype=np.uint8)
                html.set_cell(row_idx=i, col_idx=ji, image=chose_img, text=str(labels[j]))
    wandb_html = wandb.Html(html.get_html_str())
    return wandb_html

def knn_predict(feat, feat_bank, label_bank, knn_k=200, knn_t=0.1):
    # B,D * D,N -> B,N
    sim_matrix = feat @ feat_bank.T
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_weight = (sim_weight / knn_t).exp()

    pick_label = torch.gather(label_bank.repeat(feat.size(0), 1), dim=-1, index=sim_indices)
    weighted_label = sim_weight * pick_label
    weighted_label = weighted_label.sum(1) / sim_weight.sum(1)

    return weighted_label
        
def test(eval_loader, memory_loader, model, epoch, args):
    model.eval()

    total_num, top1 = 0, 0
    feat_bank, label_bank = [], []
    # build memory bank
    with torch.no_grad():
        for images, labels in tqdm(memory_loader, desc='feature extracting'):
            labels = labels.float().cuda(args.gpu, non_blocking=True)
            images = images.cuda(args.gpu, non_blocking=True)
            
            _, feat = model.module.extract_feature(images)
            feat_bank.append(feat)
            label_bank.append(labels)
        feat_bank = torch.cat(feat_bank, dim=0).contiguous()
        label_bank = torch.cat(label_bank).contiguous()

        test_bar = tqdm(eval_loader)
        for images, labels in test_bar:
            labels = labels.float().cuda(args.gpu, non_blocking=True)
            images = images.cuda(args.gpu, non_blocking=True)

            _, feat = model.module.extract_feature(images)
            pred_labels = knn_predict(feat, feat_bank, label_bank)
            total_num += images.size(0)
            top1 += (torch.abs(pred_labels - labels) < 0.08).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, top1 / total_num * 100))

    wandb.log({'eval-top1': top1 / total_num * 100})

# def evaluate(eval_loader, model, criterion, epoch, args):
#     model.eval()
#     batch_time = AverageMeter('Time', ':6.3f')
#     data_time = AverageMeter('Data', ':6.3f')
#     losses = AverageMeter('Ins Loss', ':.4e')
#     top1 = AverageMeter('Ins Acc@1', ':6.2f')
#     top5 = AverageMeter('Ins Acc@5', ':6.2f')
#     action_losses = AverageMeter('Act Loss', ':.4e')
#     action_top1 = AverageMeter('Act Acc@1', ':6.2f')
#     action_top5 = AverageMeter('Act Acc@5', ':6.2f')
# 
#     with torch.no_grad():
#         for i, (images, labels) in enumerate(eval_loader):
#             labels = labels.float()
#             # measure data loading time
#             if args.gpu is not None:
#                 images[0] = images[0].cuda(args.gpu, non_blocking=True)
#                 images[1] = images[1].cuda(args.gpu, non_blocking=True)
# 
#             q, k, queue_k, queue_label = model(im_q=images[0], im_k=images[1], label=labels, is_eval=True)
#             loss_list, acc1, acc5 = criterion(q, k, labels.to('cuda'), queue_k, queue_label)
# 
#             losses.update(loss_list[0].item(), images[0].size(0))
#             top1.update(acc1[0], images[0].size(0))
#             top5.update(acc5[0], images[0].size(0))
#             action_losses.update(loss_list[1].item(), images[0].size(0))
#             action_top1.update(acc1[1], images[0].size(0))
#             action_top5.update(acc5[1], images[0].size(0))
#         
#     log_dict = {
#         'eval/ins/loss': losses.avg,
#         'eval/ins/top1': top1.avg,
#         'eval/ins/top5': top5.avg,
#         'eval/act/loss': action_losses.avg,
#         'eval/act/top1': action_top1.avg,
#         'eval/act/top5': action_top5.avg,
#         }
#     wandb.log(log_dict)

def train(train_loader, model, criterion, optimizer, epoch, args, cnt):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Ins Loss', ':.4e')
    top1 = AverageMeter('Ins Acc@1', ':6.2f')
    top5 = AverageMeter('Ins Acc@5', ':6.2f')
    action_losses = AverageMeter('Act Loss', ':.4e')
    action_top1 = AverageMeter('Act Acc@1', ':6.2f')
    action_top5 = AverageMeter('Act Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, action_losses, action_top1, action_top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode

    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        labels = labels.float()    
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        q, k, queue_k, queue_label = model(im_q=images[0], im_k=images[1], label=labels)
        loss_list, acc1, acc5 = criterion(q, k, labels.to('cuda'), queue_k, queue_label)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        
        losses.update(loss_list[0].item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))
        action_losses.update(loss_list[1].item(), images[0].size(0))
        action_top1.update(acc1[1], images[0].size(0))
        action_top5.update(acc5[1], images[0].size(0))
        
        # compute gradient and do SGD step
        lambda_1 = (0.9) ** cnt
        optimizer.zero_grad()
        loss = loss_list[0]*1 +  loss_list[1] *0
        # loss = loss_list[0]
        loss.backward()
        optimizer.step()

        
        log_dict = {
            'lambda': lambda_1,
            'loss': loss.item(),
            'ins_loss': loss_list[0].item(),
            'ins_acc1': acc1[0],
            'ins_acc5': acc5[0],
            'act_loss': loss_list[1].item(),
            'act_acc1': acc1[1],
            'act_acc5': acc5[1],
            }
        if i % args.log_img_freq == 0:
            logimg = get_log_img(images, labels, q)
            log_dict['img'] = logimg
        wandb.log(log_dict)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


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
