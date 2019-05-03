import argparse
import os
import random
import shutil
import time
import warnings
import logging as log

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dp_models import SqueezeNet_DP


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0
best_total_num_zeros = 0

DP_zeros = 0
DP_loss = 0

DP_num_zeros_for_each_block = [0, 0, 0, 0]

# torch.set_printoptions(threshold=512*256)

log.basicConfig(filename='./train.log',
                format='%(asctime)s %(message)s', level=log.DEBUG)


def main():
    global args, best_prec1, best_total_num_zeros, DP_num_zeros_for_each_block
    args = parser.parse_args()

    model = SqueezeNet_DP()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(
        lambda p: p.requires_grad, model.parameters()),
                            args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            print("=> no checkpoint found at '{}'".format(args.resume))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    valdir = os.path.join(args.data, 'val')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    traindir = os.path.join(args.data, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        log.info('------ Epoch #{0} ------'.format(epoch))		
        train(train_loader, model, criterion, optimizer, epoch)
        adjust_learning_rate(optimizer, epoch)

        prec1 = validate(val_loader, model, criterion)

        is_best_acc = prec1 > best_prec1

        DP_total_num_zeros = 0
        for num_zeros in DP_num_zeros_for_each_block:
            DP_total_num_zeros += num_zeros
        DP_num_zeros_for_each_block = [0, 0, 0, 0]

        is_best_prun = DP_total_num_zeros > best_total_num_zeros

        best_prec1 = max(prec1, best_prec1)
        best_total_num_zeros = max(DP_total_num_zeros, best_total_num_zeros)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best_acc, is_best_prun)


def train(train_loader, model, criterion, optimizer, epoch):
    global DP_loss, DP_zeros

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    handle_1 = model.fire_1_dp.prun.register_forward_hook(DP_results_train)
    handle_2 = model.fire_2_dp.prun.register_forward_hook(DP_results_train)
    handle_3 = model.fire_3_dp.prun.register_forward_hook(DP_results_train)
    handle_4 = model.fire_4_dp.prun.register_forward_hook(DP_results_train)

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        output = model(input)

        loss = criterion(output, target)
        loss += DP_loss

        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(		  
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

        DP_loss = 0
        DP_zeros = 0

    handle_1.remove()
    handle_2.remove()
    handle_3.remove()
    handle_4.remove()


def validate(val_loader, model, criterion):
    global DP_zeros, DP_num_zeros_for_each_block, DP_loss
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    dp_losses = AverageMeter()
    dp_zeros = AverageMeter()

    handle1 = model.fire_1_dp.prun.register_forward_hook(DP_get_num_zeros_1)
    handle2 = model.fire_2_dp.prun.register_forward_hook(DP_get_num_zeros_2)
    handle3 = model.fire_3_dp.prun.register_forward_hook(DP_get_num_zeros_3)
    handle4 = model.fire_4_dp.prun.register_forward_hook(DP_get_num_zeros_4)

    handle_1 = model.fire_1_dp.prun.register_forward_hook(DP_results)
    handle_2 = model.fire_2_dp.prun.register_forward_hook(DP_results)
    handle_3 = model.fire_3_dp.prun.register_forward_hook(DP_results)
    handle_4 = model.fire_4_dp.prun.register_forward_hook(DP_results)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            output = model(input)
            loss = criterion(output, target)
            loss += DP_loss

            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            dp_losses.update(DP_loss)
            dp_zeros.update(DP_zeros)
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            log.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'DP_Loss {dp_loss.val:.4f} ({dp_loss.avg:.4f})\t'
                  'Number of zeros {num.val} ({num.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   dp_loss=dp_losses, num=dp_zeros,
                   top1=top1, top5=top5))

            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'DP_Loss {dp_loss.val:.4f} ({dp_loss.avg:.4f})\t'
                  'Number of zeros {num.val} ({num.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   dp_loss=dp_losses, num=dp_zeros,
                   top1=top1, top5=top5))

            DP_loss = 0
            DP_zeros = 0

            if i == len(val_loader) - 1:
                last_bs = output.size(0)

        handle1.remove()
        handle2.remove()
        handle3.remove()
        handle4.remove()

        handle_1.remove()
        handle_2.remove()
        handle_3.remove()
        handle_4.remove()

        DP_percentage_zeros_for_each_block = []
        DP_percentage_zeros_for_each_block.append(
            DP_num_zeros_for_each_block[0] / (
                ((len(val_loader) - 1) * args.batch_size + last_bs) * 16
                )
            )
        DP_percentage_zeros_for_each_block.append(
            DP_num_zeros_for_each_block[1] / (
                ((len(val_loader) - 1) * args.batch_size + last_bs) * 16
                )
            )
        DP_percentage_zeros_for_each_block.append(
            DP_num_zeros_for_each_block[2] / (
                ((len(val_loader) - 1) * args.batch_size + last_bs) * 32
                )
            )
        DP_percentage_zeros_for_each_block.append(
            DP_num_zeros_for_each_block[3] / (
                ((len(val_loader) - 1) * args.batch_size + last_bs) * 32
                )
            )

        log.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        log.info(DP_percentage_zeros_for_each_block)
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        print('\t', DP_percentage_zeros_for_each_block)

    return top1.avg


def DP_results(self, input, output):
    global DP_zeros, DP_loss
    vec = output
    vec = vec.cpu()

    for i in vec.view(-1):
        if i == 0:
            DP_zeros += 1

    DP_loss += torch.dist(vec.mean(), torch.tensor(0.5))


def DP_results_train(self, input, output):
    global DP_loss
    vec = output
    vec = vec.cpu()

    DP_loss += torch.dist(vec.mean(), torch.tensor(0.5))


def DP_get_num_zeros_1(self, input, output):
    global DP_num_zeros_for_each_block
    vec = output
    vec = vec.cpu()
    print(vec.size())
    for i in vec.view(-1):
        if i == 0:
            DP_num_zeros_for_each_block[0] += 1


def DP_get_num_zeros_2(self, input, output):
    global DP_num_zeros_for_each_block
    vec = output
    vec = vec.cpu()
    print(vec.size())
    for i in vec.view(-1):
        if i == 0:
            DP_num_zeros_for_each_block[1] += 1


def DP_get_num_zeros_3(self, input, output):
    global DP_num_zeros_for_each_block
    vec = output
    vec = vec.cpu()
    print(vec.size())
    for i in vec.view(-1):
        if i == 0:
            DP_num_zeros_for_each_block[2] += 1


def DP_get_num_zeros_4(self, input, output):
    global DP_num_zeros_for_each_block
    vec = output
    vec = vec.cpu()
    print(vec.size())
    for i in vec.view(-1):
        if i == 0:
            DP_num_zeros_for_each_block[3] += 1


def save_checkpoint(state, is_best_acc, is_best_prun,
                    filename='../DP_SqueezeNet/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best_acc:
        shutil.copyfile(filename,
                        '../DP_SqueezeNet/model_best_acc.pth.tar')
    if is_best_prun:
        shutil.copyfile(filename,
                        '../DP_SqueezeNet/model_best_prun.pth.tar')


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
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr * (0.1 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
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

# end_of_file
