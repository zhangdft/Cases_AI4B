
import os
import sys
import argparse
import shutil
import time
import warnings
import csv
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

# Add cgcnn path to sys.path
CGCNN_PATH = "/data/home/5240019/class_AI4Bat/cgcnn"
sys.path.append(CGCNN_PATH)

from cgcnn.data import CIFData, collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

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

def mae(prediction, target):
    """Computes the mean absolute error between prediction and target"""
    return torch.mean(torch.abs(prediction - target))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks - Fine-tune')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--cpu-threads', default=0, type=int, metavar='N',
                    help='number of CPU threads for computation (default: 0, uses all)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: [100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pre-trained model (default: none)')
parser.add_argument('--train-ratio', default=0.8, type=float, metavar='N',
                    help='number of training data to be loaded (default 0.8)')
parser.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default 0.1)')
parser.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')

parser.add_argument('--optim', default='Adam', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: Adam)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=32, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=4, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()
best_mae_error = 1e10

def main():
    global args, best_mae_error

    # Set CPU threads for computation if specified
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        print(f"=> Using {args.cpu_threads} CPU threads for computation")
    
    # load data
    dataset = CIFData(*args.data_options)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=None,
        val_size=None,
        test_size=None,
        return_test=True)

    # obtain target value normalizer
    if len(dataset) < 500:
        sample_data_list = [dataset[i] for i in range(len(dataset))]
    else:
        sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,
                                n_h=args.n_h,
                                classification=False)
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)

    # load pre-trained model
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pre-trained model '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location='cuda' if args.cuda else 'cpu')
            
            # Match weights
            model_dict = model.state_dict()
            pretrained_dict = checkpoint['state_dict']
            
            # Filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            
            # Update current model dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no pre-trained model found at '{}'".format(args.pretrained))

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer)

        # evaluate on validation set
        mae_error = validate(val_loader, model, criterion, normalizer)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer, test=True)


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    model.train()
    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]), Variable(input[1]), input[2], input[3])
        
        target_normed = normalizer.norm(target)
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        output = model(*input_var)
        loss = criterion(output, target_var)

        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, mae_errors=mae_errors))


def validate(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    model.eval()
    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]), Variable(input[1]), input[2], input[3])
        
        target_normed = normalizer.norm(target)
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        output = model(*input_var)
        loss = criterion(output, target_var)

        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        
        if test:
            test_targets.extend(target.tolist())
            test_preds.extend(normalizer.denorm(output.data.cpu()).view(-1).tolist())
            test_cif_ids.extend(batch_cif_ids)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                mae_errors=mae_errors))

    if test:
        star_label = '**'
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, t, p in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, t, p))
    else:
        star_label = '*'
    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label, mae_errors=mae_errors))
    return mae_errors.avg

if __name__ == '__main__':
    main()
