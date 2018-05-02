from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import ipdb

import platform

print('python version: {}'.format(platform.python_version()))
print('PyTorch version: {}'.format(torch.__version__))

print('\nTest on python 3.6.4\t Pytorch 0.4.0\n')

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
# train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC_person', choices=['VOC_person', 'COCO_person', 'RAP'],
                    type=str, help='VOC_person, COCO_person or RAP')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--gpu_id', default='2', type=str,
                    help='gpu id')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print('GPU ID: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    global writer
    writer = SummaryWriter('runs_{}'.format(args.dataset))
    cfg = person_cfg
    if args.dataset == 'COCO_person':
        train_dataset = COCOPersonDetection(phase='trian', transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
        val_dataset = COCOPersonDetection(phase='test', transform=BaseTransform(300, MEANS))
    elif args.dataset == 'VOC_person':
        train_dataset = VOCPersonDetection(phase='train',
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
        val_dataset = VOCPersonDetection(phase='test', transform=BaseTransform(300, MEANS))
    elif args.dataset == 'RAP':
        print('not implement error')
        return

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume) # finetuning here
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    net = net.cuda()

    # draw net graph in tensorboardX :)
    # dummy_input = Variable(torch.rand(1, 1, 300, 300))
    # dummy_input.cuda()
    # writer.add_graph(net, (dummy_input,))

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], overlap_thresh=0.5,
                            prior_for_matching=True, bkg_label=0, neg_mining=True,
                            neg_pos=3, neg_overlap=0.5,
                            encode_target=False)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    print('Loading the dataset...')
    epoch_size = len(train_dataset) // args.batch_size
    print('epoch_size: {}'.format(epoch_size))
    print('Training SSD on:', train_dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    train_loader = data.DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    # create batch iterator
    batch_iterator = iter(train_loader)  # error 517
    # for i, (input, target) in enumerate(train_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if iteration in cfg['lr_steps']:
            step_index += 1
            lr = adjust_learning_rate(optimizer, args.gamma, step_index)
            writer.add_scalar('lr', lr, iteration)
        try:
            # load train data
            images, targets = next(batch_iterator)  # error
        except StopIteration:
            batch_iterator = iter(train_loader)
            images, targets = next(batch_iterator)  # error
        # measure data loading time
        data_time.update(time.time() - end)

        images = Variable(images.cuda())
        # targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        targets = [Variable(ann.cuda()) for ann in targets]
        # forward
        t1 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        # loc_loss += loss_l.data[0]
        # conf_loss += loss_c.data[0]
        losses.update(loss.data[0].cpu(), images.size(0))
        writer.add_scalar('loss', loss.data[0], iteration)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % 10 == 0:
            # ipdb.set_trace()
            print(
                'iter: {}\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    iteration, data_time=data_time, batch_time=batch_time,
                    loss=losses
                ))

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_{}_'.format(rgs.dataset) +
                       repr(iteration) + '.pth')
            for name, param in net.named_parameters():
                writer.add_histogram(name, param, iteration, bins="auto")
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

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


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    train()
