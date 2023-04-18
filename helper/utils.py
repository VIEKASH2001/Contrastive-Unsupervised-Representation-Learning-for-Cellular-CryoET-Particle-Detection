import os
import sys
import time
import math

import operator
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import datetime
import cv2
import matplotlib.pyplot as plt
import copy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def comp_avg_angle(V):
    """
    computes average angle between pair of vectors in V
    :param V: vector of size bs x k x C x W x H, where k is number of tensors
    :return: angle in numpy degree
    """

    bs = V.shape[0]
    k = V.shape[1]
    g_v = V.view(bs, k, -1)
    dot = g_v @ g_v.transpose(1, 2)

    norm_v = g_v.norm(2, 2, keepdim=True)
    norm_v = torch.clamp_min(norm_v, 1e-10)

    mul = norm_v @ norm_v.transpose(1, 2)
    cos_t = dot / mul

    cos_t = torch.clamp(cos_t, -1, 1)
    rad = torch.acos(cos_t)

    # a mask to zero out diagonal values of the cosine similarity matrix
    eye_c = 1 - torch.eye(k).unsqueeze(0).to(rad.device)

    rad = rad * eye_c
    rad = rad.detach().cpu().numpy()

    deg = np.rad2deg(rad)

    deg_avg = deg.sum(1).sum(1).mean() / (k * (k - 1))
    return deg_avg


def combine_img_heatmap(img, heatmap, mix=0.3):
    im = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_RGB2BGR)
    heatmap = np.uint8(heatmap * 255)
    map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    mixed = cv2.addWeighted(im, 1 - mix, map, mix, 0)

    # mixed = cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB) / 255
    map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
    return mixed, map


def norm_minmax(x):
    """
    min-max normalization of numpy array
    """
    return (x - x.min()) / (x.max() - x.min())


def plot_tensor(t, normalize=True):
    """
    plot pytorch tensors
    input: list of tensors t
    """
    for i in range(len(t)):
        ti_np = t[i].cpu().detach().numpy().squeeze()


        if normalize:
            ti_np = norm_minmax(ti_np)
        if len(ti_np.shape) > 2:
            ti_np = ti_np.transpose(1, 2, 0)
        plt.subplot(1, len(t), i + 1)
        plt.imshow(ti_np)
    plt.show()


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class Logger:
    def __init__(self, var_names=None, format=None, args=None, print_args=False, save_dir=None):


        if args is None:
            self.dir = save_dir
        else:
            self.dir = args.save_dir
        self.var_names = var_names
        self.format = format
        self.vars = []

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        file = open(self.dir + '/log.txt', 'w')
        file.write('Log file created on ' + str(datetime.datetime.now()) + '\n\n')


        if args is not None:
            dict = {}
            for arg in vars(args):
                dict[arg] = str(getattr(args, arg))
                print(arg, str(getattr(args, arg)))

            for d in sorted(dict.keys()):
                file.write(d + ' : ' + dict[d] + '\n')
            file.write('\n')
            file.close()


    def store(self, vars, log=False):
        self.vars = self.vars + vars
        if log:
            self.log()

    def log(self):

        vars = self.vars
        file = open(self.dir + '/log.txt', 'a')
        st = ''
        for i in range(len(vars)):
            st += self.var_names[i] + ': ' + self.format[i] % (vars[i]) + ', '
        st += 'time: ' + str(datetime.datetime.now()) + '\n'
        file.write(st)
        file.close()
        self.vars = []

    def store_text(self, text):
        file = open(self.dir + '/log.txt', 'a')
        st = ''
        file.write(text + '\n')
        file.close()


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.lr * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(args, acc_current, acc_best, net, optimizer, epoch, net_best):
    if acc_current > acc_best:
        print('Saving best model...')
        state = {
            'net': net.state_dict(),
            'optim': optimizer.state_dict(),
            'acc': acc_current,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(args.save_dir, 'model_best.pt'))
        acc_best = acc_current
        net_best = copy.deepcopy(net)
    if args.trace:
        print('Saving for trace...')
        state = {
            'net': net.state_dict(),
            'optim': optimizer.state_dict(),
            'acc': acc_current,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(args.save_dir, 'model_' + str(epoch) + '.pt'))
    if epoch + 2 in args.lr_decay_epochs:
        print('Saving for trace2...')
        state = {
            'net': net.state_dict(),
            'optim': optimizer.state_dict(),
            'acc': acc_current,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(args.save_dir, 'model_' + str(epoch) + '.pt'))
    return acc_best, net_best

class ToFloat(object):
    """
    Convert a ``Tensor`` to FloatTensor.
    """

    def __call__(self, inp):
        
        return inp.to(torch.float32)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class CircularList(list):
    """
    Helper for spliting training and validation scans since we have limited data
    """
    def __getitem__(self, x):
        if isinstance(x, slice):
            return [self[x] for x in self._rangeify(x)]

        index = operator.index(x)
        try:
            return super().__getitem__(index % len(self))
        except ZeroDivisionError:
            raise IndexError('list index out of range')

    def _rangeify(self, slice):
        start, stop, step = slice.start, slice.stop, slice.step
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        if step is None:
            step = 1
        return range(start, stop, step)