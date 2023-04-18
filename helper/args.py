import argparse
import os


def get_params(train=False):
    parser = argparse.ArgumentParser(description='PyTorch Training')

    # general params
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset can be cifar10, svhn')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for training the model')
    parser.add_argument('--trial', type=str, default='test',
                        help='name for the experiment')
    parser.add_argument('--chkpt', type=str, default='',
                        help='checkpoint for resuming the training')

    # dense contrastive params
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='feature dim for the output of the backbone')

    # training params

    parser.add_argument('--bs', type=int, default=256,
                        help='batchsize')
    parser.add_argument('--nw', type=int, default=0,
                        help='number of workers for the dataloader')
    parser.add_argument('--save_dir', type=str, default='./logs',
                        help='directory to log and save checkpoint')
    parser.add_argument('--lr', default=1, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd')
    parser.add_argument('--epochs', default=1200, type=int, help='training epochs')

    args = parser.parse_args()

    if train and not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args
