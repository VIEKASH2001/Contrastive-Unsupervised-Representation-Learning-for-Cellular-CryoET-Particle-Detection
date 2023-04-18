import math

import torch
import torch.nn as nn
from models_simclr.backbones import *
from models_simclr.cifar_resnet import resnet18 as cifar_resnet18
import torch.nn.functional as F


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048, dataset='CIFAR10'):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        if dataset in ['CIFAR10', 'CIFAR100']:
            # cfg.logger.info('CIFAR Identity layer')
            self.layer2 = nn.Identity()
        else:
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


class prediction_MLP_transformer(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.mhsa = MHSA(512, width=1, height=1, heads=4)

        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)




        x = x.unsqueeze(2).unsqueeze(2)
        x = self.mhsa(x)
        x = x[:, :, 0, 0]
        x = self.layer2(x)
        return x


class prediction_MLP_flip(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        # self.layer1 = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=True)
        # )

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.layer_w1 = nn.Linear(1, in_dim * hidden_dim)
        self.layer_b1 = nn.Linear(1, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x, fliplbl):
        bs = len(x)

        w1 = self.layer_w1(fliplbl)
        # w1 = self.bn1(w1)

        b1 = self.layer_b1(fliplbl)

        w1 = w1.view(bs, self.hidden_dim, self.in_dim)

        # w1 = w1/math.sqrt(self.in_dim)

        x = x.unsqueeze(2)
        y = w1 @ x

        y = y.squeeze(2)

        y = y + b1

        y = self.bn1(y)

        y = F.relu(y)

        # x = self.layer1(x)
        x = self.layer2(y)
        return x


class prediction_MLP_v2(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        # self.layer1 = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=True)
        # )

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.fctest = nn.Linear(in_dim, hidden_dim)
        self.layer_w1 = nn.Linear(2, in_dim * hidden_dim)

        self.layer_w1.weight.data /= 45
        self.layer_w1.bias.data.zero_()

        # print('wtf!')
        # exit()

        # a = self.layer_w1.weight.data.sum(1)
        #
        # a = a.view(in_dim, hidden_dim)
        #
        # b = self.fctest.weight
        #
        # print(b.shape)
        # print(a.shape)
        #
        #
        # print(b.std(0).mean())
        # print(a.std(1).mean())
        #
        #
        # exit()

        self.layer_b1 = nn.Linear(2, hidden_dim)
        self.layer_b1.weight.data.zero_()
        self.layer_b1.bias.data.zero_()

        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.layer2 = nn.Linear(hidden_dim, out_dim)

        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x, d):
        bs = len(x)

        w1 = self.layer_w1(d)
        # w1 = self.bn1(w1)
        b1 = self.layer_b1(d)
        w1 = w1.view(bs, self.hidden_dim, self.in_dim)
        # w1 = w1/math.sqrt(self.in_dim)
        x = x.unsqueeze(2)
        y = w1 @ x
        y = y.squeeze(2)
        y = y + b1
        y = self.bn1(y)
        y = F.relu(y)

        # x = self.layer1(x)
        x = self.layer2(y)
        return x


# class prediction_MLP_flip2(nn.Module):
#     def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
#         super().__init__()
#         ''' page 3 baseline setting
#         Prediction MLP. The prediction MLP (h) has BN applied
#         to its hidden fc layers. Its output fc does not have BN
#         (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers.
#         The dimension of h’s input and output (z and p) is d = 2048,
#         and h’s hidden layer’s dimension is 512, making h a
#         bottleneck structure (ablation in supplement).
#         '''
#         # self.layer1 = nn.Sequential(
#         #     nn.Linear(in_dim, hidden_dim),
#         #     nn.BatchNorm1d(hidden_dim),
#         #     nn.ReLU(inplace=True)
#         # )
#
#         self.in_dim = in_dim
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#
#         self.layer_w1 = nn.Linear(1, in_dim * hidden_dim)
#         self.layer_b1 = nn.Linear(1, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#
#         # self.layer2 = nn.Linear(hidden_dim, out_dim)
#
#         self.layer_w2 = nn.Linear(1, hidden_dim * out_dim)
#         self.layer_b2 = nn.Linear(1, out_dim)
#         self.bn2 = nn.BatchNorm1d(out_dim)
#
#         """
#         Adding BN to the output of the prediction MLP h does not work
#         well (Table 3d). We find that this is not about collapsing.
#         The training is unstable and the loss oscillates.
#         """
#
#     def forward(self, x, fliplbl):
#         bs = len(x)
#
#         w1 = self.layer_w1(fliplbl)
#         b1 = self.layer_b1(fliplbl)
#         w1 = w1.view(bs, self.hidden_dim, self.in_dim)
#
#         x = x.unsqueeze(2)
#         y = w1 @ x
#
#         y = y.squeeze(2)
#
#         # y = y + b1
#
#         y = self.bn1(y)
#
#         y = F.relu(y)
#
#         # x = self.layer1(x)
#         # x = self.layer2(y)
#
#         w2 = self.layer_w2(fliplbl)
#         b2 = self.layer_b2(fliplbl)
#         w2 = w2.view(bs, self.out_dim, self.hidden_dim)
#
#         y = y.unsqueeze(2)
#         y = w2 @ y
#         y = y.squeeze(2)
#         # y = y+b2
#         y = self.bn2(y)
#         return y


class prediction_MLP_v3(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        # self.layer1 = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=True)
        # )

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.fctest = nn.Linear(in_dim, hidden_dim)
        self.layer_w1 = nn.Linear(4, in_dim * hidden_dim)

        self.layer_w1.weight.data /= 45
        self.layer_w1.bias.data.zero_()

        # print('wtf!')
        # exit()

        # a = self.layer_w1.weight.data.sum(1)
        #
        # a = a.view(in_dim, hidden_dim)
        #
        # b = self.fctest.weight
        #
        # print(b.shape)
        # print(a.shape)
        #
        #
        # print(b.std(0).mean())
        # print(a.std(1).mean())
        #
        #
        # exit()

        self.pre1 = nn.Linear(2, 32)
        self.pre2 = nn.Linear(32, 32)
        self.pre3 = nn.Linear(32, 4)

        self.layer_b1 = nn.Linear(4, hidden_dim)
        self.layer_b1.weight.data.zero_()
        self.layer_b1.bias.data.zero_()

        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.layer2 = nn.Linear(hidden_dim, out_dim)

        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x, d):
        bs = len(x)

        # prepare transformation data
        pre = nn.functional.relu(self.pre1(d))
        pre = nn.functional.relu(self.pre2(pre))
        pre = nn.functional.relu(self.pre3(pre))

        w1 = self.layer_w1(pre)
        # w1 = self.bn1(w1)
        b1 = self.layer_b1(pre)
        w1 = w1.view(bs, self.hidden_dim, self.in_dim)
        # w1 = w1/math.sqrt(self.in_dim)
        x = x.unsqueeze(2)
        y = w1 @ x
        y = y.squeeze(2)
        y = y + b1
        y = self.bn1(y)
        y = F.relu(y)

        # x = self.layer1(x)
        x = self.layer2(y)
        return x


def get_backbone(cfg, castrate=True):
    backbone = cfg.backbone
    if cfg.set in ['CIFAR10', 'CIFAR100']:
        backbone = 'cifar_resnet18'
    backbone = eval(f"{backbone}(zero_init_residual=True)")
    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


class SimSiam(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = get_backbone(cfg)
        self.projector = projection_MLP(cfg, self.backbone.output_dim, hidden_dim=cfg.emb_dim, out_dim=cfg.emb_dim)

        # self.encoder = nn.Sequential(  # f encoder
        #     self.backbone,
        #     self.projector
        # )
        self.predictor = prediction_MLP(in_dim=cfg.emb_dim, out_dim=cfg.emb_dim)

    def forward(self, x1, x2):
        bb = self.backbone
        f = self.projector
        h = self.predictor

        bb1, bb2 = bb(x1), bb(x2)
        z1, z2 = f(bb1), f(bb2)
        p1, p2 = h(z1), h(z2)
        # L = D(p1, z2) / 2 + D(p2, z1) / 2
        return ((bb1, z1, p1), (bb2, z2, p2))
