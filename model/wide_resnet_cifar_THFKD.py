
import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .AFF import BF
"""
Original Author: Wei Yang
"""

__all__ = ['wrn']


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False, droprate=0.0):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes ,kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        # self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.droprate = nn.Dropout(p=droprate)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)

        out = self.droprate(out)
        if self.is_last:
            return out, preact
        else:
            return out

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, num_branches=3):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        self.num_branches = num_branches
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        # self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        for i in range(num_branches):
            setattr(self, 'block3_' + str(i), NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate))
            setattr(self, 'classifier3_' + str(i), nn.Linear(nChannels[3], num_classes))

        self.layer_deep = Bottleneck(nChannels[3], nChannels[3] * 3, stride=1, droprate=0.0)
        self.layer_deep1 = Bottleneck(nChannels[3], nChannels[3] * 2, stride=1, droprate=0.0)
        self.layer_deep2 = Bottleneck(nChannels[3], nChannels[3], stride=1, droprate=0.0)
        self.bf = BF(inplanes=nChannels[3], r=4)
        self.bf2 = BF(inplanes=nChannels[3], r=4)
        self.bf3 = BF(inplanes=nChannels[3], r=4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(nChannels[3], num_classes)
        self.fc2 = nn.Linear(nChannels[3], num_classes)
        self.fc3 = nn.Linear(nChannels[3], num_classes)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        if widen_factor == 1:
            self.query_weight = nn.Linear(nChannels[3], 8, bias=False)
            self.key_weight = nn.Linear(nChannels[3], 8, bias=False)
        elif widen_factor == 2:
            self.query_weight = nn.Linear(nChannels[3], 16, bias=False)
            self.key_weight = nn.Linear(nChannels[3], 16, bias=False)

        self.to('cuda')

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(self.block3)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1

        return [bn1, bn2, bn3]

    def forward(self, x, is_feat=False, preact=False):
        f_s = []
        out = self.conv1(x)
        # f0 = out
        out = self.block1(out)
        # f1 = out
        out = self.block2(out)
        # f2 = out
        # out = self.block3(out)
        # f3 = out
        x_3 = getattr(self, 'block3_0')(out)
        x_3 = self.layer_deep(x_3)
        # f3 = x_3
        x_3 = self.relu(self.bn1(x_3))
        f_s.append(x_3)
        x_3 = F.avg_pool2d(x_3, 8)
        # f4 = x_3
        x_3 = x_3.view(-1, self.nChannels)
        x_3_1 = getattr(self, 'classifier3_0')(x_3)  # B x num_classes
        pro = x_3_1
        # out = self.fc(out)

        for i in range(1, self.num_branches):
            if i == 1:
                temp = getattr(self, 'block3_' + str(i))(out)
                temp = self.layer_deep1(temp)
                f_s.append(temp)
                temp = F.avg_pool2d(temp, 8)
                temp = temp.view(temp.size(0), -1)
                temp_1 = getattr(self, 'classifier3_' + str(i))(temp)

            elif i == 2:
                temp = getattr(self, 'block3_' + str(i))(out)
                temp = self.layer_deep2(temp)
                f_s.append(temp)
                temp = F.avg_pool2d(temp, 8)
                temp = temp.view(temp.size(0), -1)
                temp_2 = getattr(self, 'classifier3_' + str(i))(temp)
            elif i == 3:
                temp = getattr(self, 'block3_' + str(i))(out)
                f_s.append(temp)
                temp = F.avg_pool2d(temp, 8)
                temp = temp.view(temp.size(0), -1)
                temp_3 = getattr(self, 'classifier3_' + str(i))(temp)

        bf1 = self.bf(f_s[2], f_s[3])
        f_s.append(bf1)
        bf1 = self.avgpool(bf1)  # 128 x 64 x 1 x 1
        bf1 = bf1.view(bf1.size(0), -1)  # 128 x 64
        bf1 = self.fc1(bf1)  # 128x100

        bf2 = self.bf2(f_s[1], f_s[4])
        f_s.append(bf2)
        bf2 = self.avgpool(bf2)
        bf2 = bf2.view(bf2.size(0), -1)
        bf2 = self.fc2(bf2)

        bf3 = self.bf3(f_s[0], f_s[5])
        bf3 = self.avgpool(bf3)
        bf3 = bf3.view(bf3.size(0), -1)
        bf3 = self.fc3(bf3)

        temp_1 = temp_1.unsqueeze(-1)
        temp_2 = temp_2.unsqueeze(-1)
        temp_3 = temp_3.unsqueeze(-1)
        pro = pro.unsqueeze(-1)
        pred = torch.cat([pro, temp_1], -1)
        pred = torch.cat([pred, temp_2], -1)
        pred = torch.cat([pred, temp_3], -1)

        if is_feat:
            if preact:
                f1 = self.block2.layer[0].bn1(f1)
                f2 = self.block3.layer[0].bn1(f2)
                f3 = self.bn1(f3)
                return [f0, f1, f2, f3, f4], out
            else:
                return pred, bf1, bf2, bf3
        else:
            return pro, proj_q, proj_k, temps_out


def wrn_GL(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model


def wrn_40_2(**kwargs):
    model = WideResNet(depth=40, widen_factor=2, num_branches=3, **kwargs)
    return model


def wrn_40_1(**kwargs):
    model = WideResNet(depth=40, widen_factor=1, num_branches=3, **kwargs)
    return model


def wrn_16_2(**kwargs):
    model = WideResNet(depth=16, widen_factor=2, num_branches=3, **kwargs)
    return model


def wrn_16_1(**kwargs):
    model = WideResNet(depth=16, widen_factor=1, num_branches=3, **kwargs)
    return model


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = wrn_40_2(num_classes=100)
    feats, logit = net(x, is_feat=True, preact=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')
