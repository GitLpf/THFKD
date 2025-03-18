from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import math
from .AFF import BF
import torch
__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False,droprate = 0.0):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out
        else:
            return out


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
            return out
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, depth, num_filters, block_name='BasicBlock', num_classes=10, num_branches=3):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.num_branches = num_branches
        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        # self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        fix_inplanes = self.inplanes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for i in range(num_branches):
            setattr(self, 'layer3_' + str(i), self._make_layer(block,  num_filters[3], n, stride=2))
            self.inplanes = fix_inplanes
            setattr(self, 'classifier3_' + str(i), nn.Linear(num_filters[3] * block.expansion, num_classes))

        self.layer_deep = Bottleneck(num_filters[3], num_filters[3] * 3, stride=1, droprate=0.0)
        self.layer_deep1 = Bottleneck(num_filters[3], num_filters[3] * 2, stride=1, droprate=0.0)
        self.layer_deep2 = Bottleneck(num_filters[3], num_filters[3], stride=1, droprate=0.0)

        self.bf1 = BF(inplanes = num_filters[3], r=4)
        self.bf2 = BF(inplanes=num_filters[3], r=4)
        self.bf3 = BF(inplanes=num_filters[3], r=4)
        self.fc1 = nn.Linear(num_filters[3], num_classes)
        self.fc2 = nn.Linear(num_filters[3], num_classes)
        self.fc3 = nn.Linear(num_filters[3], num_classes)
        # self.query_weight = nn.Linear(64, 8, bias=False)
        # self.key_weight = nn.Linear(64, 8, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.to('cuda')

    def _make_layer(self, block, planes, blocks, stride=1, droprate=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1), droprate=droprate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks - 1)))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3]

    def forward(self, x, is_feat=False, preact=False):
        f_s = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        f0 = x

        x = self.layer1(x)  # 32x32
        f1 = x
        x = self.layer2(x)  # 16x16
        f2 = x
        x_3 = getattr(self, 'layer3_0')(x)
        x_3 = self.layer_deep(x_3)
        #x_3 = self.layer_deep(x_3)
        f_s.append(x_3)
        x_3 = self.avgpool(x_3)
        x_3 = x_3.view(x_3.size(0), -1)  # B x 64
        x_3_1 = getattr(self, 'classifier3_0')(x_3)  # B x num_classes
        pro = x_3_1

        for i in range(1, self.num_branches):
            if i == 1:
                temp = getattr(self, 'layer3_' + str(i))(x)
                temp = self.layer_deep1(temp)
                #temp = self.layer_deep1(temp)
                f_s.append(temp)
                temp = self.avgpool(temp)
                temp = temp.view(temp.size(0), -1)
                temp_1 = getattr(self, 'classifier3_' + str(i))(temp)

            elif i == 2:
                temp = getattr(self, 'layer3_' + str(i))(x)
                temp = self.layer_deep2(temp)
                #temp = self.layer_deep2(temp)
                f_s.append(temp)
                temp = self.avgpool(temp)
                temp = temp.view(temp.size(0), -1)
                temp_2 = getattr(self, 'classifier3_' + str(i))(temp)

            elif i == 3:
                temp = getattr(self, 'layer3_' + str(i))(x)
                f_s.append(temp)
                temp = self.avgpool(temp)
                temp = temp.view(temp.size(0), -1)
                temp_3 = getattr(self, 'classifier3_' + str(i))(temp)

        bf1 = self.bf1(f_s[2], f_s[3])
        f_s.append(bf1)
        bf1 = self.avgpool(bf1)  # 128 x 64 x 1 x 1
        bf1 = bf1.view(bf1.size(0), -1) # 128 x 64
        bf1 = self.fc1(bf1) # 128x100

        bf2 = self.bf2(f_s[1], f_s[4])
        f_s.append(bf2)
        bf2 = self.avgpool(bf2)
        bf2 = bf2.view(bf2.size(0), -1)
        bf2 = self.fc2(bf2)

        bf3 = self.bf3(f_s[0], f_s[5])
        f_s.append(bf3)
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

                return [f0, f1_pre, f2_pre], x  
            else:
                return pred, bf1, bf2, bf3
        else:
            return pro,


def resnet8(**kwargs):
    return ResNet(8, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet14(**kwargs):
    return ResNet(14, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet20(**kwargs):
    return ResNet(20, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet32(**kwargs):
    return ResNet(32, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet44(**kwargs):
    return ResNet(44, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet56(**kwargs):
    return ResNet(56, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet110(**kwargs):
    return ResNet(110, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet8x4(**kwargs):
    return ResNet(8, [32, 64, 128, 256], 'basicblock', **kwargs)


def resnet32x4(**kwargs):
    return ResNet(32, [32, 64, 128, 256], 'basicblock', **kwargs)

def build_GLKD_backbone(depth=20, num_classes = 10, num_branches=3):
    if depth == 20:
        return resnet20(num_classes=num_classes, num_branches=num_branches)
    elif depth == 32:
        return resnet32(num_classes=num_classes, num_branches=num_branches)
    elif depth == 56:
        return resnet56(num_classes=num_classes, num_branches=num_branches)
    elif depth == 110:
        return resnet110(num_classes=num_classes, num_branches=num_branches)

def build_GLKDx4_backbone(depth=20, num_classes = 10, num_branches=3):
    if depth == 8:
        return resnet8x4(num_classes=num_classes, num_branches=num_branches)
    elif depth == 32:
        return resnet32x4(num_classes=num_classes, num_branches=num_branches)


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = resnet8x4(num_classes=20)
    feats, logit = net(x, is_feat=True, preact=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')
