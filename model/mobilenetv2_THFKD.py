"""
MobileNetV2 implementation used in
<Knowledge Distillation via Route Constrained Optimization>
"""

import torch
import torch.nn as nn
import math
from .AFF import BF
import torch.nn.functional as F
__all__ = ['mobilenetv2_T_w', 'mobile_half']

BN = None


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def forward(self, x):
        t = x
        if self.use_res_connect:
            return t + self.conv(x)
        else:
            return self.conv(x)

class Bottleneck1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False, droprate=0.0):
        super(Bottleneck1, self).__init__()
        mid_planes = int(planes/2)
        # mid_planes = planes
        self.is_last = is_last
        self.conv1 = nn.Conv2d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, inplanes ,kernel_size=1, bias=False)
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

class MobileNetV2(nn.Module):
    """mobilenetV2"""
    def __init__(self, T,
                 feature_dim,
                 input_size=32,
                 width_mult=1.,
                 remove_avg=False,
                 num_branches=3):
        super(MobileNetV2, self).__init__()
        self.remove_avg = remove_avg

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],   #8
            [T, 24, 2, 1],   #12
            [T, 32, 3, 2],   #16
            [T, 64, 4, 2],   #32
        ]
        self.irs = [
            # t, c, n, s
            [T, 96, 3, 1],  # 48
            [T, 160, 3, 2],  #80
            [T, 320, 1, 1],  #160
        ]
        self.num_branches = num_branches
        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.conv1 = conv_bn(3, input_channel, 2)

        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        # self.blocks_branch = nn.ModuleList([])
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))
        fix_inplanes = input_channel
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        for i in range(num_branches):
            branches, updated_input_channel = self._Branch(width_mult, input_channel)
            # self.blocks_branch.append(nn.Sequential(*layers))
            setattr(self, 'block_' + str(i), branches)
            setattr(self, 'classifier3_' + str(i), nn.Linear(self.last_channel, feature_dim),)
        input_channel = updated_input_channel
        self.conv2 = conv_1x1_bn(input_channel, self.last_channel)

        # # building classifier
        # self.classifier = nn.Sequential(
        #     # nn.Dropout(0.5),
        #     nn.Linear(self.last_channel, feature_dim),
        # )

        H = input_size // (32//2)
        self.avgpool = nn.AvgPool2d(H, ceil_mode=True)
        self._initialize_weights()
        print(T, width_mult)

        self.layer_deep = Bottleneck1(160, 160 * 3, stride=1, droprate=0.0)  # droprate=0.0
        self.layer_deep1 = Bottleneck1(160, 160 * 2, stride=1, droprate=0.0)
        self.layer_deep2 = Bottleneck1(160, 160 , stride=1, droprate=0.0)
        # self.layer_deep = Bottleneck1(960, 1088, stride=1, groups=4, is_last=False)  #  droprate=0.0
        # self.layer_deep1 = Bottleneck1(960, 544, stride=1, groups=4, is_last=False)
        self.bf1 = BF(inplanes=self.last_channel, r=4)
        self.bf2 = BF(inplanes=self.last_channel, r=4)
        self.bf3 = BF(inplanes=self.last_channel, r=4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.last_channel, feature_dim)
        self.fc2 = nn.Linear(self.last_channel, feature_dim)
        self.fc3 = nn.Linear(self.last_channel, feature_dim)
        self.to('cuda')

    def _Branch(self, width_mult, input_channel):
        branches = nn.ModuleList([])
        for t, c, n, s in self.irs:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
            branches.append(nn.Sequential(*layers))
        return branches, input_channel

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.blocks)
        return feat_m

    def forward(self, x, is_feat=False, preact=False):
        f_s = []
        out = self.conv1(x)
        out = self.blocks[0](out)
        out = self.blocks[1](out)
        out = self.blocks[2](out)
        out = self.blocks[3](out)
        b_0 = getattr(self, 'block_0')
        out_3 = b_0[0](out)
        out_3 = b_0[1](out_3)
        out_3 = b_0[2](out_3)
        out_3 = self.layer_deep(out_3)
        out_3 = self.conv2(out_3)
        f_s.append(out_3)
        if not self.remove_avg:
            out_3 = self.avgpool(out_3)
        out_3 = out_3.view(out_3.size(0), -1)
        out_3_1 = getattr(self, 'classifier3_0')(out_3)
        pro = out_3_1

        for i in range(1, self.num_branches):
            if i == 1:
                b_1 = getattr(self, 'block_' + str(i))
                temp = b_1[0](out)
                temp = b_1[1](temp)
                temp = b_1[2](temp)
                temp = self.layer_deep1(temp)
                temp = self.conv2(temp)
                f_s.append(temp)
                if not self.remove_avg:
                    temp = self.avgpool(temp)
                temp = temp.view(temp.size(0), -1)
                temp_1 = getattr(self, 'classifier3_' + str(i))(temp)

            elif i == 2:
                b_2 = getattr(self, 'block_' + str(i))
                temp = b_2[0](out)
                temp = b_2[1](temp)
                temp = b_2[2](temp)
                temp = self.layer_deep2(temp)
                temp = self.conv2(temp)
                f_s.append(temp)
                if not self.remove_avg:
                    temp = self.avgpool(temp)
                temp = temp.view(temp.size(0), -1)
                temp_2 = getattr(self, 'classifier3_' + str(i))(temp)

            elif i == 3:
                b_3 = getattr(self, 'block_' + str(i))
                temp = b_3[0](out)
                temp = b_3[1](temp)
                temp = b_3[2](temp)
                temp = self.conv2(temp)
                f_s.append(temp)
                if not self.remove_avg:
                    temp = self.avgpool(temp)
                temp = temp.view(temp.size(0), -1)
                temp_3 = getattr(self, 'classifier3_' + str(i))(temp)

        bf1 = self.bf1(f_s[2], f_s[3])
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
                return [f0, f1_pre, f2_pre, f3_pre, f4], out
            else:
                return pred, bf1, bf2, bf3
        else:
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2_T_w(T, W, feature_dim=100, num_branches=3):
    model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W, num_branches=num_branches)
    return model

# To be consistent with the previous paper (CRD), MobileNetV2 is instantiated by mobile_half
def mobile_half(num_classes, num_branches=3):
    return mobilenetv2_T_w(6, 0.5, num_classes, num_branches=num_branches)

# MobileNetV2x2 is instantiated by mobile_half_double
def mobile_half_double(num_classes, num_branches=3):
    return mobilenetv2_T_w(6, 1.0, num_classes, num_branches=num_branches)

if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)

    net = mobile_half(100)

    feats, logit = net(x, is_feat=True, preact=True)
    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    num_params_stu = (sum(p.numel() for p in net.parameters())/1000000.0)
    print('Total params_stu: {:.3f} M'.format(num_params_stu))
