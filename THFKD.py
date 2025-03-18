import pdb
import time
import argparse
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms
import ssl

import utils

ssl._create_default_https_context = ssl._create_unverified_context
parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar100',
                    )
parser.add_argument('--model', '-a', default='resnet32',
                    )
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.05,
                    help='learning rate')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay ratio')
parser.add_argument('--lr_adjust_step', default=[150, 225], type=int, nargs='+',
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')
parser.add_argument('--suffix', type=str, default='lr0.05_sc0._branch4_T=5',
                    help='label')
parser.add_argument('--test', action='store_true', default=False,
                    help='test')
parser.add_argument('--resume', type=str, default='',
                    help='resume')
parser.add_argument('--teacher', type=str, default='resnet110',
                    help='teacher model')
parser.add_argument('--teacher-weight', type=str, default='checkpoints_teacher_baseline/cifar100_resnet110__baseline1_best.pt',
                    help='teacher model weight path')
parser.add_argument('--kd-loss-weight', type=float, default=5.0,
                    help='review kd loss weight')
parser.add_argument('--kd-warm-up', type=float, default=20.0,
                    help='feature konwledge distillation loss weight warm up epochs')

parser.add_argument('--use-kl', action='store_true', default=False,
                    help='use kl kd loss')
parser.add_argument('--kl-loss-weight', type=float, default=5.0,
                    help='kl konwledge distillation loss weight')
parser.add_argument('-T', type=float, default=5.0,
                    help='knowledge distillation loss temperature')
parser.add_argument('--ce-loss-weight', type=float, default=1.0,
                    help='cross entropy loss weight')

parser.add_argument('--loss', default='KL', type=str, help = 'Define the loss between student output and group output: default(KL_Loss)')
parser.add_argument('--num_branches', default=4, type=int, help = 'Input the number of branches: default(4)')
parser.add_argument('--length', default=80, type=float, help='length ratio: default(80)')
parser.add_argument('--start_consistency', default=0., type=float, help = 'Input the start consistency rate: default(0.5)')
parser.add_argument('--alpha', default=1.0, type=float, help = 'Input the relative rate: default(1.0)')

args = parser.parse_args()
assert torch.cuda.is_available()

cudnn.deterministic = True
cudnn.benchmark = False
if args.seed == 0:
    args.seed = np.random.randint(1000)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)


from util.misc import *
from util.kd import DistillKL

from model.resnet import ResNet18, ResNet50
from model.resnet_cifar import build_resnet_backbone, build_resnetx4_backbone
from model.resnetv2_cifar import ResNet50
from model.vgg import build_vgg_backbone
from model.vgg_number_branch import build_GLKD_vgg_backbone
#from model.mobilenetv2 import mobile_half
from model.shufflenetv1_BranchFusion import ShuffleV1
from model.shufflenetv2 import ShuffleV2
from model.wide_resnet_cifar_GL_BranchFusion_number_branch import wrn_GL
from model.wide_resnet_cifar import wrn
from model.wide_resnet import WideResNet
from model.reviewkd import build_review_kd, hcl
from model.resnet_cifar_GLx4_F_Deep2 import build_GLKDX4_backbone
from model.resnet_cifar_GL_BranchFusion_number_branch import build_GLKD_backbone, build_GLKDx4_backbone
from model.mobilenetv2_BranchFusion import mobile_half

if not os.path.exists('logs_v4'):
    os.makedirs('logs_v4')

test_id = args.dataset + '_' + args.model + '_' + args.teacher + '_' + args.suffix
filename = 'logs_v5/' + test_id + '.txt'
logger = Logger(args=args, filename=filename)
print(args)

# Image Preprocessing
normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

# dataset
if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='data/',train=True,transform=train_transform,download=True)
    test_dataset = datasets.CIFAR10(root='data/',train=False,transform=test_transform,download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(root='D:/rzh/datas',train=True,transform=train_transform,download=False)
    test_dataset = datasets.CIFAR100(root='D:/rzh/datas',train=False,transform=test_transform,download=False)
else:
    assert False
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                           shuffle=True, pin_memory=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=args.batch_size,
                                          shuffle=False,pin_memory=True,num_workers=1)

# teacher model
if 'x4' in args.teacher:
    teacher = build_resnetx4_backbone(depth = int(args.teacher[6:-2]), num_classes=num_classes)
elif 'resnet' in args.teacher:
    teacher = build_resnet_backbone(depth = int(args.teacher[6:]), num_classes=num_classes)
elif 'ResNet50' in args.teacher:
    teacher = ResNet50(num_classes=num_classes)
elif 'vgg' in args.teacher:
    teacher = build_vgg_backbone(depth = int(args.teacher[3:]), num_classes=num_classes, batch_norm=True)
elif 'mobile' in args.teacher:
    teacher = mobile_half(num_classes=num_classes)
elif 'wrn' in args.teacher:
    teacher = wrn(depth = int(args.teacher[4:6]), widen_factor = int(args.teacher[-1:]), num_classes=num_classes)
elif args.teacher == '':
    teacher = None
else:
    assert False
if teacher is not None:
    load_teacher_weight(teacher, args.teacher_weight, args.teacher)

# model

if 'x4' in args.model:
    cnn = build_GLKDx4_backbone(depth = int(args.model[6:-2]), num_classes=num_classes, num_branches=args.num_branches)
elif 'resnet' in args.model:
    cnn = build_GLKD_backbone(depth = int(args.model[6:]), num_classes=num_classes, num_branches=args.num_branches)
elif 'ResNet50' in args.model:
    cnn = ResNet50(num_classes=num_classes)
elif 'vgg' in args.model:
    cnn = build_GLKD_vgg_backbone(depth = int(args.model[3:]), num_classes=num_classes, batch_norm=True, num_branches=args.num_branches)
elif 'mobile' in args.model:
    cnn = mobile_half(num_classes=num_classes, num_branches=args.num_branches)
elif 'shufflev1' in args.model:
    cnn = ShuffleV1(num_classes=num_classes, num_branches=args.num_branches)
elif 'shufflev2' in args.model:
    cnn = ShuffleV2(num_classes=num_classes, num_branches=args.num_branches)
elif 'wrn' in args.model:
    cnn = wrn_GL(depth = int(args.model[4:6]), widen_factor = int(args.model[-1:]), num_classes=num_classes, num_branches=args.num_branches)
elif args.model == 'wideresnet':
    cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)
else:
    assert False
if 'shuffle' in args.model or 'mobile' in args.model:
    args.lr = 0.02

criterion = nn.CrossEntropyLoss()
if args.loss == "KL":
    criterion_T = utils.KL_Loss(args.T).cuda()
elif args.loss == "CE":
    criterion_T = utils.CE_Loss(args.T).cuda()

accuracy_a = utils.accuracy
wd = args.wd
lr = args.lr
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr,
                                momentum=0.9, nesterov=True, weight_decay=args.wd)
# 指定要创建的文件夹名称
folder_name = './checkpoints_ReviewKD'

# 使用os模块中的makedirs()函数创建文件夹
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

def get_current_consistency_weight(current, rampup_length = args.length):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

# test
def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    correct_xm1 = 0.
    correct_xm2 = 0.
    correct_stu0 = 0.
    correct_stu1 = 0.
    correct_stu2 = 0.
    correct_stu3 = 0.
    correct_stu4 = 0.
    correct_bf1 = 0.
    correct_bf2 = 0.
    correct_bf3 = 0.
    correct_bf4 = 0.
    correct_xm_e = 0.
    correct_bf_e = 0.

    for images, labels in loader:
        images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            pred, bf1, bf2, bf3 = cnn(images, is_feat=True)

        bf_1 = torch.max(bf1.data, 1)[1]
        bf_2 = torch.max(bf2.data, 1)[1]
        bf_3 = torch.max(bf3.data, 1)[1]
        # bf_4 = torch.max(bf4.data, 1)[1]
        bf_e = (bf1 + bf2 +bf3) // 4
        bf_e = torch.max(bf_e.data, 1)[1]
        stu_1 = torch.max(pred[:, :, 1].data, 1)[1]
        stu_0 = torch.max(pred[:, :, 0].data, 1)[1]
        stu_2 = torch.max(pred[:, :, 2].data, 1)[1]
        stu_3 = torch.max(pred[:, :, 3].data, 1)[1]
        # stu_4 = torch.max(pred[:, :, 4].data, 1)[1]
        # xm_e = torch.cat([x_m1,x_m2], dim=2)
        # xm_e = torch.max(torch.mean(xm_e, dim=2).data, 1)[1]
        # x_m1 = torch.max(torch.mean(x_m1, dim=2).data, 1)[1]
        # x_m2 = torch.max(torch.mean(x_m2, dim=2).data, 1)[1]
        # xstu = torch.max(x_stu.data, 1)[1]
        # x_stu = x_stu.unsqueeze(-1)
        # pred = torch.cat([pred, x_stu], -1)
        pred = torch.mean(pred, dim=2)
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        # correct_xm1 += (x_m1 == labels).sum().item()
        # correct_xm2 += (x_m2 == labels).sum().item()
        correct_stu0 += (stu_0 == labels.data).sum().item()
        correct_stu1 += (stu_1 == labels.data).sum().item()
        correct_stu2 += (stu_2 == labels.data).sum().item()
        correct_stu3 += (stu_3 == labels.data).sum().item()
        # correct_stu4 += (stu_4 == labels.data).sum().item()
        correct_bf1 += (bf_1 == labels.data).sum().item()
        correct_bf2 += (bf_2 == labels.data).sum().item()
        correct_bf3 += (bf_3 == labels.data).sum().item()
        # correct_bf4 += (bf_4 == labels.data).sum().item()
        # correct_xm_e += (xm_e == labels.data).sum().item()
        correct_bf_e += (bf_e == labels.data).sum().item()
    stu1_acc = correct_stu1 / total
    stu0_acc = correct_stu0 / total
    stu2_acc = correct_stu2 / total
    stu3_acc = correct_stu3 / total
    # stu4_acc = correct_stu4 / total
    val_acc = correct / total
    # xm1_acc = correct_xm1 / total
    # xm2_acc = correct_xm2 / total
    bf1_acc = correct_bf1 / total
    bf2_acc = correct_bf2 / total
    bf3_acc = correct_bf3 / total
    # bf4_acc = correct_bf4 / total
    bfe_acc = correct_bf_e / total
    # xme_acc = correct_xm_e / total

    cnn.train()
    return val_acc, stu1_acc, stu0_acc, stu2_acc, stu3_acc, bf1_acc, bf2_acc, bf3_acc,bfe_acc

scheduler = MultiStepLR(cnn_optimizer, milestones=args.lr_adjust_step, gamma=0.1)
start_epoch = 0
if args.test:
    cnn.load_state_dict(torch.load(args.resume))
    scheduler.step(start_epoch - 1)
    print(test(test_loader))
    exit()

# train

best_acc = 0.0
st_time = time.time()

if __name__ == '__main__':
    for epoch in range(args.epochs):

        consistency_epoch = args.start_consistency * args.epochs
        if epoch < consistency_epoch:
            consistency_weight = 1
        else:
            consistency_weight = get_current_consistency_weight(epoch - consistency_epoch, args.length)
        correct = 0.
        total = 0.
        loss_avg = {}
        cnt_ft = {}
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()

            losses = {}
            with torch.no_grad():
                t_pred = teacher(images, is_feat=False, preact=False)
            # teachers = []
            # teacher_p = t_pred
            # teacher_p = teacher_p.unsqueeze(-1)
            # for i in range(1, args.num_branches - 1):
            #     teachers.append(t_pred)
            # for i in range(0, args.num_branches - 2):
            #     teacher_1 = teachers[i]
            #     teacher_1 = teacher_1.unsqueeze(-1)
            #     teacher_p = torch.cat([teacher_p, teacher_1], -1)

            # 使用模型进行前向传播，返回输出、交互信息和学生输出。
            pred, bf1, bf2, bf3 = cnn(images, is_feat=True, preact=False)  # 用学生模型预测训练数据

            # energy = torch.bmm(proj_q, proj_k.permute(0, 2, 1))
            # # attention = F.softmax(energy, dim=-1)
            # # x_m = torch.bmm(teacher_p, attention.permute(0, 2, 1))

            loss_true = 0  # 初始化分支模型的损失值
            loss_group = 0  # 初始化分组模型的损失值
            loss_KD = 0
            loss_bf = 0
            loss_predE = 0
            # 遍历从1到3的三个“peer”分支。
            for i in range(args.num_branches):
                loss_true += criterion(pred[:, :, i], labels)  # 计算当前分支的交叉熵损失。
                loss_KD += criterion_T(pred[:, :, i], t_pred)
                loss_predE += criterion_T(pred[:, :, i], torch.mean(pred, dim=2))
                loss_group += criterion_T(pred[:, :, i], bf3)
            # loss_group += criterion_T(bf1, bf2)
            # loss_KD += criterion_T(pred[:, :, 2], torch.mean(pred, dim=2))~
            loss_bf = criterion(bf1, labels) + criterion(bf2, labels) + criterion(bf3, labels)
            # loss_KD = criterion_T(bf1, t_pred) + criterion_T(bf2, t_pred)
            # loss_KD = criterion_T(bf1, t_pred) + criterion_T(bf2, t_pred)
            # ensemble = torch.cat([pred, x_m], -1)
            # 计算总损失，包括交叉熵损失、学生和教师输出之间的知识蒸馏损失、组内知识蒸馏损失和组间知识蒸馏损失。
            loss = loss_true + args.alpha * consistency_weight * (
              loss_KD + loss_group +loss_predE)


            losses['loss_true'] = loss_true
            losses['loss_group'] = loss_group
            losses['loss_KD'] = loss_KD
            num_params_stu = (sum(p.numel() for p in cnn.parameters()) / 1000000.0),
            cnn_optimizer.zero_grad()
            loss.backward()
            cnn_optimizer.step()

            for key in losses:
                if not key in loss_avg:
                    loss_avg[key] = AverageMeter()
                else:
                    loss_avg[key].update(losses[key])

            # x_stu = x_stu.unsqueeze(-1)
            # pred = torch.cat([pred, x_stu], -1)
            pred = torch.mean(pred, dim=2)
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            # Calculate running average of accuracy
            accuracy = correct / total

        test_acc, stu1_acc, stu0_acc, stu2_acc, stu3_acc, bf1_acc, bf2_acc, bf3_acc, bfe_acc = test(test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(cnn.state_dict(), 'checkpoints_ReviewKD/' + test_id + '_best.pt')
        scheduler.step()
        loss_avg = {k: loss_avg[k].val for k in loss_avg}
        row = { 'epoch': str(epoch),
                'train_acc': '%.2f'%(accuracy*100),
                'test_acc': '%.2f'%(test_acc*100),
                'best_acc': '%.2f'%(best_acc*100),
                # 'xm1_acc': '%.2f' % (xm1_acc * 100),
                # 'xm2_acc': '%.2f' % (xm2_acc * 100),
                # 'xme_acc': '%.2f' % (xme_acc * 100),
                'bfe_acc': '%.2f' % (bfe_acc * 100),
                'stu0_acc': '%.2f' % (stu0_acc * 100),
                'stu1_acc': '%.2f' % (stu1_acc * 100),
                'stu2_acc': '%.2f' % (stu2_acc * 100),
                'stu3_acc': '%.2f' % (stu3_acc * 100),
                # 'stu4_acc': '%.2f' % (stu4_acc * 100),
                'bf1_acc': '%.2f' % (bf1_acc * 100),
                'bf2_acc': '%.2f' % (bf2_acc * 100),
                'bf3_acc': '%.2f' % (bf3_acc * 100),
                # 'bf4_acc': '%.2f' % (bf4_acc * 100),
                'lr': '%.5f'%(lr),
                'loss': '%.5f'%(sum(loss_avg.values())),
                'params': '%.5f'%(num_params_stu),
                }
        loss_avg = {k: '%.5f'%loss_avg[k] for k in loss_avg}
        row.update(loss_avg)
        row.update({
                'time': format_time(time.time()-st_time),
                'eta': format_time((time.time()-st_time)/(epoch+1)*(args.epochs-epoch-1)),
                })
        print(row)
        logger.writerow(row)


# # 指定要创建的文件夹名称
# folder_name = './checkpoint'
#
# # 使用os模块中的makedirs()函数创建文件夹
# if not os.path.exists(folder_name):
#     os.makedirs(folder_name)
# torch.save(cnn.state_dict(), 'checkpoints_baseline/' + test_id + '.pt')
logger.close()


