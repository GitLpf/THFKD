import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def value(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

def load_json_to_dict(json_path):
    """Loads json file to dict 

    Args:
        json_path: (string) path to json file
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim= 1, largest= True, sorted= True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class kd_loss_fn(nn.Module):
    def __init__(self, num_classes, args):
        super(kd_loss_fn, self).__init__()
        self.num_classes = num_classes
        self.alpha = args.alpha
        self.T = args.temperature
        
    def forward(self, output_batch, labels_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # labels_batch  -> B, LongTensor
        # teacher_outputs -> B X num_classes
        
        # torch.save(output_batch, './output_batch')
        # torch.save(labels_batch,'./labels_batch')
        # torch.save(teacher_outputs,'./teacher_outputs')
    
        # zero-mean, and small value
        # teacher_outputs = (teacher_outputs - torch.mean(teacher_outputs, dim=1).view(-1,1))/100.0
        # output_batch = (output_batch - torch.mean(output_batch, dim=1).view(-1,1))/100.0
    
        teacher_outputs=F.softmax(teacher_outputs/self.T,dim=1)
        output_batch=F.log_softmax(output_batch/self.T,dim=1)    
    
        #CE_teacher = -torch.sum(torch.sum(torch.mul(teacher_outputs,output_batch)))/teacher_outputs.size(0)
        #CE_teacher.requires_grad_(True)
        KL_teacher = nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs) * self.T
        CE_true = nn.CrossEntropyLoss()(output_batch, labels_batch) 
        loss = KL_teacher * self.alpha + CE_true * (1 - self.alpha)    
        return loss

       

class Att_Loss(nn.Module):
    def __init__(self, temperature = 1, loss = 'CE'):
        super(Att_Loss, self).__init__()        
        self.T = temperature
        self.loss = loss
    def forward(self, output_batch, labels_batch, attention):
        # output_batch  -> B X num_classes X num_student
        # attention     -> B X num_student X num_student
        # teacher_outputs -> B X num_classes
        
        batch_size, num_classes, num_student = output_batch.size()
        labels_batch = labels_batch.view(-1,1).repeat(1, num_student)  # B X num_student
        loss_true = nn.CrossEntropyLoss()(output_batch, labels_batch) * num_student
        # teacher_outputs = teacher_outputs.repeat(args.num_student, 1, 1).view(-1, num_classes, args.num_student) # B X num_classes X num_student    
        
        attention_label = torch.bmm(output_batch, attention.permute(0,2,1))     # B X num_classes X num_student
        
        if self.loss == 'CE':
            output_batch = F.log_softmax(output_batch/self.T, dim=1)
            attention_outputs = F.softmax(attention_label/self.T, dim=1)            # B X num_classes X num_student
            loss_att = -torch.sum(torch.mul(output_batch, attention_outputs))/batch_size
        elif self.loss == 'MSE':
            # calculate the average distance between attention and identity
            output_batch = F.softmax(output_batch, dim=1)    
            attention_outputs = F.softmax(attention_label, dim=1)            # B X num_classes X num_student
            loss_att = torch.sum((output_batch - attention_outputs) ** 2) / batch_size 
        # calculate the log angle 
        identity = torch.eye(num_student).reshape(1, num_student, num_student).repeat(batch_size, 1, 1).cuda()
        # calculate the average distance between attention and identity
        scale = torch.Tensor([batch_size * num_student]).sqrt().cuda()
        dist_att = torch.norm(attention - identity, p='fro')/scale
        # dist_p = torch.norm(output_batch, p='fro')
        # angle = torch.log(loss_att) - torch.log(dist) - torch.log(dist_p)
        # angle = loss_att/(dist * dist_p)
        return loss_true, loss_att, dist_att
        
class KL_Loss(nn.Module):
    def __init__(self, temperature = 1):
        super(KL_Loss, self).__init__()
        self.T = temperature
    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes            
        # teacher_outputs -> B X num_classes
        
        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)
        
        output_batch = F.log_softmax(output_batch/self.T, dim = 1)    
        teacher_outputs = F.softmax(teacher_outputs/self.T, dim = 1) + 10**(-7)
    
        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs) 
        
        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss

        
class CE_Loss(nn.Module):
    def __init__(self, temperature = 1):
        super(CE_Loss, self).__init__()
        self.T = temperature
    def forward(self, output_batch, teacher_outputs):
    
        # output_batch      -> B X num_classes 
        # teacher_outputs   -> B X num_classes
        
        output_batch = F.log_softmax(output_batch/self.T,dim=1)    
        teacher_outputs = F.softmax(teacher_outputs/self.T,dim=1)
        
        # Same result CE-loss implementation torch.sum -> sum of all element
        loss = -self.T*self.T*torch.sum(torch.mul(output_batch, teacher_outputs))/teacher_outputs.size(0)
        
        return loss

class MSE_Loss(nn.Module):
    def __init__(self):
        super(MSE_Loss, self).__init__()
        
    def forward(self, output_batch, teacher_outputs):
    
        # output_batch      -> B X num_classes 
        # teacher_outputs   -> B X num_classes
        
        batch_size = output_batch.size(0)
        output_batch = F.softmax(output_batch, dim = 1)
        teacher_outputs = F.softmax(teacher_outputs, dim = 1)
        # Same result MSE-loss implementation torch.sum -> sum of all element
        loss = torch.sum((output_batch - teacher_outputs) ** 2) / batch_size 
        
        return loss

class E_Loss(nn.Module):
    def __init__(self, temperature = 1):
        super(E_Loss, self).__init__()
        self.T = temperature
    def forward(self, output_batch, teacher_outputs):
    
        # output_batch      -> B X num_classes 
        # teacher_outputs   -> B X num_classes
        
        output_batch = F.log_softmax(output_batch/self.T,dim=1)    
        self_outputs = F.softmax(output_batch/self.T,dim=1)
        
        # Same result CE-loss implementation torch.sum -> sum of all element
        loss = -self.T*self.T*torch.sum(torch.mul(output_batch, self_outputs))/output_batch.size(0)
        
        return loss

class euclidean_distance(nn.Module):
    def __int__(self):
        super(euclidean_distance, self).__int__()
    def forward(self, q1, q2, q3):
        squared_diff1 = (q1 - q2) ** 2
        sum_squared_diff1 = torch.sum(squared_diff1)
        distance1 = torch.sqrt(sum_squared_diff1)

        squared_diff2 = (q1 - q3) ** 2
        sum_squared_diff2 = torch.sum(squared_diff2)
        distance2 = torch.sqrt(sum_squared_diff2)

        squared_diff3 = (q3 - q2) ** 2
        sum_squared_diff3 = torch.sum(squared_diff3)
        distance3 = torch.sqrt(sum_squared_diff3)
        average_distance = (distance1+distance2+distance3) / 3

        return average_distance
class MSE(nn.Module):
    def __int__(self):
        super(MSE, self).__init__()
    def forward(self, fstudent, fteacher):
        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
            c, h, w = fs.shape
            loss = F.mse_loss(fs, ft, reduction='mean')
            cnt = 1.0
            tot = 1.0
            for l in [4, 2, 1]:
                if l >= h:
                    continue
                tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
                tmpft = F.adaptive_avg_pool2d(ft, (l, l))
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                tot += cnt
            loss = loss / tot
            loss_all = loss_all + loss
        return loss_all
    # def __init__(self, s_n, t_n, factor=2):
    #     super(MSE, self).__init__()
    #
    #     def conv1x1(in_channels, out_channels, stride=1):
    #         return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
    #
    #     def conv3x3(in_channels, out_channels, stride=1, groups=1):
    #         return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False,
    #                          groups=groups)
    #
    #     # A bottleneck design to reduce extra parameters
    #     setattr(self, 'transfer', nn.Sequential(
    #         conv1x1(s_n, t_n // factor),
    #         nn.BatchNorm2d(t_n // factor),
    #         nn.ReLU(inplace=True),
    #         conv3x3(t_n // factor, t_n // factor),
    #         # depthwise convolution
    #         # conv3x3(t_n//factor, t_n//factor, groups=t_n//factor),
    #         nn.BatchNorm2d(t_n // factor),
    #         nn.ReLU(inplace=True),
    #         conv1x1(t_n // factor, t_n),
    #         nn.BatchNorm2d(t_n),
    #         nn.ReLU(inplace=True),
    #     ))
    #
    # def forward(self, feat_s, feat_t):
    #     trans_feat_s = getattr(self, 'transfer')(feat_s)
    #     mse_loss = nn.MSELoss()
    #     temp_feat = mse_loss(trans_feat_s, feat_t)
    #     return temp_feat

def lookup(model_name):
    if model_name == "resnet8" or model_name == "resnet14" or model_name == "resnet20" or model_name == "resnet32":
        input_channel = 64
    elif model_name == "densenetd40k12":
        input_channel = 132
    elif model_name == "densenetd100k12":
        input_channel = 342
    elif model_name == "densenetd100k40":
        input_channel = 1126
    elif model_name == "resnet110":
        input_channel = 256
    elif model_name == "vgg16" or model_name == "resnet34":
        input_channel = 512
    elif model_name == "wide_resnet20_8" or model_name == "wide_resnet28_10":
        input_channel = 256
    # imagenet
    elif model_name == "shufflenet_v2_x1_0": 
        input_channel = 1024
    return input_channel
