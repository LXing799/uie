import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Function
import functools
from torch.optim import lr_scheduler
import numpy as np
import cv2
import math
import torch.nn.functional as F
import random


###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            # epoch_count之前进行了多少轮  最后一轮     niter_decay最后一轮额外增加多少轮
            # 这样设置的原因是为了在新一轮的运行中可以重新或者继承之前的最后的学习率，在这通过新添加的轮数来重新线性下降

            # niter是学习率衰减起始的位置   epoch是原执行多少周期  epoch_count当前又执行了多少周期  niter_decay衰减结束周期
            return lr_l
            # 这个返回值就是lr每次调用乘的系数
        # 需要调整的时候只需要scheduler.step()就可以
        # 注意这里不是累成  真实学习率就是直接lr*lr_l    lr始终是lr  所以才叫线性
        # 注意不是  lr = lr*lr_l   而是返回gt_lr = lr*lr_l 不会更新学习率 而是每次都用原始学习率计算  不是指数衰减！！！

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        # 这里传入的epoch是指当前执行到第几个epoch   epoch_count是指从第多少轮开始的  niter是指从多少步开始调整学习率
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        # 每经过opt.lr_decay_iters次调用，学习率衰减为原来的十分之一
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        # 发现损失不再降低或者准确率不再提高后就更新优化器，这里模式为min说明关注的是损失不再降低，不再降低的阈值是0.01，次数为5次，5次不降低就调整学习率
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, device="cpu"):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """

    net.to(device)
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_model(init_type='normal', init_gain=0.02, device="cpu"):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    # 创建一个模型 参数包含 输入通道数 输出通道数     最后卷积卷积核数   结构的名字  标准化使用的方法  是否使用dropout
    # 初始化方法  一个系数参数   gpu信息
    # 首先获得标准化层
    # net = MlUNet(input_nc, output_nc, ngf)
    BF_channel = [64, 32, 64, 32, 32, 46, 16, 8, 8]
    BF_channelout = [32, 32, 16, 8]
    FE_channel = [3, 32, 64, 64, 64]
    BR_channel = [64, 32, 32, 32, 16]
    DR_channel = [3, 32, 64, 64, 64]
    net = MlUNet(DR_channel, BF_channelout, BR_channel, FE_channel, BF_channel)
    return init_net(net, init_type, init_gain, device)


##############################################################################
# Classes
##############################################################################

# **************************************************************
# 这里实现了检测的特征信息提取部分,提取的结果为四个，每次下采样2，也就是1/2大小
# 1/4，1/8，1/16大小
# **************************************************************

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, bias=False)  # 2维卷积，其中采用了自动填充函数。

        # self.ln = nn.LayerNorm(c2, elementwise_affine=True)
        # self.ln = nn.GroupNorm(1, c2)  # ln
        self.ln = nn.GroupNorm(c2, c2)  # in
        # 这里通过gn层的写法实现了ln层，这样可以避免直接调用ln层算法需要传入图像特征尺寸的弊端，直接将BN层改为ln层
        # self.bn = nn.BatchNorm2d(c2)  # 使得每一个batch的特征图均满足均值为0，方差为1的分布规律 由于我们这里的batch数较小，使用BN层的话会导致很大的错误率，这时应该换更有效的layernormalization或者groupnormalization
        # 如果act=True 则采用默认的激活函数SiLU；如果act的类型是nn.Module，则采用传入的act; 否则不采取任何动作 （nn.Identity函数相当于f(x)=x，只用做占位，返回原始的输入）。
        # self.act = nn.SiLU()
        self.act = nn.Tanh()
        # 这里需要注意，在yolo任务中，输出的标定框以及预测分类的概率都是大于0的数值，所以使用SiLU函数的结果正好将数值稳定在0到1
        # 但是我们的任务需要结果在-1到1之间，所以我们在这里考虑使用tanh

    def forward(self, x):  # 前向传播
        # return self.act(self.bn(self.conv(x)))
        return self.act(self.ln(self.conv(x)))


class C2f_1(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.channels = channels_in
        self.conv0 = Conv(self.channels, self.channels, k=1, s=1, p=0)
        self.conv1 = Conv(int(self.channels / 2), int(self.channels / 4), k=3, s=1, p=1)
        self.conv2 = Conv(int(self.channels / 4), int(self.channels / 2), k=3, s=1, p=1)
        self.conv3 = Conv(int(self.channels/2), int(self.channels / 4), k=3, s=1, p=1)
        self.conv4 = Conv(int(self.channels / 4), int(self.channels / 2), k=3, s=1, p=1)
        self.conv5 = Conv(int(self.channels / 2 + self.channels / 2 + self.channels / 2), self.channels, k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv0(x)
        x1, x2 = x.split(int(self.channels/2), dim=1)
        x2_in = x2
        x2_in = self.conv1(x2_in)
        x2_in = self.conv2(x2_in)
        xout1 = x2_in
        x2 = x2 + x2_in
        x2_in = x2
        x2_in = self.conv3(x2_in)
        x2_in = self.conv4(x2_in)
        x2 = x2 + x2_in
        x = torch.cat([x1, x2, xout1], dim=1)
        x = self.conv5(x)
        return x


class C2f_2(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.channels = channels_in
        self.conv0 = Conv(self.channels, self.channels, k=1, s=1, p=0)
        self.conv1 = Conv(int(self.channels / 2), int(self.channels / 4), k=3, s=1, p=1)
        self.conv2 = Conv(int(self.channels / 4), int(self.channels / 2), k=3, s=1, p=1)
        self.conv3 = Conv(int(self.channels/2), int(self.channels / 4), k=3, s=1, p=1)
        self.conv4 = Conv(int(self.channels / 4), int(self.channels / 2), k=3, s=1, p=1)
        self.conv5 = Conv(int(self.channels/2), int(self.channels / 4), k=3, s=1, p=1)
        self.conv6 = Conv(int(self.channels / 4), int(self.channels / 2), k=3, s=1, p=1)
        self.conv7 = Conv(int(self.channels/2), int(self.channels / 4), k=3, s=1, p=1)
        self.conv8 = Conv(int(self.channels / 4), int(self.channels / 2), k=3, s=1, p=1)
        self.conv9 = Conv(int(self.channels / 2 * 5), self.channels, k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv0(x)
        x1, x2 = x.split(int(self.channels/2), dim=1)
        x2_in = x2
        x2_in = self.conv1(x2_in)
        x2_in = self.conv2(x2_in)
        xout1 = x2_in
        x2 = x2 + x2_in
        x2_in = x2
        x2_in = self.conv3(x2_in)
        x2_in = self.conv4(x2_in)
        xout2 = x2_in
        x2 = x2 + x2_in
        x2_in = x2
        x2_in = self.conv5(x2_in)
        x2_in = self.conv6(x2_in)
        xout3 = x2_in
        x2 = x2 + x2_in
        x2_in = x2
        x2_in = self.conv7(x2_in)
        x2_in = self.conv8(x2_in)
        x2 = x2 + x2_in
        x = torch.cat([x1, x2, xout1, xout2, xout3], dim=1)
        x = self.conv9(x)
        return x


class SPPF_mean(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.channels = channels_in
        self.conv0 = Conv(self.channels, self.channels, k=1, s=1, p=0)
        self.avgpool = nn.AvgPool2d(5, 1, padding=2)
        self.conv1 = Conv(self.channels * 4, self.channels, k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv0(x)
        o1 = self.avgpool(x)
        o2 = self.avgpool(o1)
        o3 = self.avgpool(o2)
        x = torch.cat([x, o1, o2, o3], dim=1)
        x = self.conv1(x)
        return x


class SPPF_dif(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.channels = channels_in
        self.conv0 = Conv(self.channels, self.channels, k=1, s=1, p=0)
        self.maxpool = nn.MaxPool2d(5, 1, padding=2)
        self.conv1 = Conv(self.channels * 4, self.channels, k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv0(x)
        o1 = self.maxpool(x)
        o2 = self.maxpool(o1)
        o3 = self.maxpool(o2)
        x = torch.cat([x, o1, o2, o3], dim=1)
        x = self.conv1(x)
        return x


class BlurFeatureExtraction(nn.Module):
    def __init__(self, FE_channel):
        super().__init__()
        self.conv1 = Conv(FE_channel[0], FE_channel[1], k=3, s=2, p=1)
        self.conv2 = Conv(FE_channel[1], FE_channel[2], k=3, s=2, p=1)
        self.sta1 = C2f_1(FE_channel[2])
        self.conv3 = Conv(FE_channel[2], FE_channel[3], k=3, s=2, p=1)
        self.sta2 = C2f_2(FE_channel[3])
        self.conv4 = Conv(FE_channel[3], FE_channel[4], k=3, s=2, p=1)
        self.sta3 = C2f_1(FE_channel[4])
        self.SPPF_m = SPPF_mean(FE_channel[4])
        self.SPPF_d = SPPF_dif(FE_channel[4])

    def forward(self, x):
        x = self.conv1(x)
        x0 = x
        x = self.conv2(x)
        x = self.sta1(x)
        x1 =x
        x = self.conv3(x)
        x = self.sta2(x)
        x2 = x
        x = self.conv4(x)
        x = self.sta3(x)
        x3_m = self.SPPF_m(x)
        x3_d = self.SPPF_d(x)

        return [x0, x1, x2, x3_m, x3_d]

# **************************************************************
# 这里实现了检测的特征信息提取部分
# **************************************************************


# **************************************************************
# 这里实现了根据多层不同尺度的特征进行整合，并最终生成两张图片的过程
# **************************************************************
class FlowUpsample(nn.Module):
    def __init__(self, l_channelin, h_channelin, outchannel):
        super().__init__()
        self.l_channelRConv = Conv(l_channelin, h_channelin, k=1, s=1, p=0)
        self.h_flowConv = Conv(h_channelin, outchannel, k=1, s=1, p=0)
        self.l_flowConv = Conv(h_channelin, outchannel, k=1, s=1, p=0)
        self.flow_make = Conv(outchannel*2, 2, k=3, s=1, p=1)

    def forward(self, x):
        # 注意low特征是面积大的特征，h特征是面积小的特征，也就是下采样次数多于low特征
        l_feature, h_feature = x
        l_feature = self.l_channelRConv(l_feature)
        l_flow = self.l_flowConv(l_feature)
        h_flow = self.h_flowConv(h_feature)
        h, w = l_feature.size()[2:]
        size = (h, w)
        h_flow = F.interpolate(h_flow, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([l_flow, h_flow], 1))
        h_feature = self.flow_warp(h_feature, flow, size=size)
        return h_feature+l_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class DualtaskBlock(nn.Module):
    '''
        该模块的输入包含两个，然后会将
        in1 --data1     conv1(**1) conv5  两边结果concat一下然后1*1卷积恢复为原样,结果拆分 (conv11(data1)+(**1 1*1卷积7))×(**1卷积9)->conv13
            --data2     conv2(**2) conv6                                           (conv12(data2)+(**2 1*1卷积8))×(**1卷积10)->conv14

        in2 --data3     conv3(**3)->全局池化GAP1->全连接层1(*3) 对两边输入进行一个全连接，再拆分为data3->全连接->叠加*3、乘法**3  ->conv15
            --data4     conv4(**4)->全局池化GAP2->全连接层2(*4)                            data4->全连接->叠加*4、乘法**4  ->conv16

    '''
    def __init__(self, inchannel, outchannel):
        # 这里的inchannel，outchannel是指总输出分成两支，每个分支的通道数，
        # 在本模块里，原本的两个分支会进一步分为四个分支，每个分支就是inchannel/2
        # 每个分支的最终输出outchannel/2，最后在concat为两个分支每个为outchannel
        super().__init__()
        self.channel = inchannel
        self.channelo = outchannel
        self.conv1 = Conv(int(inchannel/2), int(outchannel/2), k=3, s=1, p=1)
        self.conv2 = Conv(int(inchannel/2), int(outchannel/2), k=3, s=1, p=1)
        self.conv3 = Conv(int(inchannel/2), int(outchannel/2), k=3, s=1, p=1)
        self.conv4 = Conv(int(inchannel/2), int(outchannel/2), k=3, s=1, p=1)
        self.conv5 = Conv(int(outchannel/2), int(outchannel/2), k=3, s=1, p=1)
        self.conv6 = Conv(int(outchannel/2), int(outchannel/2), k=3, s=1, p=1)
        self.conv = Conv(outchannel, outchannel, k=1, s=1, p=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv7 = Conv(int(outchannel / 2), int(outchannel / 2), k=1, s=1, p=0)
        self.conv8 = Conv(int(outchannel / 2), int(outchannel / 2), k=1, s=1, p=0)
        self.conv9 = Conv(int(outchannel / 2), int(outchannel / 2), k=1, s=1, p=0)
        self.conv10 = Conv(int(outchannel / 2), int(outchannel / 2), k=1, s=1, p=0)
        self.line1 = nn.Linear(int(outchannel / 2), int(outchannel / 2))
        self.line2 = nn.Linear(int(outchannel / 2), int(outchannel / 2))
        self.line = nn.Linear(outchannel, outchannel)
        self.line3 = nn.Linear(int(outchannel / 2), int(outchannel / 2))
        self.line4 = nn.Linear(int(outchannel / 2), int(outchannel / 2))
        self.conv11 = Conv(int(outchannel / 2), int(outchannel / 2), k=3, s=1, p=1)
        self.conv12 = Conv(int(outchannel / 2), int(outchannel / 2), k=3, s=1, p=1)
        self.conv13 = Conv(int(outchannel / 2), int(outchannel / 2), k=3, s=1, p=1)
        self.conv14 = Conv(int(outchannel / 2), int(outchannel / 2), k=3, s=1, p=1)
        self.conv15 = Conv(int(outchannel / 2), int(outchannel / 2), k=3, s=1, p=1)
        self.conv16 = Conv(int(outchannel / 2), int(outchannel / 2), k=3, s=1, p=1)

    def forward(self, x):
        xd, xg = x
        # xd是上支xg是下分支
        x1, x2 = xd.split(int(self.channel/2), dim=1)
        x3, x4 = xg.split(int(self.channel/2), dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv1(x3)
        x4 = self.conv2(x4)
        x3m = x3
        x4m = x4
        x1a = self.conv7(x1)
        x2a = self.conv8(x2)
        x1m = self.conv9(x1)
        x2m = self.conv10(x2)
        x1 = self.conv5(x1)
        x2 = self.conv6(x2)
        x3 = self.avgpool(x3)
        x4 = self.avgpool(x4)
        x3 = x3.reshape(x3.shape[0:2])
        x4 = x4.reshape(x4.shape[0:2])
        x3 = self.line1(x3)
        x4 = self.line2(x4)
        x3a = x3
        x4a = x4
        xd = torch.cat([x1, x2], dim=1)
        xd = self.conv(xd)
        x1, x2 = xd.split(int(self.channelo/2), dim=1)
        xg = torch.cat([x3, x4], dim=1)
        xg = self.line(xg)
        x3, x4 = xg.split(int(self.channelo/2), dim=1)
        x1 = self.conv11(x1)
        x2 = self.conv12(x2)
        x3 = self.line3(x3)
        x4 = self.line4(x4)
        x1 = (x1 + x1a) * x1m
        x2 = (x2 + x2a) * x2m
        x3 = x3 + x3a
        x4 = x4 + x4a
        x3 = x3.unsqueeze(2).unsqueeze(2)
        x4 = x4.unsqueeze(2).unsqueeze(2)
        x3 = x3 * x3m
        x4 = x4 * x4m
        x1 = self.conv13(x1)
        x2 = self.conv14(x2)
        x3 = self.conv15(x3)
        x4 = self.conv16(x4)
        xd = torch.cat([x1, x3], dim=1)
        xm = torch.cat([x2, x4], dim=1)

        return [xd, xm]


class FeaD(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv1 = Conv(channel_in, channel_out, k=1, s=1, p=0)
        self.conv2 = Conv(channel_out, channel_out, k=3, s=1, p=1)
        self.conv3 = Conv(channel_out, channel_out, k=3, s=1, p=1)
        self.conv4 = Conv(channel_out, channel_out, k=3, s=1, p=1)
        self.conv5 = Conv(channel_out, channel_out, k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class BlurFeatureMix(nn.Module):
    def __init__(self, FE_channel, BF_channel, BF_channelout):
        super().__init__()
        # 懂了是通过F.interpolate硬采样采上去的
        self.FE_channel = FE_channel
        self.head1d = FeaD(FE_channel[4], BF_channel[0])
        self.head1g = FeaD(FE_channel[4], BF_channel[0])
        self.dtb1 = DualtaskBlock(BF_channel[0], BF_channel[1])
        self.dtb1out = DualtaskBlock(BF_channel[0], BF_channelout[0])

        self.fu1 = FlowUpsample(int(FE_channel[3] / 2), BF_channel[1], BF_channel[2])
        self.fu2 = FlowUpsample(int(FE_channel[3] / 2), BF_channel[1], BF_channel[2])
        self.head2d = FeaD(BF_channel[1], BF_channel[3])
        self.head2g = FeaD(BF_channel[1], BF_channel[3])
        self.dtb2 = DualtaskBlock(BF_channel[3], BF_channel[3])
        self.dtb2out = DualtaskBlock(BF_channel[3], BF_channelout[1])

        self.fu3 = FlowUpsample(int(FE_channel[2] / 2), BF_channel[3], BF_channel[4])
        self.fu4 = FlowUpsample(int(FE_channel[2] / 2), BF_channel[3], BF_channel[4])
        self.head3d = FeaD(BF_channel[3], BF_channel[5])
        self.head3g = FeaD(BF_channel[3], BF_channel[5])
        self.dtb3 = DualtaskBlock(BF_channel[5], BF_channel[5])
        self.dtb3out = DualtaskBlock(BF_channel[5], BF_channelout[2])

        self.fu5 = FlowUpsample(int(FE_channel[1] / 2), BF_channel[5], BF_channel[6])
        self.fu6 = FlowUpsample(int(FE_channel[1] / 2), BF_channel[5], BF_channel[6])
        self.head4d = FeaD(BF_channel[5], BF_channel[7])
        self.head4g = FeaD(BF_channel[5], BF_channel[7])
        self.dtb4out = DualtaskBlock(BF_channel[7], BF_channelout[3])

    def forward(self, x):
        x_2, x_4, x_8, x_16m, x_16d = x
        x_8d, x_8m = x_8.split(int(self.FE_channel[3]/2), dim=1)
        x_4d, x_4m = x_4.split(int(self.FE_channel[2]/2), dim=1)
        x_2d, x_2m = x_2.split(int(self.FE_channel[1]/2), dim=1)

        x_16d = self.head1d(x_16d)
        x_16m = self.head1g(x_16m)
        x_16dout, x_16mout = self.dtb1out([x_16d, x_16m])
        x_16d, x_16m = self.dtb1([x_16d, x_16m])

        x_8d = self.fu1([x_8d, x_16d])
        x_8m = self.fu2([x_8m, x_16m])
        x_8d = self.head2d(x_8d)
        x_8m = self.head2g(x_8m)
        x_8dout, x_8mout = self.dtb2out([x_8d, x_8m])
        x_8d, x_8m = self.dtb2([x_8d, x_8m])

        x_4d = self.fu3([x_4d, x_8d])
        x_4m = self.fu4([x_4m, x_8m])
        x_4d = self.head3d(x_4d)
        x_4m = self.head3g(x_4m)
        x_4dout, x_4mout = self.dtb3out([x_4d, x_4m])
        x_4d, x_4m = self.dtb3([x_4d, x_4m])

        x_2d = self.fu5([x_2d, x_4d])
        x_2m = self.fu6([x_2m, x_4m])
        x_2d = self.head4d(x_2d)
        x_2m = self.head4g(x_2m)
        x_2dout, x_2mout = self.dtb4out([x_2d, x_2m])

        xd_h, xd_w = x_2dout.size(2), x_2dout.size(3)
        xd_16 = F.interpolate(x_16dout, (xd_h, xd_w), mode='bilinear')
        xd_8 = F.interpolate(x_8dout, (xd_h, xd_w), mode='bilinear')
        xd_4 = F.interpolate(x_4dout, (xd_h, xd_w), mode='bilinear')
        x_d = torch.cat([x_2dout, xd_4, xd_8, xd_16], 1)

        xm_h, xm_w = x_2mout.size(2), x_2mout.size(3)
        xm_16 = F.interpolate(x_16mout, (xm_h, xm_w), mode='bilinear')
        xm_8 = F.interpolate(x_8mout, (xm_h, xm_w), mode='bilinear')
        xm_4 = F.interpolate(x_4mout, (xm_h, xm_w), mode='bilinear')
        x_m = torch.cat([x_2mout, xm_4, xm_8, xm_16], 1)

        return [x_d, x_m]


class BR(nn.Module):
    # blur reconstructing
    def __init__(self, BF_channelout, BR_channel, FE_channel, BF_channel):
        super().__init__()
        BFout = sum(BF_channelout)
        self.FE = BlurFeatureExtraction(FE_channel)
        self.FM = BlurFeatureMix(FE_channel, BF_channel, BF_channelout)
        self.convd1 = Conv(BFout, BR_channel[0], k=3, s=1, p=1)
        self.convm1 = Conv(BFout, BR_channel[0], k=3, s=1, p=1)
        self.convd2 = Conv(BR_channel[0], BR_channel[1], k=3, s=1, p=1)
        self.convm2 = Conv(BR_channel[0], BR_channel[1], k=3, s=1, p=1)
        self.convd3 = Conv(BR_channel[1], BR_channel[2], k=1, s=1, p=0)
        self.convm3 = Conv(BR_channel[1], BR_channel[2], k=1, s=1, p=0)
        self.convd4 = Conv(BR_channel[2], BR_channel[2], k=3, s=1, p=1)
        self.convm4 = Conv(BR_channel[2], BR_channel[2], k=3, s=1, p=1)

        self.upd = nn.ConvTranspose2d(BR_channel[2], BR_channel[2], kernel_size=2, stride=2)
        self.upm = nn.ConvTranspose2d(BR_channel[2], BR_channel[2], kernel_size=2, stride=2)

        self.convd5 = Conv(BR_channel[2], BR_channel[2], k=3, s=1, p=1)
        self.convm5 = Conv(BR_channel[2], BR_channel[2], k=3, s=1, p=1)
        self.convd6 = Conv(BR_channel[2], BR_channel[3], k=3, s=1, p=1)
        self.convm6 = Conv(BR_channel[2], BR_channel[3], k=3, s=1, p=1)
        self.convd7 = Conv(BR_channel[3], BR_channel[4], k=1, s=1, p=0)
        self.convm7 = Conv(BR_channel[3], BR_channel[4], k=1, s=1, p=0)
        self.convd8 = Conv(BR_channel[4], 3, k=3, s=1, p=1)
        self.convm8 = Conv(BR_channel[4], 3, k=3, s=1, p=1)

    def forward(self, x):
        x = self.FE(x)
        xd, xm = self.FM(x)
        # 实在不行就在这里添加一个sigmiod函数然后再-0.5
        xd = self.convd1(xd)
        xd = self.convd2(xd)
        xd = self.convd3(xd)
        xd = self.convd4(xd)
        xd = self.upd(xd)

        xm = self.convd1(xm)
        xm = self.convd2(xm)
        xm = self.convd3(xm)
        xm = self.convd4(xm)
        xm = self.upd(xm)

        xd = self.convd5(xd)
        xd = self.convd6(xd)
        xd = self.convd7(xd)
        xd = self.convd8(xd)

        xm = self.convd5(xm)
        xm = self.convd6(xm)
        xm = self.convd7(xm)
        xm = self.convd8(xm)

        return [xd, xm]
# **************************************************************
# 这里实现了根据多层不同尺度的特征进行整合，并最终生成两张图片的过程
# **************************************************************


class CA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1, bias=True)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        f = x
        x = self.GAP(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigm(x)
        return f * x


class PDU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = Conv(in_channels, in_channels, k=3, s=1, p=1)
        self.conv2 = Conv(in_channels, in_channels, k=3, s=1, p=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.line1 = nn.Linear(in_channels, in_channels)
        self.line2 = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        x1 = self.avgpool(x)
        x1 = x1.reshape(x1.shape[0:2])
        x1 = torch.nn.functional.relu(self.line1(x1))
        x1 = torch.nn.functional.sigmoid(self.line2(x1))
        x1 = x1.unsqueeze(2).unsqueeze(2)
        x2 = self.conv1(x)
        x2 = self.conv2(x2)
        x2 = torch.nn.functional.sigmoid(self.conv3(x2))

        out1 = x1*(1 - x2)
        out2 = x*x2
        return out1+out2


class PBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, in_channels, k=3, s=1, p=1)
        self.conv2 = Conv(in_channels, in_channels, k=3, s=1, p=1)
        self.ca = CA(in_channels)
        self.pdu = PDU(in_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.ca(x1)
        x1 = self.pdu(x1)

        return x+x1


class Gro(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.b1 = PBlock(in_channels)
        self.b2 = PBlock(in_channels)
        self.b3 = PBlock(in_channels)
        self.b4 = PBlock(in_channels)
        self.conv = Conv(in_channels, in_channels, k=3, s=1, p=1)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)

        return self.conv(x)


class Gout(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.b1 = PBlock(in_channels)
        self.b2 = PBlock(in_channels)
        self.conv1 = Conv(in_channels, int(in_channels / 2), k=3, s=1, p=1)
        self.b3 = PBlock(int(in_channels / 2))
        self.b4 = PBlock(int(in_channels / 2))
        self.conv2 = Conv(int(in_channels / 2), int(in_channels / 4), k=3, s=1, p=1)
        self.b5 = PBlock(int(in_channels / 4))
        self.b6 = PBlock(int(in_channels / 4))
        self.conv3 = Conv(int(in_channels / 4), int(in_channels / 8), k=3, s=1, p=1)
        self.conv4 = Conv(int(in_channels / 8), 3, k=3, s=1, p=1)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.conv1(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.conv2(x)
        x = self.b5(x)
        x = self.b6(x)
        x = self.conv3(x)

        return self.conv4(x)


class DR(nn.Module):
    def __init__(self, DR_channel):
        super().__init__()
        self.conv0 = Conv(DR_channel[0], DR_channel[1], k=3, s=2, p=1)
        self.conv1 = Conv(DR_channel[1], DR_channel[2], k=3, s=2, p=1)
        self.g1 = Gro(DR_channel[2])
        self.conv2 = Conv(DR_channel[2], DR_channel[3], k=3, s=2, p=1)
        self.g2 = Gro(DR_channel[3])
        self.conv3 = Conv(DR_channel[3], DR_channel[4], k=3, s=2, p=1)
        self.g3 = Gro(DR_channel[4])
        self.upd1 = nn.ConvTranspose2d(DR_channel[4], DR_channel[4], kernel_size=2, stride=2)
        self.upd2 = nn.ConvTranspose2d(DR_channel[3] + DR_channel[4], DR_channel[3] + DR_channel[4], kernel_size=2, stride=2)
        self.upd3 = nn.ConvTranspose2d(DR_channel[2] + DR_channel[3] + DR_channel[4], DR_channel[2] + DR_channel[3] + DR_channel[4], kernel_size=2, stride=2)
        self.upd4 = nn.ConvTranspose2d(DR_channel[1] + DR_channel[2] + DR_channel[3] + DR_channel[4], DR_channel[1] + DR_channel[2] + DR_channel[3] + DR_channel[4], kernel_size=2, stride=2)
        self.out = Gout(DR_channel[1] + DR_channel[2] + DR_channel[3] + DR_channel[4])

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x2 = self.g1(x2)
        x3 = self.conv2(x2)
        x3 = self.g2(x3)
        x4 = self.conv3(x3)
        x4 = self.g2(x4)
        x4 = self.upd1(x4)
        x3 = torch.cat([x3, x4], dim=1)
        x3 = self.upd2(x3)
        x2 = torch.cat([x2, x3], dim=1)
        x2 = self.upd3(x2)
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.upd4(x1)
        xout = self.out(x1)

        return xout + x


class outM(nn.Module):
    def __init__(self, channel, mid_channel, out_channel):
        super().__init__()
        self.conv0 = Conv(channel, mid_channel, k=3, s=1, p=1)
        # 细节恢复
        self.ca = CA(mid_channel)
        # 模糊区域恢复
        self.pdu = PDU(mid_channel)
        self.conv1 = Conv(mid_channel, out_channel, k=3, s=1, p=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.ca(x)
        x = self.pdu(x)
        x = self.conv1(x)

        return x


class MlUNet(nn.Module):
    def __init__(self, DR_channel, BF_channelout, BR_channel, FE_channel, BF_channel):
        super().__init__()
        # 细节恢复
        self.dm = DR(DR_channel)
        # 模糊区域恢复
        self.bm = BR(BF_channelout, BR_channel, FE_channel, BF_channel)
        self.outm1 = outM(3, 32, 3)
        self.outm2 = outM(3, 32, 3)

    def forward(self, x):
        x_d, x_b = x
        x_dout = self.dm(x_d)
        x_bd, x_bm = self.bm(x_b)
        x_bout = x_bd + x_bm
        x_bout = self.outm1(x_bout)
        x_out = x_dout + x_bout
        x_out = self.outm2(x_out)

        return x_out, x_dout, x_bd, x_bm, x_bout






'''
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Function
import functools
from torch.optim import lr_scheduler
import numpy as np
import cv2
import math
import torch.nn.functional as F
import random


###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            # epoch_count之前进行了多少轮  最后一轮     niter_decay最后一轮额外增加多少轮
            # 这样设置的原因是为了在新一轮的运行中可以重新或者继承之前的最后的学习率，在这通过新添加的轮数来重新线性下降

            # niter是学习率衰减起始的位置   epoch是原执行多少周期  epoch_count当前又执行了多少周期  niter_decay衰减结束周期
            return lr_l
            # 这个返回值就是lr每次调用乘的系数
        # 需要调整的时候只需要scheduler.step()就可以
        # 注意这里不是累成  真实学习率就是直接lr*lr_l    lr始终是lr  所以才叫线性
        # 注意不是  lr = lr*lr_l   而是返回gt_lr = lr*lr_l 不会更新学习率 而是每次都用原始学习率计算  不是指数衰减！！！

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        # 这里传入的epoch是指当前执行到第几个epoch   epoch_count是指从第多少轮开始的  niter是指从多少步开始调整学习率
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        # 每经过opt.lr_decay_iters次调用，学习率衰减为原来的十分之一
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        # 发现损失不再降低或者准确率不再提高后就更新优化器，这里模式为min说明关注的是损失不再降低，不再降低的阈值是0.01，次数为5次，5次不降低就调整学习率
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, device="cpu"):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """

    net.to(device)
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_model(init_type='normal', init_gain=0.02, device="cpu"):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    # 创建一个模型 参数包含 输入通道数 输出通道数     最后卷积卷积核数   结构的名字  标准化使用的方法  是否使用dropout
    # 初始化方法  一个系数参数   gpu信息
    # 首先获得标准化层
    # net = MlUNet(input_nc, output_nc, ngf)
    BF_channel = [128, 64, 128, 64, 64, 32, 32, 16, 16]
    BF_channelout = [64, 64, 32, 16]
    FE_channel = [3, 32, 64, 128, 64]
    BR_channel = [128, 64, 64, 32, 16]
    DR_channel = [3, 32, 64, 64, 64]
    net = MlUNet(DR_channel, BF_channelout, BR_channel, FE_channel, BF_channel)
    return init_net(net, init_type, init_gain, device)


##############################################################################
# Classes
##############################################################################

# **************************************************************
# 这里实现了检测的特征信息提取部分,提取的结果为四个，每次下采样2，也就是1/2大小
# 1/4，1/8，1/16大小
# **************************************************************

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, bias=False)  # 2维卷积，其中采用了自动填充函数。
        self.bn = nn.BatchNorm2d(c2)  # 使得每一个batch的特征图均满足均值为0，方差为1的分布规律
        # 如果act=True 则采用默认的激活函数SiLU；如果act的类型是nn.Module，则采用传入的act; 否则不采取任何动作 （nn.Identity函数相当于f(x)=x，只用做占位，返回原始的输入）。
        self.act = nn.SiLU()

    def forward(self, x):  # 前向传播
        return self.act(self.bn(self.conv(x)))


class C2f_1(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.channels = channels_in
        self.conv0 = Conv(self.channels, self.channels, k=1, s=1, p=0)
        self.conv1 = Conv(int(self.channels / 2), int(self.channels / 4), k=3, s=1, p=1)
        self.conv2 = Conv(int(self.channels / 4), int(self.channels / 2), k=3, s=1, p=1)
        self.conv3 = Conv(int(self.channels/2), int(self.channels / 4), k=3, s=1, p=1)
        self.conv4 = Conv(int(self.channels / 4), int(self.channels / 2), k=3, s=1, p=1)
        self.conv5 = Conv(int(self.channels / 2 + self.channels / 2 + self.channels / 2), self.channels, k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv0(x)
        x1, x2 = x.split(int(self.channels/2), dim=1)
        x2_in = x2
        x2_in = self.conv1(x2_in)
        x2_in = self.conv2(x2_in)
        xout1 = x2_in
        x2 = x2 + x2_in
        x2_in = x2
        x2_in = self.conv3(x2_in)
        x2_in = self.conv4(x2_in)
        x2 = x2 + x2_in
        x = torch.cat([x1, x2, xout1], dim=1)
        x = self.conv5(x)
        return x


class C2f_2(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.channels = channels_in
        self.conv0 = Conv(self.channels, self.channels, k=1, s=1, p=0)
        self.conv1 = Conv(int(self.channels / 2), int(self.channels / 4), k=3, s=1, p=1)
        self.conv2 = Conv(int(self.channels / 4), int(self.channels / 2), k=3, s=1, p=1)
        self.conv3 = Conv(int(self.channels/2), int(self.channels / 4), k=3, s=1, p=1)
        self.conv4 = Conv(int(self.channels / 4), int(self.channels / 2), k=3, s=1, p=1)
        self.conv5 = Conv(int(self.channels/2), int(self.channels / 4), k=3, s=1, p=1)
        self.conv6 = Conv(int(self.channels / 4), int(self.channels / 2), k=3, s=1, p=1)
        self.conv7 = Conv(int(self.channels/2), int(self.channels / 4), k=3, s=1, p=1)
        self.conv8 = Conv(int(self.channels / 4), int(self.channels / 2), k=3, s=1, p=1)
        self.conv9 = Conv(int(self.channels / 2 * 5), self.channels, k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv0(x)
        x1, x2 = x.split(int(self.channels/2), dim=1)
        x2_in = x2
        x2_in = self.conv1(x2_in)
        x2_in = self.conv2(x2_in)
        xout1 = x2_in
        x2 = x2 + x2_in
        x2_in = x2
        x2_in = self.conv3(x2_in)
        x2_in = self.conv4(x2_in)
        xout2 = x2_in
        x2 = x2 + x2_in
        x2_in = x2
        x2_in = self.conv5(x2_in)
        x2_in = self.conv6(x2_in)
        xout3 = x2_in
        x2 = x2 + x2_in
        x2_in = x2
        x2_in = self.conv7(x2_in)
        x2_in = self.conv8(x2_in)
        x2 = x2 + x2_in
        x = torch.cat([x1, x2, xout1, xout2, xout3], dim=1)
        x = self.conv9(x)
        return x


class SPPF_mean(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.channels = channels_in
        self.conv0 = Conv(self.channels, self.channels, k=1, s=1, p=0)
        self.avgpool = nn.AvgPool2d(5, 1, padding=2)
        self.conv1 = Conv(self.channels * 4, self.channels, k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv0(x)
        o1 = self.avgpool(x)
        o2 = self.avgpool(o1)
        o3 = self.avgpool(o2)
        x = torch.cat([x, o1, o2, o3], dim=1)
        x = self.conv1(x)
        return x


class SPPF_dif(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.channels = channels_in
        self.conv0 = Conv(self.channels, self.channels, k=1, s=1, p=0)
        self.maxpool = nn.MaxPool2d(5, 1, padding=2)
        self.conv1 = Conv(self.channels * 4, self.channels, k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv0(x)
        o1 = self.maxpool(x)
        o2 = self.maxpool(o1)
        o3 = self.maxpool(o2)
        x = torch.cat([x, o1, o2, o3], dim=1)
        x = self.conv1(x)
        return x


class BlurFeatureExtraction(nn.Module):
    def __init__(self, FE_channel):
        super().__init__()
        self.conv1 = Conv(FE_channel[0], FE_channel[1], k=3, s=2, p=1)
        self.conv2 = Conv(FE_channel[1], FE_channel[2], k=3, s=2, p=1)
        self.sta1 = C2f_1(FE_channel[2])
        self.conv3 = Conv(FE_channel[2], FE_channel[3], k=3, s=2, p=1)
        self.sta2 = C2f_2(FE_channel[3])
        self.conv4 = Conv(FE_channel[3], FE_channel[4], k=3, s=2, p=1)
        self.sta3 = C2f_1(FE_channel[4])
        self.SPPF_m = SPPF_mean(FE_channel[4])
        self.SPPF_d = SPPF_dif(FE_channel[4])

    def forward(self, x):
        x = self.conv1(x)
        x0 = x
        x = self.conv2(x)
        x = self.sta1(x)
        x1 =x
        x = self.conv3(x)
        x = self.sta2(x)
        x2 = x
        x = self.conv4(x)
        x = self.sta3(x)
        x3_m = self.SPPF_m(x)
        x3_d = self.SPPF_d(x)

        return [x0, x1, x2, x3_m, x3_d]

# **************************************************************
# 这里实现了检测的特征信息提取部分
# **************************************************************


# **************************************************************
# 这里实现了根据多层不同尺度的特征进行整合，并最终生成两张图片的过程
# **************************************************************
class FlowUpsample(nn.Module):
    def __init__(self, l_channelin, h_channelin, outchannel):
        super().__init__()
        self.l_channelRConv = Conv(l_channelin, h_channelin, k=1, s=1, p=0)
        self.h_flowConv = Conv(h_channelin, outchannel, k=1, s=1, p=0)
        self.l_flowConv = Conv(h_channelin, outchannel, k=1, s=1, p=0)
        self.flow_make = Conv(outchannel*2, 2, k=3, s=1, p=1)

    def forward(self, x):
        # 注意low特征是面积大的特征，h特征是面积小的特征，也就是下采样次数多于low特征
        l_feature, h_feature = x
        l_feature = self.l_channelRConv(l_feature)
        l_flow = self.l_flowConv(l_feature)
        h_flow = self.h_flowConv(h_feature)
        h, w = l_feature.size()[2:]
        size = (h, w)
        h_flow = F.interpolate(h_flow, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([l_flow, h_flow], 1))
        h_feature = self.flow_warp(h_feature, flow, size=size)
        return h_feature+l_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class DualtaskBlock(nn.Module):
   
        
        #该模块的输入包含两个，然后会将
        #in1 --data1     conv1(**1) conv5  两边结果concat一下然后1*1卷积恢复为原样,结果拆分 (conv11(data1)+(**1 1*1卷积7))×(**1卷积9)->conv13
        #    --data2     conv2(**2) conv6                                           (conv12(data2)+(**2 1*1卷积8))×(**1卷积10)->conv14

        #in2 --data3     conv3(**3)->全局池化GAP1->全连接层1(*3) 对两边输入进行一个全连接，再拆分为data3->全连接->叠加*3、乘法**3  ->conv15
        #    --data4     conv4(**4)->全局池化GAP2->全连接层2(*4)                            data4->全连接->叠加*4、乘法**4  ->conv16
        
    
    def __init__(self, inchannel, outchannel):
        # 这里的inchannel，outchannel是指总输出分成两支，每个分支的通道数，
        # 在本模块里，原本的两个分支会进一步分为四个分支，每个分支就是inchannel/2
        # 每个分支的最终输出outchannel/2，最后在concat为两个分支每个为outchannel
        super().__init__()
        self.channel = inchannel
        self.channelo = outchannel
        self.conv1 = Conv(int(inchannel/2), int(outchannel/2), k=3, s=1, p=1)
        self.conv2 = Conv(int(inchannel/2), int(outchannel/2), k=3, s=1, p=1)
        self.conv3 = Conv(int(inchannel/2), int(outchannel/2), k=3, s=1, p=1)
        self.conv4 = Conv(int(inchannel/2), int(outchannel/2), k=3, s=1, p=1)
        self.conv5 = Conv(int(outchannel/2), int(outchannel/2), k=3, s=1, p=1)
        self.conv6 = Conv(int(outchannel/2), int(outchannel/2), k=3, s=1, p=1)
        self.conv = Conv(outchannel, outchannel, k=1, s=1, p=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv7 = Conv(int(outchannel / 2), int(outchannel / 2), k=1, s=1, p=0)
        self.conv8 = Conv(int(outchannel / 2), int(outchannel / 2), k=1, s=1, p=0)
        self.conv9 = Conv(int(outchannel / 2), int(outchannel / 2), k=1, s=1, p=0)
        self.conv10 = Conv(int(outchannel / 2), int(outchannel / 2), k=1, s=1, p=0)
        self.line1 = nn.Linear(int(outchannel / 2), int(outchannel / 2))
        self.line2 = nn.Linear(int(outchannel / 2), int(outchannel / 2))
        self.line = nn.Linear(outchannel, outchannel)
        self.line3 = nn.Linear(int(outchannel / 2), int(outchannel / 2))
        self.line4 = nn.Linear(int(outchannel / 2), int(outchannel / 2))
        self.conv11 = Conv(int(outchannel / 2), int(outchannel / 2), k=3, s=1, p=1)
        self.conv12 = Conv(int(outchannel / 2), int(outchannel / 2), k=3, s=1, p=1)
        self.conv13 = Conv(int(outchannel / 2), int(outchannel / 2), k=3, s=1, p=1)
        self.conv14 = Conv(int(outchannel / 2), int(outchannel / 2), k=3, s=1, p=1)
        self.conv15 = Conv(int(outchannel / 2), int(outchannel / 2), k=3, s=1, p=1)
        self.conv16 = Conv(int(outchannel / 2), int(outchannel / 2), k=3, s=1, p=1)

    def forward(self, x):
        xd, xg = x
        # xd是上支xg是下分支
        x1, x2 = xd.split(int(self.channel/2), dim=1)
        x3, x4 = xg.split(int(self.channel/2), dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv1(x3)
        x4 = self.conv2(x4)
        x3m = x3
        x4m = x4
        x1a = self.conv7(x1)
        x2a = self.conv8(x2)
        x1m = self.conv9(x1)
        x2m = self.conv10(x2)
        x1 = self.conv5(x1)
        x2 = self.conv6(x2)
        x3 = self.avgpool(x3)
        x4 = self.avgpool(x4)
        x3 = x3.reshape(x3.shape[0:2])
        x4 = x4.reshape(x4.shape[0:2])
        x3 = self.line1(x3)
        x4 = self.line2(x4)
        x3a = x3
        x4a = x4
        xd = torch.cat([x1, x2], dim=1)
        xd = self.conv(xd)
        x1, x2 = xd.split(int(self.channelo/2), dim=1)
        xg = torch.cat([x3, x4], dim=1)
        xg = self.line(xg)
        x3, x4 = xg.split(int(self.channelo/2), dim=1)
        x1 = self.conv11(x1)
        x2 = self.conv12(x2)
        x3 = self.line3(x3)
        x4 = self.line4(x4)
        x1 = (x1 + x1a) * x1m
        x2 = (x2 + x2a) * x2m
        x3 = x3 + x3a
        x4 = x4 + x4a
        x3 = x3.unsqueeze(2).unsqueeze(2)
        x4 = x4.unsqueeze(2).unsqueeze(2)
        x3 = x3 * x3m
        x4 = x4 * x4m
        x1 = self.conv13(x1)
        x2 = self.conv14(x2)
        x3 = self.conv15(x3)
        x4 = self.conv16(x4)
        xd = torch.cat([x1, x3], dim=1)
        xm = torch.cat([x2, x4], dim=1)

        return [xd, xm]


class FeaD(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv1 = Conv(channel_in, channel_out, k=1, s=1, p=0)
        self.conv2 = Conv(channel_out, channel_out, k=3, s=1, p=1)
        self.conv3 = Conv(channel_out, channel_out, k=3, s=1, p=1)
        self.conv4 = Conv(channel_out, channel_out, k=3, s=1, p=1)
        self.conv5 = Conv(channel_out, channel_out, k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class BlurFeatureMix(nn.Module):
    def __init__(self, FE_channel, BF_channel, BF_channelout):
        super().__init__()
        # 懂了是通过F.interpolate硬采样采上去的
        self.FE_channel = FE_channel
        self.head1d = FeaD(FE_channel[4], BF_channel[0])
        self.head1g = FeaD(FE_channel[4], BF_channel[0])
        self.dtb1 = DualtaskBlock(BF_channel[0], BF_channel[1])
        self.dtb1out = DualtaskBlock(BF_channel[0], BF_channelout[0])

        self.fu1 = FlowUpsample(int(FE_channel[3] / 2), BF_channel[1], BF_channel[2])
        self.fu2 = FlowUpsample(int(FE_channel[3] / 2), BF_channel[1], BF_channel[2])
        self.head2d = FeaD(BF_channel[1], BF_channel[3])
        self.head2g = FeaD(BF_channel[1], BF_channel[3])
        self.dtb2 = DualtaskBlock(BF_channel[3], BF_channel[3])
        self.dtb2out = DualtaskBlock(BF_channel[3], BF_channelout[1])

        self.fu3 = FlowUpsample(int(FE_channel[2] / 2), BF_channel[3], BF_channel[4])
        self.fu4 = FlowUpsample(int(FE_channel[2] / 2), BF_channel[3], BF_channel[4])
        self.head3d = FeaD(BF_channel[3], BF_channel[5])
        self.head3g = FeaD(BF_channel[3], BF_channel[5])
        self.dtb3 = DualtaskBlock(BF_channel[5], BF_channel[5])
        self.dtb3out = DualtaskBlock(BF_channel[5], BF_channelout[2])

        self.fu5 = FlowUpsample(int(FE_channel[1] / 2), BF_channel[5], BF_channel[6])
        self.fu6 = FlowUpsample(int(FE_channel[1] / 2), BF_channel[5], BF_channel[6])
        self.head4d = FeaD(BF_channel[5], BF_channel[7])
        self.head4g = FeaD(BF_channel[5], BF_channel[7])
        self.dtb4out = DualtaskBlock(BF_channel[7], BF_channelout[3])

    def forward(self, x):
        x_2, x_4, x_8, x_16m, x_16d = x
        x_8d, x_8m = x_8.split(int(self.FE_channel[3]/2), dim=1)
        x_4d, x_4m = x_4.split(int(self.FE_channel[2]/2), dim=1)
        x_2d, x_2m = x_2.split(int(self.FE_channel[1]/2), dim=1)

        x_16d = self.head1d(x_16d)
        x_16m = self.head1g(x_16m)
        x_16dout, x_16mout = self.dtb1out([x_16d, x_16m])
        x_16d, x_16m = self.dtb1([x_16d, x_16m])

        x_8d = self.fu1([x_8d, x_16d])
        x_8m = self.fu2([x_8m, x_16m])
        x_8d = self.head2d(x_8d)
        x_8m = self.head2g(x_8m)
        x_8dout, x_8mout = self.dtb2out([x_8d, x_8m])
        x_8d, x_8m = self.dtb2([x_8d, x_8m])

        x_4d = self.fu3([x_4d, x_8d])
        x_4m = self.fu4([x_4m, x_8m])
        x_4d = self.head3d(x_4d)
        x_4m = self.head3g(x_4m)
        x_4dout, x_4mout = self.dtb3out([x_4d, x_4m])
        x_4d, x_4m = self.dtb3([x_4d, x_4m])

        x_2d = self.fu5([x_2d, x_4d])
        x_2m = self.fu6([x_2m, x_4m])
        x_2d = self.head4d(x_2d)
        x_2m = self.head4g(x_2m)
        x_2dout, x_2mout = self.dtb4out([x_2d, x_2m])

        xd_h, xd_w = x_2dout.size(2), x_2dout.size(3)
        xd_16 = F.interpolate(x_16dout, (xd_h, xd_w), mode='bilinear')
        xd_8 = F.interpolate(x_8dout, (xd_h, xd_w), mode='bilinear')
        xd_4 = F.interpolate(x_4dout, (xd_h, xd_w), mode='bilinear')
        x_d = torch.cat([x_2dout, xd_4, xd_8, xd_16], 1)

        xm_h, xm_w = x_2mout.size(2), x_2mout.size(3)
        xm_16 = F.interpolate(x_16mout, (xm_h, xm_w), mode='bilinear')
        xm_8 = F.interpolate(x_8mout, (xm_h, xm_w), mode='bilinear')
        xm_4 = F.interpolate(x_4mout, (xm_h, xm_w), mode='bilinear')
        x_m = torch.cat([x_2mout, xm_4, xm_8, xm_16], 1)

        return [x_d, x_m]


class BR(nn.Module):
    # blur reconstructing
    def __init__(self, BF_channelout, BR_channel, FE_channel, BF_channel):
        super().__init__()
        BFout = sum(BF_channelout)
        self.FE = BlurFeatureExtraction(FE_channel)
        self.FM = BlurFeatureMix(FE_channel, BF_channel, BF_channelout)
        self.convd1 = Conv(BFout, BR_channel[0], k=3, s=1, p=1)
        self.convm1 = Conv(BFout, BR_channel[0], k=3, s=1, p=1)
        self.convd2 = Conv(BR_channel[0], BR_channel[1], k=3, s=1, p=1)
        self.convm2 = Conv(BR_channel[0], BR_channel[1], k=3, s=1, p=1)
        self.convd3 = Conv(BR_channel[1], BR_channel[2], k=1, s=1, p=0)
        self.convm3 = Conv(BR_channel[1], BR_channel[2], k=1, s=1, p=0)
        self.convd4 = Conv(BR_channel[2], BR_channel[2], k=3, s=1, p=1)
        self.convm4 = Conv(BR_channel[2], BR_channel[2], k=3, s=1, p=1)

        self.upd = nn.ConvTranspose2d(BR_channel[2], BR_channel[2], kernel_size=2, stride=2)
        self.upm = nn.ConvTranspose2d(BR_channel[2], BR_channel[2], kernel_size=2, stride=2)

        self.convd5 = Conv(BR_channel[2], BR_channel[2], k=3, s=1, p=1)
        self.convm5 = Conv(BR_channel[2], BR_channel[2], k=3, s=1, p=1)
        self.convd6 = Conv(BR_channel[2], BR_channel[3], k=3, s=1, p=1)
        self.convm6 = Conv(BR_channel[2], BR_channel[3], k=3, s=1, p=1)
        self.convd7 = Conv(BR_channel[3], BR_channel[4], k=1, s=1, p=0)
        self.convm7 = Conv(BR_channel[3], BR_channel[4], k=1, s=1, p=0)
        self.convd8 = Conv(BR_channel[4], 3, k=3, s=1, p=1)
        self.convm8 = Conv(BR_channel[4], 3, k=3, s=1, p=1)

    def forward(self, x):
        x = self.FE(x)
        xd, xm = self.FM(x)
        xd = self.convd1(xd)
        xd = self.convd2(xd)
        xd = self.convd3(xd)
        xd = self.convd4(xd)
        xd = self.upd(xd)

        xm = self.convd1(xm)
        xm = self.convd2(xm)
        xm = self.convd3(xm)
        xm = self.convd4(xm)
        xm = self.upd(xm)

        xd = self.convd5(xd)
        xd = self.convd6(xd)
        xd = self.convd7(xd)
        xd = self.convd8(xd)

        xm = self.convd5(xm)
        xm = self.convd6(xm)
        xm = self.convd7(xm)
        xm = self.convd8(xm)

        return [xd, xm]
# **************************************************************
# 这里实现了根据多层不同尺度的特征进行整合，并最终生成两张图片的过程
# **************************************************************


class CA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1, bias=True)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        f = x
        x = self.GAP(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigm(x)
        return f * x


class PDU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = Conv(in_channels, in_channels, k=3, s=1, p=1)
        self.conv2 = Conv(in_channels, in_channels, k=3, s=1, p=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.line1 = nn.Linear(in_channels, in_channels)
        self.line2 = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        x1 = self.avgpool(x)
        x1 = x1.reshape(x1.shape[0:2])
        x1 = torch.nn.functional.relu(self.line1(x1))
        x1 = torch.nn.functional.sigmoid(self.line2(x1))
        x1 = x1.unsqueeze(2).unsqueeze(2)
        x2 = self.conv1(x)
        x2 = self.conv2(x2)
        x2 = torch.nn.functional.sigmoid(self.conv3(x2))

        out1 = x1*(1 - x2)
        out2 = x*x2
        return out1+out2


class PBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, in_channels, k=3, s=1, p=1)
        self.conv2 = Conv(in_channels, in_channels, k=3, s=1, p=1)
        self.ca = CA(in_channels)
        self.pdu = PDU(in_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.ca(x1)
        x1 = self.pdu(x1)

        return x+x1


class Gro(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.b1 = PBlock(in_channels)
        self.b2 = PBlock(in_channels)
        self.b3 = PBlock(in_channels)
        self.b4 = PBlock(in_channels)
        self.conv = Conv(in_channels, in_channels, k=3, s=1, p=1)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)

        return self.conv(x)


class Gout(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.b1 = PBlock(in_channels)
        self.b2 = PBlock(in_channels)
        self.conv1 = Conv(in_channels, int(in_channels / 2), k=3, s=1, p=1)
        self.b3 = PBlock(int(in_channels / 2))
        self.b4 = PBlock(int(in_channels / 2))
        self.conv2 = Conv(int(in_channels / 2), int(in_channels / 4), k=3, s=1, p=1)
        self.b5 = PBlock(int(in_channels / 4))
        self.b6 = PBlock(int(in_channels / 4))
        self.conv3 = Conv(int(in_channels / 4), int(in_channels / 8), k=3, s=1, p=1)
        self.conv4 = Conv(int(in_channels / 8), 3, k=3, s=1, p=1)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.conv1(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.conv2(x)
        x = self.b5(x)
        x = self.b6(x)
        x = self.conv3(x)

        return self.conv4(x)


class DR(nn.Module):
    def __init__(self, DR_channel):
        super().__init__()
        self.conv0 = Conv(DR_channel[0], DR_channel[1], k=3, s=2, p=1)
        self.conv1 = Conv(DR_channel[1], DR_channel[2], k=3, s=2, p=1)
        self.g1 = Gro(DR_channel[2])
        self.conv2 = Conv(DR_channel[2], DR_channel[3], k=3, s=2, p=1)
        self.g2 = Gro(DR_channel[3])
        self.conv3 = Conv(DR_channel[3], DR_channel[4], k=3, s=2, p=1)
        self.g3 = Gro(DR_channel[4])
        self.upd1 = nn.ConvTranspose2d(DR_channel[4], DR_channel[4], kernel_size=2, stride=2)
        self.upd2 = nn.ConvTranspose2d(DR_channel[3] + DR_channel[4], DR_channel[3] + DR_channel[4], kernel_size=2, stride=2)
        self.upd3 = nn.ConvTranspose2d(DR_channel[2] + DR_channel[3] + DR_channel[4], DR_channel[2] + DR_channel[3] + DR_channel[4], kernel_size=2, stride=2)
        self.upd4 = nn.ConvTranspose2d(DR_channel[1] + DR_channel[2] + DR_channel[3] + DR_channel[4], DR_channel[1] + DR_channel[2] + DR_channel[3] + DR_channel[4], kernel_size=2, stride=2)
        self.out = Gout(DR_channel[1] + DR_channel[2] + DR_channel[3] + DR_channel[4])

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x2 = self.g1(x2)
        x3 = self.conv2(x2)
        x3 = self.g2(x3)
        x4 = self.conv3(x3)
        x4 = self.g2(x4)
        x4 = self.upd1(x4)
        x3 = torch.cat([x3, x4], dim=1)
        x3 = self.upd2(x3)
        x2 = torch.cat([x2, x3], dim=1)
        x2 = self.upd3(x2)
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.upd4(x1)
        xout = self.out(x1)

        return xout + x


class outM(nn.Module):
    def __init__(self, channel, mid_channel, out_channel):
        super().__init__()
        self.conv0 = Conv(channel, mid_channel, k=3, s=1, p=1)
        # 细节恢复
        self.ca = CA(mid_channel)
        # 模糊区域恢复
        self.pdu = PDU(mid_channel)
        self.conv1 = Conv(mid_channel, out_channel, k=3, s=1, p=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.ca(x)
        x = self.pdu(x)
        x = self.conv1(x)

        return x


class MlUNet(nn.Module):
    def __init__(self, DR_channel, BF_channelout, BR_channel, FE_channel, BF_channel):
        super().__init__()
        # 细节恢复
        self.dm = DR(DR_channel)
        # 模糊区域恢复
        self.bm = BR(BF_channelout, BR_channel, FE_channel, BF_channel)
        self.outm1 = outM(3, 32, 3)
        self.outm2 = outM(3, 32, 3)

    def forward(self, x):
        x_d, x_b = x
        x_dout = self.dm(x_d)
        x_bd, x_bm = self.bm(x_b)
        x_bout = x_bd + x_bm
        x_bout = self.outm1(x_bout)
        x_out = x_dout + x_bout
        x_out = self.outm2(x_out)

        return x_out, x_dout, x_bd, x_bm, x_bout
'''