"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import random
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math




def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)



def mask2RGB(mask, ref):
    """
    Converts a 8 channels mask tensor to a numpy image array.
    :param input_image: the input image tensor array with 8 channels
    :return: 3 channels numpy image array
    """
    mask_data = mask
    ref_data = ref
    color = [tuple(np.random.choice(range(30, 220, 38), size=3)) for _ in range(40)]
    ref_np = ref_data.cpu().float().numpy()
    ref_np = (ref_np + 0.5) * 255
    ref_np = ref_np*0.5
    mask_np = mask_data.cpu().float().numpy()

    for i in range(mask_np.shape[0]):
        for j in range(3):
            ref_np[j] = ref_np[j] + mask_np[i] * color[i][j] * 0.5
    ref_np = np.clip(ref_np, 0, 255)

    ref_np = ref_np.astype(np.uint8)

    return ref_np

def tensor2im(input_image, imtype=np.uint8, mask=False):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image
        else:
            return input_image
        # 如果数据不是numpy数组并且不是张量的话就直接输出(不是张量不是数组的话好像没什么了，因为输入到这里的肯定不是掩膜)
        image_numpy = image_tensor.cpu().detach().numpy()  # convert it into a numpy array
        # 将图片数据中的第一个转到cpu上转为浮点型numpy数组
        if len(image_numpy.shape) < 3:
            image_numpy = np.expand_dims(image_numpy, axis=0)
            # 如果数据只有两个维度的话就给他额外扩充一个维度
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # 这里检测三个维度中的第一个 如果第一个维度是1说明是灰度图，然后沿着第一个维度也就是channel维度扩展为3倍
        image_numpy = (image_numpy + 0.5) * 255
        image_numpy = np.clip(image_numpy, 0, 255)

        # 这里是将CHW 转为 HWC  并且将 -1~1 的数据转为0~2 再转为0~1 最后乘255
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)  # 最后返回无符号8类型的数据

def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_numpy = image_numpy[::-1].transpose((1, 2, 0))
    cv2.imwrite(image_path, image_numpy)




def resize(img, new_size,):
    img_size = img.shape[0:2]
    # HWC这里取得是HW
    r = min(new_size / img_size[0], new_size / img_size[1])
    # r选取出来的是最大的那条边与标准长度的比值
    new_unpad = int(round(img_size[0] * r)), int(round(img_size[1] * r))
    dh, dw = new_size - new_unpad[0], new_size - new_unpad[1]  # wh padding
    dh /= 2  # divide padding into 2 sides
    dw /= 2
    if img_size != (new_size, new_size):  # resize
        # new_unpad就是缩放后的图像大小
        if r > 1:
            img = cv2.resize(img, new_unpad[::-1], interpolation=cv2.INTER_LINEAR)
        else:
            img = cv2.resize(img, new_unpad[::-1], interpolation=cv2.INTER_AREA)
    # 后面的INTER_LINEAR指的是双线性插值法，注意这里的resize方法，他的图像大小的参数要求形式是WH的形式，而不能是HW，我们之前的到的new_unpad是HW的形式需要转换一下
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
    return img


def resize_mask(img, new_size):
    img_size = img.shape[0:2]
    # HWC这里取得是HW
    r = min(new_size / img_size[0], new_size / img_size[1])
    # r选取出来的是最大的那条边与标准长度的比值
    new_unpad = int(round(img_size[0] * r)), int(round(img_size[1] * r))
    dh, dw = new_size - new_unpad[0], new_size - new_unpad[1]  # wh padding
    dh /= 2  # divide padding into 2 sides
    dw /= 2
    if img_size != (new_size, new_size):  # resize
        # new_unpad就是缩放后的图像大小
        if r > 1:
            img = cv2.resize(img, new_unpad[::-1], interpolation=cv2.INTER_LINEAR)
        else:
            img = cv2.resize(img, new_unpad[::-1], interpolation=cv2.INTER_AREA)
    # 后面的INTER_LINEAR指的是双线性插值法，注意这里的resize方法，他的图像大小的参数要求形式是WH的形式，而不能是HW，我们之前的到的new_unpad是HW的形式需要转换一下
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # add border
    return img








###############################################################################################################
#               psnr                   ssim

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def SSIM(img1,img2):
    # pdb.set_trace()
    img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0)/255.0
    img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0)/255.0
    img1 = Variable( img1,  requires_grad=False)    # torch.Size([256, 256, 3])
    img2 = Variable( img2, requires_grad = False)
    # ssim_value = pytorch_ssim.ssim(img1, img2).item()
    ssim_value = float(ssim(img1, img2))
    # print(ssim_value)
    return ssim_value

def PSNR(img1, img2):
    # pdb.set_trace()
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
         return 100
    PIXEL_MAX = 255.0
    # PIXEL_MAX = 1.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    # print(psnr)
    return psnr

##################################################################################################################
'''
c = x[1]
c = c[0]
c =(c+0.5)*255
c = np.clip((c.cpu().numpy()), 0, 255)
c =c.astype(np.uint8)
c= c[::-1].transpose((1, 2, 0))
cv2.imshow('n02',c)
cv2.waitKey(0)
'''

