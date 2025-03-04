import torch.utils.data
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import os
import os.path as osp
import cv2
import numpy as np
from util.util import resize, resize_mask


IMG_FORMATS = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP', ]
# 所获取的原图像和对比图像的格式应在IMG_FORMATS参数指定范围内，如需添加其他格式文件，请将图片格式后缀名添加在此列表中


def create_traindataloader(opt):
    train_dataset = TrainValDataset(opt)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=TrainValDataset.collate_fn,
    )

    return train_dataloader, len(train_dataset)


# dataset
# 加载的时候按顺序获得图像原图、ref图、图像区域mask、细节原图、ref细节、颜色区域分割mask
class TrainValDataset(Dataset):

    # 这里是一个抽象的数据集基类
    # 如果要创建一个子类的话，就需要重构<__init__>、<__len__>、<__getitem__>、<modify_commandline_options>这几个方法
    # 其中前三个大家都知道  最后一个是前面已经使用过的用于增加控制变量的方法

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot
        # self.img_size = opt.crop_size

        # 图像大小等于图像裁剪大小的配置好的数据
        # 分别确定三个不同的路径

        self.data_paths = get_datapath(self.root)
        # raw_paths, ref_paths, mask_paths

        # 调用函数获取数据路径，其中make_dataset出现在util.util里面
        self.data_size = len(self.data_paths[0])  # get the size of dataset raw
        # 初始化了一下总数据量和输入输出通道数

    def __getitem__(self, index):
        """Return the total number of images in the dataset."""
        data_path = [p[index] for p in self.data_paths]  # make sure index is within then range
        data = DataProcess(data_path, self.opt)
        data = data + (data_path[0],)

        return data
        # raw, ref, raw_d, ref_d, mask, mask_img, path

    def __len__(self):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        return self.data_size

    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        # raw, ref, raw_d, ref_d, raw_blur, ref_blur, ref_localmean, ref_localdiff, mask, mask_img
        data = zip(*batch)
        raw, ref, raw_d, ref_d, raw_blur, ref_blur, ref_localmean, ref_localdiff, mask, mask_img, path = zip(*batch)
        # new_img = np.stack((Red, Green, Blue), axis=2)
        max = 0
        for m in mask:
            if max < m.shape[0]:
                max = m.shape[0]
        mask_list = []
        for m in mask:
            if max != m.shape[0]:
                mask_add = torch.zeros(((max - m.shape[0]), 3) + mask_img[0].shape[1:])
                m = torch.cat([m, mask_add], dim=0)
                mask_list.append(m)
            else:
                mask_list.append(m)

        return torch.stack(raw, 0), torch.stack(ref, 0), torch.stack(raw_d, 0), \
            torch.stack(ref_d, 0), torch.stack(raw_blur, 0), torch.stack(ref_blur, 0), \
            torch.stack(ref_localmean, 0), torch.stack(ref_localdiff, 0), \
                torch.stack(mask_list, 0).transpose(0, 1), torch.stack(mask_img, 0), path

def get_datapath(dataroot):
    dir_raw = os.path.join(dataroot, 'train\\raw\\img')
    dir_raw_blur = os.path.join(dataroot, 'train\\raw\\blur')
    dir_ref = os.path.join(dataroot, 'train\\ref\\img')
    dir_ref_blur = os.path.join(dataroot, 'train\\ref\\blur')
    dir_ref_locmean = os.path.join(dataroot, 'train\\ref\\local_mean')
    dir_mask = os.path.join(dataroot, 'train\\ref\\segment')
    raw_paths = glob.glob(dir_raw + '/*.*')
    raw_bpaths = glob.glob(dir_raw_blur + '/*.*')
    ref_paths = glob.glob(dir_ref + '/*.*')
    ref_bpaths= glob.glob(dir_ref_blur + '/*.*')
    ref_lmpaths = glob.glob(dir_ref_locmean + '/*.*')
    mask_paths = glob.glob(dir_mask + '/*.*')
    assert raw_paths, f"No images found in {dir_raw}."
    assert ref_paths, f"No images found in {dir_ref}."
    assert mask_paths, f"No images found in {dir_mask}."
    assert raw_bpaths, f"No images found in {dir_raw_blur}."
    assert ref_bpaths, f"No images found in {dir_ref_blur}."
    assert ref_lmpaths, f"No images found in {dir_ref_locmean}."
    raw_paths = sorted(
        p for p in raw_paths if p.split(".")[-1].lower() in IMG_FORMATS
    )
    ref_paths = sorted(
        p for p in ref_paths if p.split(".")[-1].lower() in IMG_FORMATS
    )
    mask_paths = sorted(
        p for p in mask_paths if p.split(".")[-1].lower() in IMG_FORMATS
    )
    raw_bpaths = sorted(
        p for p in raw_bpaths if p.split(".")[-1].lower() in IMG_FORMATS
    )
    ref_bpaths = sorted(
        p for p in ref_bpaths if p.split(".")[-1].lower() in IMG_FORMATS
    )
    ref_lmpaths = sorted(
        p for p in ref_lmpaths if p.split(".")[-1].lower() in IMG_FORMATS
    )

    return raw_paths, raw_bpaths, ref_paths, ref_bpaths, ref_lmpaths, mask_paths


def DataProcess(data_path, opt):
    new_size = opt.load_size
    raw = cv2.imread(data_path[0])
    raw_blur = cv2.imread(data_path[1])
    ref = cv2.imread(data_path[2])
    ref_blur = cv2.imread(data_path[3])
    ref_localmean = cv2.imread(data_path[4])
    mask = cv2.imread(data_path[5], cv2.IMREAD_GRAYSCALE)
    mask = mask + 1
    # 这里补充一位是为了显示扩充边缘的时候扩充的边缘
    # 也就是现在数据从1开始计算，后面resize的时候添加0



    raw = resize(raw, new_size)
    raw_blur = resize(raw_blur, new_size)
    ref = resize(ref, new_size)
    ref_blur = resize(ref_blur, new_size)
    ref_localmean = resize(ref_localmean, new_size)
    mask = resize_mask(mask, new_size)

    mask, mask_img = get_mask(mask)
    raw_d = raw.astype(np.float32) - raw_blur.astype(np.float32)
    ref_d = ref.astype(np.float32) - ref_blur.astype(np.float32)
    ref_localdiff = ref_blur.astype(np.float32) - ref_localmean.astype(np.float32)

    # 数据准备，包含原图，原图模糊图，原图细节图
    # 参考图，参考模糊图，参考细节图，参考局部均值，参考局部细节
    # raw,raw_blur,raw_detail
    # ref,ref_blur,ref_detail,ref_localmean,ref_localdiff

    # 整理数据格式维度以及内存规范
    raw = raw.transpose((2, 0, 1))[::-1]  # HWC BGR到 CHW RGB
    raw_d = raw_d.transpose((2, 0, 1))[::-1]
    raw = np.ascontiguousarray(raw)
    raw_d = np.ascontiguousarray(raw_d)
    ref = ref.transpose((2, 0, 1))[::-1]  # HWC BGR到 CHW RGB
    ref_d = ref_d.transpose((2, 0, 1))[::-1]
    ref = np.ascontiguousarray(ref)
    ref_d = np.ascontiguousarray(ref_d)
    raw_blur = raw_blur.transpose((2, 0, 1))[::-1]
    ref_blur = ref_blur.transpose((2, 0, 1))[::-1]
    ref_localmean = ref_localmean.transpose((2, 0, 1))[::-1]
    ref_localdiff = ref_localdiff.transpose((2, 0, 1))[::-1]
    raw_blur = np.ascontiguousarray(raw_blur)
    ref_blur = np.ascontiguousarray(ref_blur)
    ref_localmean = np.ascontiguousarray(ref_localmean)
    ref_localdiff = np.ascontiguousarray(ref_localdiff)



    # 格式转换以及归一化
    raw = torch.Tensor(raw)
    ref = torch.Tensor(ref)
    raw_d = torch.Tensor(raw_d)
    ref_d = torch.Tensor(ref_d)
    mask = torch.Tensor(mask)
    mask_img = torch.Tensor(mask_img)
    raw = raw / 255 - 0.5
    ref = ref / 255 - 0.5
    raw_d = raw_d / 255
    ref_d = ref_d / 255
    raw_blur = torch.Tensor(raw_blur)
    ref_blur = torch.Tensor(ref_blur)
    ref_localmean = torch.Tensor(ref_localmean)
    ref_localdiff = torch.Tensor(ref_localdiff)
    raw_blur = raw_blur / 255 - 0.5
    ref_blur = ref_blur / 255 - 0.5
    ref_localmean = ref_localmean / 255 - 0.5
    ref_localdiff = ref_localdiff / 255

    return raw, ref, raw_d, ref_d, raw_blur, ref_blur, ref_localmean, ref_localdiff, mask, mask_img
    # 这里的raw就是输入使用的原图，ref则是对比的清晰图像，raw_d、raw_blur分别是作为输入的细节图和模糊图，对应的ref_d、ref_blur
    # 就是清晰图像的细节图和对应的模糊图，ref_localmean则是按照分割提取出来的区域均值、ref_localdiff则是区域的变化情况，mask是
    # 用于分割不同区域的掩膜，mask_img则是用于提取放缩后图像的掩膜


def get_mask(mask):
    cal = np.unique(mask).tolist()
    mask_list = np.full((len(cal)-1, 3) + mask.shape, 0)
    img_mask = np.full((3,) + mask.shape, 1)
    # 这里会获得两个掩膜一个是图像所在区域掩膜，另一个是颜色区域掩膜

    for img_h in range(mask.shape[0]):
        for img_w in range(mask.shape[1]):
            if mask[img_h, img_w] == 0:
                img_mask[:, img_h, img_w] = 0
            elif mask[img_h, img_w] != 0:
                mask_list[cal.index(mask[img_h, img_w]) - 1, :, img_h, img_w] = 1

    return mask_list, img_mask


