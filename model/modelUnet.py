import os
import torch
from collections import OrderedDict
from model.layers import get_model, get_scheduler
import cv2
import numpy as np
from util import util

def create_model(opt):
    return Model(opt)


class Model():
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )  # get device name: CPU or GPU

        self.batch_size = opt.batch_size
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.loss_names = ['loss_total', 'loss_detail', 'loss_pixel', 'loss_localmean', 'loss_blur', 'loss_localdiff', 'loss_m']
        self.epoch_loss_names = ['epoch_loss_total', 'epoch_loss_detail', 'epoch_loss_pixel',
                                 'epoch_loss_localmean', 'epoch_loss_blur', 'epoch_loss_localdiff', 'epoch_loss_m']
        # self.visual_names = ['raw', 'gt_mask', 'pred_enhancement', 'ref_enhancement']
        self.visual_names = ['raw', 'pred_ref', 'ref', 'raw_d', 'pred_ref_d', 'ref_d', 'mask']
        self.net = get_model(opt.init_type, opt.init_gain, self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.image_paths = []
        # 这个图片路径指的是这个批次的图片路径
        self.metric = 0  # used for learning rate policy 'plateau'

        # define networks
        # 就是获得对应的模型
        self.train_data_dir = os.path.join(opt.checkpoints_dir, opt.name, 'train_data')
        if not os.path.exists(self.train_data_dir):
            os.makedirs(self.train_data_dir)

        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss(reduction='sum')

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # 首先是如果是在训练阶段那么就将优化器里面的学习率替换为
        if self.isTrain:
            self.scheduler = get_scheduler(self.optimizer, opt)
        # 如果不是训练阶段或者是继续训练的话 加载对应的模型数据
        if not self.isTrain:
            self.load_networks('latest')
            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_data')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        self.print_networks()

    def print_networks(self):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in self.net.parameters():
            num_params += param.numel()
        print(self.net)
        print('[Network %s] Total number of parameters : %.3f M' % ('MlUNet', num_params / 1e6))
        print('-----------------------------------------------')

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.epoch_loss_total = 0
        self.epoch_loss_detail = 0
        self.epoch_loss_pixel = 0
        self.epoch_loss_localmean = 0
        self.epoch_loss_blur = 0
        self.epoch_loss_localdiff = 0
        self.epoch_loss_m = 0

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        # raw, ref, raw_d, ref_d, raw_blur, ref_blur, ref_localmean, ref_localdiff, mask, mask_img
        if self.isTrain:
            self.raw = input[0].to(self.device)
            self.ref = input[1].to(self.device)
            self.raw_d = input[2].to(self.device)
            self.ref_d = input[3].to(self.device)
            self.raw_b = input[4].to(self.device)
            self.ref_b = input[5].to(self.device)
            self.ref_localmean = input[6].to(self.device)
            self.ref_localdiff = input[7].to(self.device)
            self.mask = input[8].to(self.device)
            self.mask_img = input[9].to(self.device)
            self.image_paths = input[10]
        if not self.isTrain:
            self.raw = input[0].to(self.device)
            self.ref = input[1].to(self.device)
            self.raw_d = input[2].to(self.device)
            self.ref_d = input[3].to(self.device)
            self.raw_b = input[4].to(self.device)
            self.ref_b = input[5].to(self.device)
            self.ref_localmean = input[6].to(self.device)
            self.ref_localdiff = input[7].to(self.device)
            self.mask_img = input[9].to(self.device)
            self.image_paths = input[10]

    def optimize_parameters(self, epoch, total_epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #self.pred_ref, self.pred_ref_d, self.pred_localdiff, self.pred_localmean, self.pred_ref_b = self.net(self.raw, self.raw_d, self.raw_b)
        self.pred_ref, self.pred_ref_d, self.pred_localdiff, self.pred_localmean, self.pred_ref_b = self.net([self.raw_d,
                                                                                                             self.raw_b])

    def backward(self):
        # l2 loss
        point_sum = self.mask_img.sum()
        # 首先需要我们获得真实的图像有多少个像素点，用于求取单个像素上的损失
        self.loss_pixel = self.criterionL2(self.pred_ref * self.mask_img, self.ref * self.mask_img) / point_sum
        self.loss_detail = self.criterionL2(self.pred_ref_d * self.mask_img, self.ref_d * self.mask_img)/ point_sum
        self.loss_localmean = self.criterionL2(self.pred_localmean * self.mask_img, self.ref_localmean * self.mask_img)/ point_sum
        self.loss_localdiff = self.criterionL2(self.pred_localdiff * self.mask_img, self.ref_localdiff * self.mask_img)/ point_sum
        self.loss_blur = self.criterionL2(self.pred_ref_b * self.mask_img, self.ref_b * self.mask_img)/ point_sum
        self.loss_m = 0
        for m in self.mask:
            self.loss_m = self.loss_m + self.criterionL2(self.pred_ref * m, self.ref * m)/m.sum()

        self.loss_total = self.loss_pixel*6 + self.loss_detail * 20 + self.loss_blur * 6 + self.loss_m
        # self.loss_total = self.loss_pixel*6 + self.loss_detail * 6 + self.loss_localmean + self.loss_localdiff + self.loss_blur * 6 + self.loss_m


        self.epoch_loss_total = self.epoch_loss_total + self.loss_total.item()
        self.epoch_loss_detail = self.epoch_loss_detail + self.loss_detail.item()
        self.epoch_loss_pixel = self.epoch_loss_pixel + self.loss_pixel.item()
        self.epoch_loss_localmean = self.epoch_loss_localmean + self.loss_localmean.item()
        self.epoch_loss_blur = self.epoch_loss_blur + self.loss_blur.item()
        self.epoch_loss_localdiff = self.epoch_loss_localdiff + self.loss_localdiff.item()
        self.epoch_loss_m = self.epoch_loss_m + self.loss_m.item()

        self.loss_total.backward()

    def save_current_data(self, epoch):
        img_rawlist = [self.raw[0].detach().clone(), self.raw_d[0].detach().clone(), self.raw_b[0].detach().clone()]
        img_predlist = [self.pred_ref[0].detach().clone(), self.pred_ref_d[0].detach().clone(), self.pred_ref_b[0].detach().clone(),
                        self.pred_localdiff[0].detach().clone(), self.pred_localmean[0].detach().clone()]
        img_reflist = [self.ref[0].detach().clone(), self.ref_d[0].detach().clone(), self.ref_b[0].detach().clone(),
                       self.ref_localdiff[0].detach().clone(), self.ref_localmean[0].detach().clone()]
        outraw = []
        outpred = []
        out_ref = []
        for img in img_rawlist:
            new_img = util.tensor2im(img)
            new_img = new_img[::-1].transpose((1, 2, 0))
            outraw.append(new_img)
        for img in img_predlist:
            new_img = util.tensor2im(img)
            new_img = new_img[::-1].transpose((1, 2, 0))
            outpred.append(new_img)
        for img in img_reflist:
            new_img = util.tensor2im(img)
            new_img = new_img[::-1].transpose((1, 2, 0))
            out_ref.append(new_img)

        img_pred = np.hstack(outpred)
        img_ref = np.hstack(out_ref)
        img_raw = np.hstack(outraw)
        img_raw = np.pad(img_raw, ((0, 0), (0, img_ref.shape[1] - img_raw.shape[1]), (0, 0)), 'constant', constant_values=(114))
        im_s = np.vstack([img_raw, img_pred, img_ref])

        if self.isTrain:
            img_path = os.path.join(self.train_data_dir, 'epoch%.3d_%s' % (epoch, os.path.basename(self.image_paths[0])))
        if not self.isTrain:
            self.testinfo_dir = os.path.join(self.save_dir, 'testdatainfo', 'epoch%d' % epoch)
            img_path = os.path.join(self.testinfo_dir, os.path.basename(self.image_paths[0]))
            if not os.path.exists(self.testinfo_dir):
                os.makedirs(self.testinfo_dir)
        cv2.imwrite(img_path, im_s)


    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        loss_list = []
        # loss_name = []
        for name in self.loss_names:
            if isinstance(name, str):
                loss_list.append(name)
                loss_list.append(getattr(self, name, 0.0))
                  # float(...) works for both scalar tensor and float number
        # 这里返回的信息只有pixel，这个损失其实是全图损失和局域损失之和
        return loss_list

    def get_epoch_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        loss_list = []
        # loss_name = []
        for name in self.epoch_loss_names:
            if isinstance(name, str):
                loss_list.append(name)
                loss_list.append(getattr(self, name, 0.0))
                  # float(...) works for both scalar tensor and float number
        # 这里返回的信息只有pixel，这个损失其实是全图损失和局域损失之和
        return loss_list

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""

        if self.opt.lr_policy == 'plateau':
            self.scheduler.step(self.metric)
        else:
            self.scheduler.step()

        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        save_filename = '%s_net_%s.pth' % (epoch, 'MlUNet')
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(self.net.state_dict(), save_path)

    def eval(self):
        """Make models eval mode during test time"""
        self.net.eval()
        self.isTrain = False

    def train(self):
        """Make models eval mode during test time"""
        self.net.train()
        self.isTrain = True

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()



    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        load_filename = '%s_net_%s.pth' % (epoch, 'MlUNet')
        load_path = os.path.join(self.save_dir, load_filename)
        self.net.load_state_dict(torch.load(load_path))

    def save_testdata(self):
        for i in range(len(self.pred_ref)):
            new_img = util.tensor2im(self.pred_ref[i])
            new_img = new_img[::-1].transpose((1, 2, 0))
            img_path = os.path.join(self.save_dir, os.path.basename(self.image_paths[i]))
            cv2.imwrite(img_path, new_img)

    def get_output(self):
        pred_list = []
        ref_list = []
        for i in range(len(self.pred_ref)):
            pred_img = util.tensor2im(self.pred_ref[i])
            ref = util.tensor2im(self.ref[i])
            pred_img = pred_img[::-1].transpose((1, 2, 0))
            ref = ref[::-1].transpose((1, 2, 0))
            pred_img = np.ascontiguousarray(pred_img)
            ref = np.ascontiguousarray(ref)
            pred_list.append(pred_img)
            ref_list.append(ref)

        return pred_list, ref_list

    def save_testtraindata(self, epoch):
        self.forward()
        self.testtrain_dir = os.path.join(self.save_dir, 'testdata', 'epoch%d' % epoch)
        if not os.path.exists(self.testtrain_dir):
            os.makedirs(self.testtrain_dir)
        for i in range(len(self.pred_ref)):
            new_img = util.tensor2im(self.pred_ref[i])
            new_img = new_img[::-1].transpose((1, 2, 0))
            img_path = os.path.join(self.testtrain_dir, os.path.basename(self.image_paths[i]))
            cv2.imwrite(img_path, new_img)

    def save_testloss(self):
        point_sum = self.mask_img.sum()
        # 首先需要我们获得真实的图像有多少个像素点，用于求取单个像素上的损失
        self.loss_pixel = self.criterionL2(self.pred_ref_d * self.mask_img, self.ref_d * self.mask_img) / point_sum
        self.loss_detail = self.criterionL2(self.pred_ref_d * self.mask_img, self.ref_d * self.mask_img) / point_sum
        self.loss_localmean = self.criterionL2(self.pred_localmean * self.mask_img,
                                               self.ref_localmean * self.mask_img) / point_sum
        self.loss_localdiff = self.criterionL2(self.pred_localdiff * self.mask_img,
                                               self.ref_localdiff * self.mask_img) / point_sum
        self.loss_blur = self.criterionL2(self.pred_ref_b * self.mask_img, self.ref_b * self.mask_img) / point_sum


    def save_train_data(self, epoch):
        self.train_dir = os.path.join(self.save_dir, 'savedata', 'epoch%d' % epoch)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        for i in range(len(self.pred_ref)):
            new_img = util.tensor2im(self.pred_ref[i])
            new_img = new_img[::-1].transpose((1, 2, 0))
            img_path = os.path.join(self.train_dir, os.path.basename(self.image_paths[i]))
            cv2.imwrite(img_path, new_img)


