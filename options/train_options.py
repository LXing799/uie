import argparse
import os
import torch

# 一般调这个都是直接调parse，然后parse里面又调了gather_options和打印
# 相当于上来先获取模型、数据集配置信息，构建好所有的配置信息，然后再将信息打印保存，而后配置gpu

class TrainOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """
    # 这个类中定义了训练和测试都需要用到的选项
    # 他也提供了一些辅助方法，例如解析 打印 存储 操作选项等
    # 同时也从命令行中收集有关配置数据集、模型两个类的有关配置信息

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        # 这个类是需要重写的 需要重写这个类来添加不在initialize初始化方法中出现的配置参数信息

    def initialize(self, parser, isTrain):
        """Define the common options that are used in both training and test."""
        # 定义了在训练和测试中都会用到的共有的配置选项信息
        # basic parameters
        parser.add_argument('--dataroot', type=str, default='./dataset', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='./outsample', help='models are saved here')
        # 基础参数： 数据集位置的根目录  实验的名字  gpu编号  模型参数保存目录

        # model parameters
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--netG_norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # 模型信息： 模型名称  输入的通道数  输出通道数  掩膜通道数  网络对应层数信息等  主干网络名称  批标准化方式(BN、instance还是直连)  --init_type参数初始化方法
        # --init_gain这应该是一个初始化参数的参数   不使用dropout

        # dataset parameters

        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=288, help='scale images to this size')
        # parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        # 数据集参数： 数据集类型   是否按顺序读入数据(只是说在dataloader的时候是否进行shuffle)  读入数据的线程数  BS这个大家都知道  load_size图像缩放大小  crop_size图像裁剪大小
        # 数据集最大数据量(目前没见过这种参数的作用)  预处理  no_flip不对图片进行旋转  打印窗口大小
        # additional parameters
        parser.add_argument('--epoch_start', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--epoch', type=int, default='0', help='The total number of model iterations')
        # 附加参数： epoch需要再执行多少周期  加载模型是通过训练的项目数还是epoch数  是否打印debug信息   自定义后缀
        # 这里的epoch_start是指加载哪个epoch的模型数据


        parser.add_argument('--display_freq', type=int, default=400,
                            help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=7,
                            help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost",
                            help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main',
                            help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--display_winsize', type=int, default=256,
                            help='display window size for both visdom and HTML')
        # 新的配置信息包含 打印频率  将所有图片显示到一个页面并且限制每行的图片数量，这个配置信息就是每行允许的最大数量  网页显示的id号
        # 打印服务器  打印环境名称   网页打印的端口号  向网页保存结果的频率   在命令行显示结果的频率   不启用网页保存训练结果选项
        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=10,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
        # 保存检查点的频率  是否按迭代步数保存数据   是否继续训练，也就是是否读取之前实验的最后一轮参数    起始训练轮数  训练的阶段
        # training parameters
        parser.add_argument('--niter', type=int, default=30, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=120,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        # niter是指从多少步开始调整学习率、niter_decay是指多少步将学习率调整完    后面是配置优化器的参数          lr_policy学习率模式   lr_decay_iters每固定步数就乘gamma
        self.isTrain = isTrain
        self.initialized = True
        return parser

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        # 这里是打印和保存配置信息   上面说他会同时打印现有的配置信息和配置信息的默认值
        # 他会将配置信息保存到一个text文件中 一般实在checkpoints_dir目录下的opt.txt这个文件中
        message = ''
        message += '--------------------------- Options --------------------------\n'
        # 这里我其实本意是想对他进行更完善一点的操作，也就是尽可能的把关键变量给保存下来，而后加载模型的时候可以直接根据
        # 实验名索引到对应的文本文档，将变量加载进来，目前感觉比较复杂，先看看架构提供了哪些东西再说

        # 首先在最开始创建了一个字符串，然后加了一些美化用的符号   而后添加配置选项信息  最后再加一个美化符号结尾
        # 每个配置信息独占一行

        # 下面这些循环里面 首先是遍历parser的那个类把所有的配置信息中对应的变量名和变量信息读出来
        # 而后default则包含循环对应的配置变量的默认值 只有当默认值和当前值不同的时候才会记录默认值
        # 打印这一段的结尾会直接把结果打印出来
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>20}: {:<30}{}\n'.format(str(k), str(v), comment)
            # 这段格式化字符串的作用是指 k也就是配置变量名长度为25并且右对齐  同理变量信息就是长30并且左对齐   而后对默认值位置、长度不做要求
        message += '--------------------------- End -----------------------------'
        print(message)

        # save to the disk
        # 将上述信息保存到硬盘中

        # 前面是最常见的目录拼接，后面直接在checkpoints_dir目录下面创建了一个以name为名字的新目录
        # 配置信息记录名则是直接在上面创建出来的新的文件夹内创建一个新的文本文档 名字使用phase和_opt.txt拼接而得
        # 然后将上面用于打印的信息存储起来，直接就是一个最基础的文本文档写操作，最后注意多了一个换行符
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'train_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, isTrain):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        # 解析选项 创建检查点目录 并且配置gpu

        # 这里首先调用了上面写的函数，获取数据集、模型内部的新配置信息，而后返回配置信息变量
        # 这个isTrain信息是后面测试、训练配置继承该类时新添的
        # opt = self.gather_options()
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser, isTrain)
        self.parser = parser
        opt = parser.parse_args()
        opt.isTrain = self.isTrain   # train or test
        opt.results_name = opt.name
        # 如果有后缀的话就将结果名加上_和后缀名

        self.print_options(opt)

        device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        torch.cuda.set_device(device)

        self.opt = opt
        return self.opt
