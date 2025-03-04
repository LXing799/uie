import numpy as np
import os
import sys
import time
from subprocess import Popen, PIPE


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        # self.use_html = opt.isTrain  # and not opt.no_html
        self.win_size = opt.display_winsize
        # self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

            self.log_text = ''

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)


    def plot_current_losses(self, epoch, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        self.plot_many(losses, epoch)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, iters, t_comp)
        for i in range(0, len(losses), 2):
            message += '%s: %.3f ' % (losses[i], losses[i+1])

        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def plot_many(self, d, epoch):
        '''
        一次plot多个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        '''
        for i in range(0, len(d), 2):
            self.plot(d[i], d[i+1], epoch)


    def plot(self, name, y, epoch, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        self.vis.line(Y=np.array([y]), X=np.array([epoch]),
                      win=name,
                      opts=dict(title=name),
                      update=None if epoch == 0 else 'append',
                      **kwargs
                      )

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1, 'lr':0.0001})
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)