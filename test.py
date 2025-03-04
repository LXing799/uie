import time
from options.train_options import TrainOptions
from data.testdataloader import create_testdataloader
from model.modelUnet import create_model
from util.visualizer import Visualizer
from util.util import SSIM,PSNR


from tqdm import tqdm
import shutil


opt = TrainOptions().parse(False)
test_loader, testdataset_size = create_testdataloader(opt)  # create a dataset given opt.dataset_mode and other options

model = create_model(opt)
model.setup(opt)
# pbar = enumerate(test_loader)

for step, data in test_loader:
    model.set_input(data)  # unpack data from data loader
    model.test()
    # model.save_testdata()
    pred_ref, ref = model.get_output()

