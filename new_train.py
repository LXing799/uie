import time
import os
from options.train_options import TrainOptions
from data.dataloader import create_traindataloader
from data.testdataloader import create_testdataloader
from model.modelUnet import create_model
from util.visualizer import Visualizer
from random import randint
from util.util import SSIM, PSNR

from outsample.resault import resault_epoch
# python -m visdom.server
# http://localhost:8097


from tqdm import tqdm
import shutil

if __name__ == '__main__':
    opt = TrainOptions().parse(True)
    train_loader, dataset_size = create_traindataloader(opt)
    test_loader, testdataset_size = create_testdataloader(opt)
    # data = next(iter(train_loader))
    model = create_model(opt)

    model.setup(opt)
    visualizer = Visualizer(opt)
    psnr_list = []
    ssim_list = []


    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=dataset_size / opt.batch_size, ncols=150, bar_format='{l_bar}{bar:10}{r_bar}{bar:-50b}')
        model.set_epoch(epoch)
        epoch_iter = 0
        for step, data in pbar:
            iter_start_time = time.time()
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters(epoch, opt.niter + opt.niter_decay)
            # visualizer.display_current_results(model.get_current_visuals(), epoch, True)
            current_losses = model.get_current_losses()
            pbar.set_description('%10s' % f'{epoch}/{opt.niter + opt.niter_decay + 1}')
            pbar.set_postfix(loss_total=current_losses[1].item(), loss_detail=current_losses[3].item(),
                             loss_pixel=current_losses[5].item())
            model.save_train_data(epoch)
            if step % 30 == 0:
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, step, current_losses, t_comp)

            if step % 50 == 0:
                model.save_current_data(epoch)

            # if (step+1) % 400 == 0:
            #    break

        epoch_losses = model.get_epoch_losses()
        visualizer.plot_current_losses(epoch, epoch_losses)
        if epoch % 5 == 0:  # cache our model every <save_epoch_freq> epochs
            model.save_networks('latest')
            model.save_networks(epoch)
        model.update_learning_rate()


"""
        if epoch % 12 == 0:
            model.eval()
            for step, data in enumerate(test_loader):
                r_num = randint(1, 60)
                model.set_input(data)  # unpack data from data loader
                # model.save_testdata()
                model.test()
                pred_ref, ref = model.get_output()
                if step % 60 == r_num:
                    model.save_current_data(epoch)

                model.save_testtraindata(epoch)
            if epoch % 4 == 0:
                epochname = os.path.join(opt.checkpoints_dir, opt.name)
                epochname = epochname + '/testdata/epoch' + '%d' % epoch
                p_epoch, n_epoch, test_item = resault_epoch(epochname)

                if epoch % 40 == 0:
                    with open('./%s%d.txt' % (opt.name, epoch), "a") as log_file:
                        log_file.write('*' * 80 + '\n')
                        log_file.write('*' * 35 + 'psnr' + '*' * 35 + '\n')
                        for line in psnr_list:
                            log_file.write(str(line) + '\n')
                        log_file.write('*' * 80 + '\n\n\n')
                        log_file.write('*' * 80 + '\n')
                        log_file.write('*' * 35 + 'ssim' + '*' * 35 + '\n')
                        for line in ssim_list:
                            log_file.write(str(line) + '\n')
                        log_file.write('*' * 80)

                psnr_list.append(sum(p_epoch) / test_item)
                ssim_list.append(sum(n_epoch) / test_item)
                print("psnr:", sum(p_epoch) / test_item)
            model.train()
        # if epoch == 6:
        #     break
    with open('./%s.txt' % opt.name, "a") as log_file:
        log_file.write('*' * 80 + '\n')
        log_file.write('*' * 35 + 'psnr' + '*' * 35 + '\n')
        for line in psnr_list:
            log_file.write(str(line) + '\n')
        log_file.write('*' * 80 + '\n\n\n')
        log_file.write('*' * 80 + '\n')
        log_file.write('*' * 35 + 'ssim' + '*' * 35 + '\n')
        for line in ssim_list:
            log_file.write(str(line) + '\n')
        log_file.write('*' * 80)
        # log_file.write('%s\n' % message)
"""