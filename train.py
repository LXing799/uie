import time
from options.train_options import TrainOptions
from data.dataloader import create_traindataloader
# from data.testdataloader import create_testdataloader
from model.modelUnet import create_model
from util.visualizer import Visualizer


# python -m visdom.server
# http://localhost:8097


from tqdm import tqdm
import shutil

NCOLS = shutil.get_terminal_size().columns

if __name__ == '__main__':
    opt = TrainOptions().parse(True)   # get training options
    # 相当于上来先获取模型、数据集配置信息，构建好所有的配置信息，然后再将信息打印保存，而后配置gpu
    # print(opt)
    # 这里看这个打印有可能相当于打印了两边信息，因为调用上面的方法的时候有一边美化后的打印，这个打印可以去掉

    train_loader, dataset_size = create_traindataloader(opt)  # create a dataset given opt.dataset_mode and other options
    # test_loader, testdataset_size = create_testdataloader(opt)
    model = create_model(opt)

    # 拆解一下这串格式化字符串的意思 首先是'%10s' + '%10.4g' * loss_num 这里其实是两段字符串，第一个表示一个占10个位置的字符串
    # 第二段表示 占位十个字符位置的保留四位小数的浮点数后面乘了一个整形数表示共有这样的数据几个
    model.setup(opt)
    visualizer = Visualizer(opt)
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=dataset_size/opt.batch_size, ncols=150, bar_format='{l_bar}{bar:10}{r_bar}{bar:-50b}')
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
            pbar.set_postfix(loss_total=current_losses[1].item(), loss_detail=current_losses[3].item(), loss_pixel=current_losses[5].item())
            if step % 30 == 0:
                t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, step, current_losses, t_comp)

            # pbar.set_description(('%10s' + '|%8s:%8.4g   |' * 3) % (f'{epoch}/{10 - 1}   ', *(losses)))
            # visualizer.plot_current_losses(epoch, losses)
            # if step%200==0:
            #     model.save_current_data(epoch)
            if step % 5 == 0:
                model.save_current_data(epoch)
            if step == 90:
                break


        epoch_losses = model.get_epoch_losses()
        visualizer.plot_current_losses(epoch, epoch_losses)
        if epoch % 5 == 0:              # cache our model every <save_epoch_freq> epochs
            model.save_networks('latest')
            model.save_networks(epoch)
        model.update_learning_rate()
        if epoch == 6:
            break
print(1)