import os
import sys
import numpy as np
import cv2
sys.path.append('..')
from utils import dataset_nomask
from model.model_full_ours_gray import model as M
import glob

def saveImg(img, path):
    img[img > 1] = 1
    img[img < 0] = 0
    op = np.zeros([img.shape[2], img.shape[3], 3])
    img = img[0].cpu().permute(1,2,0) * 255
    op[:,:,0] = img[:,:,2]
    op[:,:,1] = img[:,:,1]
    op[:,:,2] = img[:,:,0]
    cv2.imwrite(path, op)

def showMessage(message, file):
    print(message)
    with open(file, 'a') as f:
        f.writelines(message + '\n')


def copyModel(rundir):
    os.makedirs(os.path.join(rundir, 'model'), exist_ok=True)
    for f in glob.glob(os.path.join('./model','*.py')):
        shutil.copy(f, os.path.join(rundir, 'model', os.path.split(f)[-1]))

    os.makedirs(os.path.join(rundir, 'utils'), exist_ok=True)
    for f in glob.glob(os.path.join('./utils','*.py')):
        shutil.copy(f, os.path.join(rundir,'utils',os.path.split(f)[-1]))


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'
    import shutil
    import torch
    from torch import nn
    from torch.optim import lr_scheduler
    import torch.optim as optim
    from tensorboardX import SummaryWriter
    from utils import EarlyStopping
    from utils import psnr_batch, ssim_batch
    from model import Metric, TimeRecorder
    import datetime
    from model.loss import Loss

    device = 'cuda'
    random_seed = 1996
    #batch_size = 16
    batch_size = 12
    lr = 2e-3
    alpha = 0.1
    gamma = 0.05
    epochs = 1200

    #run_dir = './log/' + datetime.date.today().__str__() + '_03'
    run_dir = './log/' + datetime.date.today().__str__() + '_05' + '_gray'
    print('rundir:', os.path.abspath(run_dir))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True

    #trainFolder = dataset('/home/lisiqi/data/SAIdata/train.txt')
    trainFolder = dataset_nomask('/home/event/projects_dir/dataset/Occlusion-400/train.txt')
    trainLoader = torch.utils.data.DataLoader(trainFolder, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=32, drop_last=True)
    #testFolder = dataset('/home/lisiqi/data/SAIdata/test.txt')
    testFolder = dataset_nomask('/home/event/projects_dir/dataset/Occlusion-400/test.txt')
    testLoader = torch.utils.data.DataLoader(testFolder, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=32)

    train_total_iter = len(trainLoader)
    test_total_iter = len(testLoader)
    print(train_total_iter, test_total_iter)

    #model = M(netParams={'Ts': 1, 'tSample': 40}, inChannels=33+11)
    model = M(netParams={'Ts': 1, 'tSample': 40}, inChannels=11)

    loss_l1 = nn.L1Loss()

    tb = SummaryWriter(run_dir)
    early_stopping = EarlyStopping(patience=epochs, verbose=True)

    if os.path.exists(os.path.join(run_dir, 'checkpoint.pth.tar')):
        print('==> loading existing model:', os.path.join(run_dir, 'checkpoint.pth.tar'))
        model_info = torch.load(os.path.join(run_dir, 'checkpoint.pth.tar'))
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir)
        cur_epoch = 0

    with open(os.path.join(run_dir, 'cfg.txt'), 'w') as f:
        f.writelines('lr %f, epoch %d, alpha %f\n' % (lr, epochs, alpha))
        f.writelines(model.__repr__())

    shutil.copy(os.path.abspath(__file__), os.path.join(run_dir, os.path.basename(__file__)))
    copyModel(run_dir)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-5)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-2)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    trainMetirc = Metric()
    testMetirc = Metric()

    psnr_list = []
    ssim_list = []
    TR = TimeRecorder(epochs-cur_epoch, train_total_iter+test_total_iter)

    for epoch in range(cur_epoch, epochs):
        model.train()

        for i, (event_vox, img, gt_img) in enumerate(trainLoader):
            event_vox = event_vox.cuda()
            img = img.cuda().float()
            gt_img = gt_img.cuda().float().mean(dim=1, keepdim=True)
            B, C, H, W = img.shape
            img = img.reshape(B, 11, -1, H, W).mean(dim=2, keepdim=False)

            output = model(event_vox, img)

            L1Loss = loss_l1(output, gt_img)

            Loss = L1Loss

            if Loss.item() >= 50:
                print('warning')
                with open(os.path.join(run_dir, 'log.txt'), 'a') as f:
                    f.writelines('warning\n')
                torch.cuda.empty_cache()
                continue

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            trainMetirc.update(L1Loss=L1Loss, total=Loss)

            if i % max(int(train_total_iter/2), 1) == 0:
                p = psnr_batch(output, gt_img)
                s = ssim_batch(output, gt_img)
                avg = trainMetirc.get_average_epoch()
                dt, remain_time, end_time = TR.get_iter_time(epoch=epoch - cur_epoch, iter=i)
                message = 'Train, Epoch: [%d]/[%d], Iter [%d]/[%d], L1 Loss:%f, Total Loss:%f, [%.3f,%.3f]' \
                          % (epoch, epochs, i, train_total_iter, avg[0], avg[1], p,s) + \
                          ", Cost Time: " + dt + ", Remain Time: " + remain_time + ', End At: ' + end_time
                showMessage(message, os.path.join(run_dir, 'log.txt'))

        scheduler.step()
        avg = trainMetirc.get_average_epoch()
        tb.add_scalar('Train_Loss/L1 Loss', avg[0], epoch)
        tb.add_scalar('Train_Loss/Total Loss', avg[1], epoch)
        trainMetirc.update_epoch()

        message = '============Train %d done, loss:%f============' % (epoch, avg[1])
        showMessage(message, os.path.join(run_dir, 'log.txt'))

        with torch.no_grad():
            model.eval()
            psnr = 0
            ssim = 0
            count = 0
            for i, (event_vox, img, gt_img) in enumerate(testLoader):
                event_vox = event_vox.cuda()
                img = img.cuda().float()
                gt_img = gt_img.cuda().float().mean(dim=1, keepdim=True)
                B, C, H, W = img.shape
                img = img.reshape(B, 11, -1, H, W).mean(dim=2, keepdim=False)
                output = model(event_vox, img)

                L1Loss = loss_l1(output, gt_img)
                Loss = L1Loss

                testMetirc.update(L1Loss=L1Loss, total=Loss)
                p = psnr_batch(output, gt_img)
                s = ssim_batch(output, gt_img)
                psnr += p
                ssim += s

                if count % 100 == 0:
                    avg = testMetirc.get_average_epoch()
                    message = 'Test, Epoch: [%d]/[%d], Iter [%d]/[%d], L1Loss:%f, TotalLoss:%f' \
                              % (epoch, epochs, count, test_total_iter, avg[0], avg[1])
                    showMessage(message, os.path.join(run_dir, 'log.txt'))
                count += 1
            torch.cuda.empty_cache()

            psnr /= count
            ssim /= count

            ssim_list.append(ssim)
            psnr_list.append(psnr)

            avg = testMetirc.get_average_epoch()
            tb.add_scalar('Test_Loss/L1 Loss', avg[0], epoch)
            tb.add_scalar('Test_Loss/Total Loss', avg[1], epoch)
            tb.add_scalar('Test_Loss/PSNR', psnr, epoch)
            tb.add_scalar('Test_Loss/SSIM', ssim, epoch)

            testMetirc.update_epoch()
            message = '============Epoch %d test done, loss:%f, PSNR:%f, SSIM:%f============' % (epoch, avg[1], psnr, ssim)
            showMessage(message, os.path.join(run_dir, 'log.txt'))

        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        tb.add_scalar('Lr', param_group['lr'], epoch)

        model_dict = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        early_stopping(avg[1], model_dict, epoch, run_dir)
        if ssim_list[-1] == max(ssim_list):
            torch.save(model_dict, os.path.join(run_dir, "checkpoint_max_ssim.pth"))
        if psnr_list[-1] == max(psnr_list):
            torch.save(model_dict, os.path.join(run_dir, "checkpoint_max_psnr.pth"))
        if early_stopping.early_stop:
            print('Stop!!!!')
            break