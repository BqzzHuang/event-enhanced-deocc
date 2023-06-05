import sys
import numpy as np
sys.path.append('..')
from utils.dataset import dataset as dataset
from model.model_full_ours import model as M
import cv2


def saveImg(img, path):
    img[img > 1] = 1
    img[img < 0] = 0
    op = np.zeros([img.shape[2], img.shape[3], 3])
    img = img[0].cpu().permute(1,2,0) * 255
    op[:,:,0] = img[:,:,2]
    op[:,:,1] = img[:,:,1]
    op[:,:,2] = img[:,:,0]
    cv2.imwrite(path, op)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'
    import torch
    from torch import nn
    from PIL import Image
    from utils import psnr_batch, ssim_batch
    import cv2

    device = 'cuda'

    run_dir = './log/2023-04-25_02'

    print(1)
    testFolder = dataset('/home/event/projects_dir/dataset/Occlusion-400/test.txt')
    testLoader = torch.utils.data.DataLoader(testFolder, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)

    model = M(netParams={'Ts': 1, 'tSample': 40}, inChannels=33 + 11)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    print('==> loading existing model:', os.path.join(run_dir, 'checkpoint_max_ssim.pth'))
    model_info = torch.load(os.path.join(run_dir, 'checkpoint_max_psnr.pth'))
    model.load_state_dict(model_info['state_dict'])

    for i, (event_vox, img, gt_img, mask) in enumerate(testLoader):
        with torch.no_grad():
            model.eval()
            psnr = 0
            ssim = 0
            count = 0

            psnr_indoor = 0
            ssim_indoor = 0
            count_indoor = 0

            psnr_outdoor = 0
            ssim_outdoor = 0
            count_outdoor = 0

            event_vox = event_vox.cuda()
            print(img.shape)
            saveImg(img[:,15:18,:,:], '/home/event/projects_dir/event_deocc/test_img_fig_5.jpg')
            img = img.cuda().float()
            mask = mask.cuda().float()
            gt_img = gt_img.cuda().float()

            mask = torch.index_select(mask, 1, torch.tensor([1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31]).cuda())
            output = model(event_vox, torch.cat([img, mask], dim=1))

            p = psnr_batch(output, gt_img)
            s = ssim_batch(output, gt_img)

            print(i, p, s)

        break
