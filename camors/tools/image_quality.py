import glob
import os

import cv2
import torch
from piq import psnr, ssim
from rich.progress import track


def toPSNR(img1, img2):
    img1 = torch.from_numpy(img1.transpose((2, 0, 1))).unsqueeze(0).cuda()
    img2 = torch.from_numpy(img2.transpose((2, 0, 1))).unsqueeze(0).cuda()
    return psnr(img1, img2, data_range=255).item()


def toSSIM(img1, img2):
    img1 = torch.from_numpy(img1.transpose((2, 0, 1))).unsqueeze(0).cuda()
    img2 = torch.from_numpy(img2.transpose((2, 0, 1))).unsqueeze(0).cuda()
    return ssim(img1, img2, data_range=255).item()


if __name__ == '__main__':
    ori_path = '/home/zytx121/mmrotate/data/DOTA/split_ss_dota/val/plane'
    adv_path = '/home/zytx121/public/faster_rcnn_plane_vmifgsm/vis'
    ori_imgs = glob.glob(ori_path + '/*.png')
    img_names = [img.split('/')[-1] for img in ori_imgs]

    PSNR_list = []
    SSIM_list = []

    for ori_img in track(ori_imgs):
        original_img = cv2.imread(ori_img)
        adv_img = cv2.imread(os.path.join(adv_path, ori_img.split('/')[-1]))
        PSNR_list.append(toPSNR(original_img, adv_img))
        SSIM_list.append(toSSIM(original_img, adv_img))

    print('PSNR: ', sum(PSNR_list) / len(PSNR_list))
    print('SSIM: ', sum(SSIM_list) / len(SSIM_list))
