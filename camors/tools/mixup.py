import os

import cv2
import numpy as np


def mixup_data(patch_path_x, patch_path_y, patch_dir):
    x = np.load(patch_path_x)
    y = np.load(patch_path_y)
    mixed = 0.5 * (x + y)
    save_patch_name = os.path.join(patch_dir, 'mixed.npy')
    np.save(save_patch_name, mixed)
    mean = np.array([0., 0., 0.])
    std = np.array([255., 255., 255.])
    mixed[0] = mixed[0] * std[0] + mean[0]
    mixed[1] = mixed[1] * std[1] + mean[1]
    mixed[2] = mixed[2] * std[2] + mean[2]
    mixed = mixed.transpose(1, 2, 0)
    save_patch_img = os.path.join(patch_dir, 'mixed.png')
    cv2.imwrite(save_patch_img, mixed)


if __name__ == '__main__':
    patch_path_x = '10.npy'
    patch_path_y = '30.npy'
    patch_dir = '.'
    mixup_data(patch_path_x, patch_path_y, patch_dir)
