import os

import cv2
import numpy as np
from mmdet.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner


@HOOKS.register_module()
class SavePatchHook(Hook):

    def __init__(self, *args, patch_dir, **kwargs):
        super().__init__()
        self.patch_dir = patch_dir

    def before_train_epoch(self, runner: Runner) -> None:
        model = runner.model
        epoch = runner.epoch
        if is_model_wrapper(model):
            model = model.module
        if epoch == 0:
            save_patch(model.patch_genetator.patch, epoch, self.patch_dir)

    def after_train_epoch(self, runner: Runner) -> None:
        model = runner.model
        epoch = runner.epoch
        if is_model_wrapper(model):
            model = model.module
        save_patch(model.patch_genetator.patch, epoch + 1, self.patch_dir)


def save_patch(patch, epoch, patch_dir):

    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)

    patch_np = patch.data.cpu().numpy()
    save_patch_name = os.path.join(patch_dir, '{}.npy'.format(epoch))

    np.save(save_patch_name, patch_np)
    mean = np.array([0., 0., 0.])
    std = np.array([255., 255., 255.])
    patch_np[0] = patch_np[0] * std[0] + mean[0]
    patch_np[1] = patch_np[1] * std[1] + mean[1]
    patch_np[2] = patch_np[2] * std[2] + mean[2]
    patch_np = patch_np.transpose(1, 2, 0)
    patch_np = patch_np[:, :, ::-1]
    save_patch_img = os.path.join(patch_dir, '{}.png'.format(epoch))
    cv2.imwrite(save_patch_img, patch_np)
