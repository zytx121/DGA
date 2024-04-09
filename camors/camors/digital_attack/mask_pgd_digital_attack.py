import numpy as np
import torch
from piq import psnr, ssim

from .base_digital_attack import BaseDigitalAttack


class MaskPGDDigitalAttack(BaseDigitalAttack):
    """https://arxiv.org/abs/1706.06083."""

    def __init__(self,
                 *args,
                 eps=16 / 255,
                 alpha=4 / 255,
                 steps=10,
                 random_start=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def attack(self, data):
        info = dict()
        ori_images = data['inputs'].clone().detach()
        preds = self.model._run_forward(data, mode='predict')
        attack_mask = self.mask_generator(preds).to(ori_images.device)
        ori_bboxes = preds[0].pred_instances.bboxes.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            data['inputs'] = data['inputs'] + \
                torch.empty_like(data['inputs']).uniform_(-self.eps, self.eps)
            data['inputs'] = torch.clamp(data['inputs'], min=0, max=1).detach()

        for step in range(self.steps):
            data['inputs'].requires_grad = True
            preds = self.model._run_forward(data, mode='attack')
            num_det = preds[0].bboxes.shape[0]

            # if no detection, adjust mask
            if num_det == 0:
                if step == 0:
                    break
                # last_attack_mask = attack_mask.clone().detach()
                attack_mask = self.adjust_attack_mask(attack_mask, ori_images,
                                                      data['inputs'],
                                                      ori_bboxes)
                data['inputs'] = ori_images + attack_mask * (
                    data['inputs'] - ori_images)
                break
                # mask_flag = True
                # print('attack_rate1: ', attack_rate)
                # if attack_rate < 0.01:
                #     break

            # compute loss
            loss = self.attack_loss(preds)

            # Update adversarial images
            grad = torch.autograd.grad(
                loss, data['inputs'], retain_graph=False,
                create_graph=False)[0]
            data['inputs'] = data['inputs'].detach(
            ) + self.alpha * grad.sign() * attack_mask
            delta = torch.clamp(
                data['inputs'] - ori_images, min=-self.eps, max=self.eps)
            data['inputs'] = torch.clamp(
                ori_images + delta, min=0, max=1).detach()

        # compute attack rate
        attack_mask = attack_mask.cpu().detach().numpy().transpose(1, 2, 0)
        attack_rate = attack_mask[attack_mask == 1].size / attack_mask.size
        info['attack_rate'] = attack_rate
        info['psnr'] = psnr(ori_images, data['inputs']).item()
        info['ssim'] = ssim(ori_images, data['inputs']).item()

        return data, info

    def adjust_attack_mask(self, attack_mask, ori_images, adv_images,
                           ori_bboxes):
        perturbation = attack_mask * (adv_images - ori_images)
        perturbation = perturbation.squeeze(
            0).cpu().detach().numpy().transpose(1, 2, 0)
        attack_mask = attack_mask.cpu().detach().numpy().transpose(1, 2, 0)
        # import cv2
        # import os
        # cv2.imwrite(os.path.join("./perturbation1.png"), attack_mask*100)
        perturbation_avg = np.abs(perturbation).sum() * 3 / perturbation[
            perturbation != np.array([0, 0, 0])].size
        decrease_index = np.where(
            (np.abs(perturbation[:, :, 0]) + np.abs(perturbation[:, :, 1]) +
             np.abs(perturbation[:, :, 2])) < perturbation_avg)
        attack_mask[decrease_index] = np.array([0, 0, 0])
        # cv2.imwrite(os.path.join("./perturbation.png"), attack_mask*100)
        # quit()
        # attack_rate =  attack_mask[attack_mask==1].size / attack_mask.size
        attack_mask = torch.from_numpy(attack_mask.transpose(2, 0, 1)).to(
            adv_images.device)
        return attack_mask

    # def adjust_attack_mask(self, attack_mask, ori_images, adv_images, ori_bboxes):
    #     attack_mask = attack_mask.cpu().detach().numpy().transpose(1, 2, 0)
    #     adv_images = adv_images.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    #     ori_images = ori_images.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    #     perturbation = np.zeros(ori_images.shape[:2])
    #     perturbation = np.stack((perturbation, perturbation, perturbation),axis=-1)
    #     for box in ori_bboxes:
    #         x1 = max(int(box[0]), 0)
    #         y1 = max(int(box[1]), 0)
    #         x2 = min(int(box[2]), ori_images.shape[0])
    #         y2 = min(int(box[3]), ori_images.shape[1])
    #         box_mask = np.zeros(ori_images.shape[:2])
    #         box_mask[y1:y2,x1:x2] = 1
    #         box_mask = np.stack((box_mask, box_mask, box_mask),axis=-1)
    #         box_perturbation = box_mask * (adv_images - ori_images)
    #         box_perturbation_avg = np.abs(box_perturbation).sum()*3 / box_perturbation[box_perturbation!=np.array([0,0,0])].size
    #         decrease_index = np.where((np.abs(box_perturbation[:,:,0])+np.abs(box_perturbation[:,:,1])+np.abs(box_perturbation[:,:,2]))<box_perturbation_avg)
    #         box_mask[decrease_index] = np.array([0,0,0])
    #         attack_mask[y1:y2,x1:x2,:] = box_mask[y1:y2,x1:x2,:]
    #     # import cv2
    #     # import os
    #     # cv2.imwrite(os.path.join("./perturbation.png"), attack_mask*100)
    #     # quit()
    #     attack_mask = torch.from_numpy(attack_mask.transpose(2, 0, 1)).to(ori_bboxes.device)
    #     return attack_mask
