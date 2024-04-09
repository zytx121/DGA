import numpy as np
import torch
import torch.nn.functional as F
from piq import psnr, ssim

from .base_digital_attack import BaseDigitalAttack


class TPADigitalAttack(BaseDigitalAttack):
    """https://arxiv.org/abs/1706.06083."""

    def __init__(self,
                 *args,
                 alpha=255 / 255,
                 steps=10,
                 random_start=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def attack(self, data):
        info = dict()
        ori_images = data['inputs'].clone().detach()
        preds = self.model._run_forward(data, mode='predict')
        attack_mask = self.mask_generator(preds).to(ori_images.device)
        ori_bboxes = preds[0].pred_instances.bboxes.clone().detach()

        # If no detection at the beginning, return directly.
        if ori_bboxes.shape[0] == 0:
            return data, info

        zero_iter = 0
        is_decrease = False
        last_attack_mask = None
        for step in range(self.steps):
            data['inputs'].requires_grad = True
            preds = self.model._run_forward(data, mode='attack')
            # num_det = preds[0].bboxes.shape[0]

            # Calculate the number of valid detection boxes.
            num_det = preds[0].scores[preds[0].scores > 0.05].shape[0]

            if num_det == 0:
                last_attack_mask = attack_mask.clone().detach()
                attack_mask = self.adjust_attack_mask(attack_mask, ori_images,
                                                      data['inputs'],
                                                      ori_bboxes)
                zero_iter = 0
                is_decrease = True

            zero_iter += 1
            if zero_iter >= 5 and last_attack_mask is not None:
                attack_mask = last_attack_mask
                zero_iter = 0
            elif zero_iter >= 5 and last_attack_mask is None:
                zero_iter = 0

            # compute loss
            preds = preds + (ori_bboxes, )
            loss = self.attack_loss(preds)

            # Update adversarial images
            grad = torch.autograd.grad(
                loss, data['inputs'], retain_graph=False,
                create_graph=False)[0]

            if is_decrease:
                data['inputs'] = ori_images + attack_mask * (
                    data['inputs'] - ori_images)
                is_decrease = False

            data['inputs'] = data['inputs'].detach(
            ) + self.alpha * grad.sign() * attack_mask
            data['inputs'] = torch.clamp(data['inputs'], min=0, max=1).detach()

        # compute attack rate
        attack_mask = attack_mask.cpu().detach().numpy().transpose(1, 2, 0)
        attack_rate = attack_mask[attack_mask == 1].size / attack_mask.size
        info['attack_rate'] = attack_rate
        # print('final attack_rate: ', attack_rate)
        info['psnr'] = psnr(ori_images, data['inputs']).item()
        info['ssim'] = ssim(ori_images, data['inputs']).item()

        return data, info

    def adjust_attack_mask(self, attack_mask, ori_images, adv_images,
                           ori_bboxes):
        perturbation = attack_mask * (adv_images - ori_images)
        perturbation = perturbation.squeeze(0).detach().permute(1, 2, 0)
        attack_mask = attack_mask.detach().permute(1, 2, 0)
        perturbation_avg = torch.abs(perturbation).sum() * 3 / perturbation[
            perturbation != torch.zeros((1, 3)).to(adv_images.device)].numel()
        decrease_index = torch.where(
            (torch.abs(perturbation[:, :, 0]) +
             torch.abs(perturbation[:, :, 1]) +
             torch.abs(perturbation[:, :, 2])) < perturbation_avg)
        attack_mask[decrease_index] = torch.zeros((1, 3)).to(adv_images.device)
        attack_mask = attack_mask.permute(2, 0, 1)
        return attack_mask

    # def adjust_attack_mask(self, attack_mask, ori_images, adv_images, ori_bboxes):
    #     perturbation = attack_mask * (adv_images - ori_images) # [1, 3, 256, 256]
    #     perturbation = perturbation.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0) # [256, 256, 3]
    #     attack_mask = attack_mask.cpu().detach().numpy().transpose(1, 2, 0) # [256, 256, 3]
    #     perturbation_avg = np.abs(perturbation).sum()*3 / perturbation[perturbation!=np.array([0,0,0])].size
    #     decrease_index = np.where((np.abs(perturbation[:,:,0])+np.abs(perturbation[:,:,1])+np.abs(perturbation[:,:,2]))<perturbation_avg)
    #     attack_mask[decrease_index] = np.array([0,0,0])
    #     attack_mask = torch.from_numpy(attack_mask.transpose(2, 0, 1)).to(adv_images.device)
    #     return attack_mask

    def attack_loss(self, preds):
        results, cls_scores_list, bbox_preds_list, ori_bboxes = preds

        # support one stage and two stage
        try:
            self.model.roi_head
            cls_scores = cls_scores_list
        except:
            mlvl_scores = []
            for cls_score_lvl in cls_scores_list:
                mlvl_scores.append(torch.flatten(cls_score_lvl))
            cls_scores = torch.cat(mlvl_scores)
            results = results[0]
        # det_scores = results.scores
        det_bboxes = results.bboxes
        # det_labels = results.labels

        iou_thre = 0.05
        if cls_scores[cls_scores >= iou_thre].shape[0] == 0:
            class_loss = F.mse_loss(cls_scores * 0,
                                    torch.zeros_like(cls_scores).cuda())
            iou_loss = torch.zeros([1]).cuda()
        else:
            class_loss = F.mse_loss(
                cls_scores[cls_scores >= iou_thre],
                torch.zeros(cls_scores[cls_scores >= iou_thre].shape).cuda())
            pred_iou = bbox_iou(det_bboxes, ori_bboxes)
            iou_loss = torch.sum(pred_iou) / det_bboxes.shape[0]
            loss = class_loss + iou_loss
        return loss


def bbox_iou(box1, box2):
    """Returns the IoU of two bounding boxes."""
    # Get the coordinates of bounding boxes
    xmin1 = box1[:, 0].unsqueeze(-1)
    ymin1 = box1[:, 1].unsqueeze(-1)
    xmax1 = box1[:, 2].unsqueeze(-1)
    ymax1 = box1[:, 3].unsqueeze(-1)

    xmin2 = box2[:, 0].unsqueeze(-1)
    ymin2 = box2[:, 1].unsqueeze(-1)
    xmax2 = box2[:, 2].unsqueeze(-1)
    ymax2 = box2[:, 3].unsqueeze(-1)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    ymin = torch.max(ymin1, torch.squeeze(ymin2, dim=-1))
    xmin = torch.max(xmin1, torch.squeeze(xmin2, dim=-1))
    ymax = torch.min(ymax1, torch.squeeze(ymax2, dim=-1))
    xmax = torch.min(xmax1, torch.squeeze(xmax2, dim=-1))

    h = torch.max(ymax - ymin, torch.zeros(ymax.shape).cuda())
    w = torch.max(xmax - xmin, torch.zeros(xmax.shape).cuda())
    intersect = h * w

    union = area1 + np.squeeze(area2, axis=-1) - intersect
    iou = intersect / union

    return iou
