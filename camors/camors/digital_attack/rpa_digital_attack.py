import torch
from piq import psnr, ssim

from .base_digital_attack import BaseDigitalAttack


class RPADigitalAttack(BaseDigitalAttack):
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
        perturbation = attack_mask * (adv_images - ori_images
                                      )  # [1, 3, 256, 256]
        perturbation = perturbation.squeeze(0).detach().permute(
            1, 2, 0)  # [256, 256, 3]
        attack_mask = attack_mask.detach().permute(1, 2, 0)  # [256, 256, 3]
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
