import torch
from piq import psnr, ssim

from .base_digital_attack import BaseDigitalAttack


class BIMDigitalAttack(BaseDigitalAttack):
    """https://arxiv.org/abs/1607.02533."""

    def __init__(self, *args, eps=16 / 255, alpha=4 / 255, steps=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def attack(self, data):
        info = dict()
        ori_images = data['inputs'].clone().detach()
        preds = self.model._run_forward(data, mode='predict')
        attack_mask = self.mask_generator(preds)

        for _ in range(self.steps):
            data['inputs'].requires_grad = True
            preds = self.model._run_forward(data, mode='attack')

            # if no detection, break
            num_det = preds[0].bboxes.shape[0]
            if num_det == 0:
                break

            # compute loss
            loss = self.attack_loss(preds)

            # Update adversarial images
            grad = torch.autograd.grad(
                loss, data['inputs'], retain_graph=False,
                create_graph=False)[0]

            data['inputs'] = data['inputs'] + self.alpha * grad.sign(
            ) * attack_mask.to(grad.device)
            a = torch.clamp(ori_images - self.eps, min=0)
            b = (data['inputs'] >=
                 a).float() * data['inputs'] + (data['inputs'] < a).float() * a
            c = (b > ori_images + self.eps).float() * (
                ori_images +
                self.eps) + (b <= ori_images + self.eps).float() * b
            data['inputs'] = torch.clamp(c, max=1).detach()

        # compute attack rate
        attack_mask = attack_mask.cpu().detach().numpy().transpose(1, 2, 0)
        attack_rate = attack_mask[attack_mask == 1].size / attack_mask.size
        info['attack_rate'] = attack_rate
        info['psnr'] = psnr(ori_images, data['inputs']).item()
        info['ssim'] = ssim(ori_images, data['inputs']).item()

        return data, info
