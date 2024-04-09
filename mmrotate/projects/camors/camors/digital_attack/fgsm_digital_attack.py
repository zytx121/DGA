import torch
from piq import psnr, ssim

from .base_digital_attack import BaseDigitalAttack


class FGSMDigitalAttack(BaseDigitalAttack):
    """https://arxiv.org/abs/1412.6572."""

    def __init__(self, *args, eps=16 / 255, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def attack(self, data):
        info = dict()
        ori_images = data['inputs'].clone().detach()
        # init mask
        preds = self.model._run_forward(data, mode='predict')
        attack_mask = self.mask_generator(preds)

        data['inputs'].requires_grad = True
        preds = self.model._run_forward(data, mode='attack')

        # compute loss
        loss = self.attack_loss(preds)

        # Update adversarial images
        grad = torch.autograd.grad(
            loss, data['inputs'], retain_graph=False, create_graph=False)[0]
        data['inputs'] = data['inputs'].detach(
        ) + self.eps * grad.sign() * attack_mask.to(grad.device)
        data['inputs'] = torch.clamp(data['inputs'], min=0, max=1).detach()

        # compute attack rate
        attack_mask = attack_mask.cpu().detach().numpy().transpose(1, 2, 0)
        attack_rate = attack_mask[attack_mask == 1].size / attack_mask.size
        info['attack_rate'] = attack_rate
        info['psnr'] = psnr(ori_images, data['inputs']).item()
        info['ssim'] = ssim(ori_images, data['inputs']).item()

        return data, info
