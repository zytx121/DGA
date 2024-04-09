import torch
from piq import psnr, ssim

from .base_digital_attack import BaseDigitalAttack


class CamoDigitalAttack(BaseDigitalAttack):
    """https://arxiv.org/abs/1706.06083."""

    def __init__(self,
                 *args,
                 eps=16 / 255,
                 alpha=4 / 255,
                 steps=10,
                 random_start=False,
                 init_iters=2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.init_iters = init_iters

    def attack(self, data):
        info = dict()
        ori_images = data['inputs'].clone().detach()
        preds = self.model._run_forward(data, mode='predict')
        attack_mask = self.mask_generator(preds)

        if self.random_start:
            # Starting at a uniformly random point
            data['inputs'] = data['inputs'] + \
                torch.empty_like(data['inputs']).uniform_(-self.eps, self.eps)
            data['inputs'] = torch.clamp(data['inputs'], min=0, max=1).detach()

        for i in range(self.steps):
            data['inputs'].requires_grad = True

            if i < self.init_iters:
                _, cls_scores, _ = self.model._run_forward(data, mode='attack')
                randVector = torch.FloatTensor(cls_scores.shape).uniform_(
                    -1., 1.).to(cls_scores.device)
                loss = (cls_scores * randVector).sum()
            else:
                preds = self.model._run_forward(data, mode='attack')
                loss = self.attack_loss(preds)

            # Update adversarial images
            grad = torch.autograd.grad(
                loss, data['inputs'], retain_graph=False,
                create_graph=False)[0]
            data['inputs'] = data['inputs'].detach(
            ) + self.alpha * grad.sign() * attack_mask.to(grad.device)
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
