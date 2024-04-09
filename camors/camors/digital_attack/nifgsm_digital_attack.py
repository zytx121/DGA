import torch
from piq import psnr, ssim

from .base_digital_attack import BaseDigitalAttack


class NIFGSMDigitalAttack(BaseDigitalAttack):
    """ICLR 2020 https://arxiv.org/abs/1908.06281."""

    def __init__(self,
                 *args,
                 eps=16 / 255,
                 alpha=4 / 255,
                 steps=10,
                 decay=1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay

    def attack(self, data):
        info = dict()
        ori_images = data['inputs'].clone().detach()
        momentum = torch.zeros_like(data['inputs']).detach().to(
            data['inputs'].device)
        preds = self.model._run_forward(data, mode='predict')
        attack_mask = self.mask_generator(preds)

        for _ in range(self.steps):
            data['inputs'].requires_grad = True
            nes_data = data.copy()
            nes_data['inputs'] = nes_data[
                'inputs'] + self.decay * self.alpha * momentum
            preds = self.model._run_forward(nes_data, mode='attack')

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

            grad = self.decay * momentum + grad / torch.mean(
                torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            momentum = grad * attack_mask.to(grad.device)
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
