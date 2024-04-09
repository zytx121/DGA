import torch
from piq import psnr, ssim

from .base_digital_attack import BaseDigitalAttack


class SINIFGSMDigitalAttack(BaseDigitalAttack):
    """ICLR 2020 https://arxiv.org/abs/1908.06281."""

    def __init__(self,
                 *args,
                 eps=16 / 255,
                 alpha=4 / 255,
                 steps=10,
                 decay=1.0,
                 m=5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.m = m

    def attack(self, data):
        info = dict()
        ori_images = data['inputs'].clone().detach()
        momentum = torch.zeros_like(data['inputs']).detach().to(
            data['inputs'].device)
        preds = self.model._run_forward(data, mode='predict')
        attack_mask = self.mask_generator(preds)

        for _ in range(self.steps):
            data['inputs'].requires_grad = True
            data[
                'inputs'] = data['inputs'] + self.decay * self.alpha * momentum

            # Calculate sum the gradients over the scale copies of the input
            adv_grad = torch.zeros_like(data['inputs']).detach().to(
                data['inputs'].device)

            for i in torch.arange(self.m):
                data['inputs'] = data['inputs'] / torch.pow(2, i)
                preds = self.model._run_forward(data, mode='attack')

                # if no detection, break
                num_det = preds[0].bboxes.shape[0]
                if num_det == 0:
                    break

                # compute loss
                loss = self.attack_loss(preds)

                # Update adversarial images
                adv_grad += torch.autograd.grad(
                    loss,
                    data['inputs'],
                    retain_graph=False,
                    create_graph=False)[0]

            adv_grad = adv_grad / self.m

            grad = self.decay*momentum + adv_grad / \
                torch.mean(torch.abs(adv_grad), dim=(1, 2, 3), keepdim=True)
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
