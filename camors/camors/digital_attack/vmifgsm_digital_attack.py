import torch
from piq import psnr, ssim

from .base_digital_attack import BaseDigitalAttack


class VMIFGSMDigitalAttack(BaseDigitalAttack):
    """CVPR 2021 https://arxiv.org/abs/2103.15571."""

    def __init__(self,
                 *args,
                 eps=16 / 255,
                 alpha=4 / 255,
                 steps=10,
                 decay=1.0,
                 N=5,
                 beta=3 / 2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.N = N
        self.beta = beta

    def attack(self, data):
        info = dict()
        ori_images = data['inputs'].clone().detach()
        momentum = torch.zeros_like(ori_images).detach().to(ori_images.device)
        v = torch.zeros_like(ori_images).detach().to(ori_images.device)
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
            adv_grad = torch.autograd.grad(
                loss, data['inputs'], retain_graph=False,
                create_graph=False)[0]

            grad = (adv_grad + v) / torch.mean(
                torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            # Calculate Gradient Variance
            neighbor_data = data.copy()
            GV_grad = torch.zeros_like(ori_images).detach().to(
                ori_images.device)
            for _ in range(self.N):
                neighbor_data['inputs'] = neighbor_data['inputs'].detach() + \
                    torch.randn_like(ori_images).uniform_(
                        -self.eps * self.beta,
                        self.eps*self.beta)
                neighbor_data['inputs'].requires_grad = True

                preds = self.model._run_forward(neighbor_data, mode='attack')

                # compute loss
                loss = self.attack_loss(preds)

                GV_grad += torch.autograd.grad(
                    loss,
                    neighbor_data['inputs'],
                    retain_graph=False,
                    create_graph=False)[0]
            # obtaining the gradient variance
            v = GV_grad / self.N - adv_grad

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
