import torch
from piq import psnr, ssim

from .base_digital_attack import BaseDigitalAttack
from load_dm import get_imagenet_dm_conf


class Denoised_Model(torch.nn.Module):
    def __init__(self, diffusion, model, t):
        super().__init__()
        self.model, self.diffusion = get_imagenet_dm_conf(device=0, respace='ddim50')
        self.t = t
    
    def sdedit(self, x, to_01=True):

        # assume the input is 0-1
        t_int = self.t
        
        x = x * 2 - 1
        
        t = torch.full((x.shape[0], ), self.t).long().to(x.device)
    
        x_t = self.diffusion.q_sample(x, t) 
        
        sample = x_t
    
        # print(x_t.min(), x_t.max())
    
        # si(x_t, 'vis/noised_x.png', to_01=True)
        
        indices = list(range(t+1))[::-1]

        # visualize 
        l_sample=[]
        l_predxstart=[]

        for i in indices:

            # out = self.diffusion.ddim_sample(self.model, sample, t)           
            out = self.diffusion.ddim_sample(self.model, sample, torch.full((x.shape[0], ), i).long().to(x.device))


            sample = out["sample"]


            l_sample.append(out['sample'])
            l_predxstart.append(out['pred_xstart'])
        
        
        # visualize
        # si(torch.cat(l_sample), 'l_sample.png', to_01=1)
        # si(torch.cat(l_predxstart), 'l_pxstart.png', to_01=1)

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2
        
        return sample

class SDEDigitalAttack(BaseDigitalAttack):
    """https://arxiv.org/abs/1706.06083."""

    def __init__(self,
                 *args,
                 t=2,
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
        self.net = Denoised_Model(t)

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

        for _ in range(self.steps):
            data['inputs'].requires_grad = True
            data['inputs'] = self.net.sdedit(data['inputs'])
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
