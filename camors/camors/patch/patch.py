import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class PatchGenerator(nn.Module):
    """PatchGenerator: generates a patch"""

    def __init__(self, patch_size, patch_dir=None):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dir = patch_dir

        if patch_dir is None:
            self.patch = nn.Parameter(
                torch.rand(3, patch_size, patch_size), requires_grad=True)
            # self.patch = nn.Parameter(
            #     torch.full((3, patch_size, patch_size), 0.5),
            #     requires_grad=True)
        else:
            patch_np = np.load(patch_dir)
            # print(patch_np.shape)
            # patch_np_1 = patch_np[::-1, :, :].copy()
            self.patch = torch.from_numpy(patch_np).float().cuda()
            
            self.patch = nn.Parameter(
                torch.from_numpy(patch_np), requires_grad=True)

    def forward(self):
        return self.patch


class MedianPool2d(nn.Module):
    """Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0],
                     self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1, )).median(dim=-1)[0]
        return x


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of
    patches, randomly adjusting brightness and contrast, adding random
    amount of noise, and rotating randomly. Resizes patches according
    to as size based on the batch of labels, and pads them to the
    dimension of an image.
    """

    def __init__(self):
        super().__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)

    def forward(self,
                adv_patch,
                lab_batch,
                img_shape,
                ori_shape,
                do_rotate=True,
                rand_loc=False,
                use_rbb=False):
        # Padding the labels to (B, max_label_num, 4) with (1,1,1,1)
        max_label_num = max(lab.shape[0] for lab in lab_batch)
        pad_lab_batch = []
        for lab in lab_batch:
            if use_rbb:
                new_lab = lab[:, :4].clone()
                new_lab[:, 1] = lab[:, 1]  + (ori_shape[1] - ori_shape[0]) / 2  # (4000 - 2250) / 2 = 875
            else:
                new_lab = lab.clone()
                new_lab[:, 0] = lab[:, 0] + (lab[:, 2] - lab[:, 0]) / 2
                new_lab[:, 1] = lab[:, 1] + (lab[:, 3] - lab[:, 1]) / 2 + (ori_shape[1] - ori_shape[0]) / 2  # (4000 - 2250) / 2 = 875
                new_lab[:, 2] = lab[:, 2] - lab[:, 0]
                new_lab[:, 3] = lab[:, 3] - lab[:, 1]

            new_lab = new_lab / ori_shape[1]
            pad_size = max_label_num - lab.shape[0]
            if (pad_size > 0):
                new_lab = F.pad(new_lab, (0, 0, 0, pad_size), value=1)
            pad_lab_batch.append(new_lab)
        lab_batch = torch.stack(pad_lab_batch, dim=0)
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # Determine size of padding
        pad = (img_shape[0] - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)
        adv_batch = adv_patch.expand(
            lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(
            self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3),
                                   adv_batch.size(-2), adv_batch.size(-1))

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(
            self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3),
                                       adv_batch.size(-2), adv_batch.size(-1))

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(
            -1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)
        msk_batch = torch.cuda.FloatTensor(adv_batch.size()).fill_(1)

        # Pad patch and mask to image dimensions
        mypad = nn.ConstantPad2d(
            (int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        # mypad = nn.ConstantPad2d((int(pad), int(pad), int(pad), int(pad)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)

        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))

        if use_rbb:
            angle = lab[:, 4]
        else:
            if do_rotate:
                angle = torch.cuda.FloatTensor(anglesize).uniform_(
                    self.minangle, self.maxangle)
            else:
                angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = torch.FloatTensor(lab_batch.size()).fill_(0)
        lab_batch_scaled[:, :, 0] = lab_batch[:, :, 0] * img_shape[0]
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_shape[0]
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_shape[0]
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_shape[0]

        target_size = torch.sqrt(((lab_batch_scaled[:, :, 2].mul(0.2))**2) + # 调节尺寸
                                 ((lab_batch_scaled[:, :, 3].mul(0.2))**2))

        target_x = lab_batch[:, :, 0].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 1].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 3].view(np.prod(batch_size))
        if (rand_loc):
            off_x = targetoff_x * (
                torch.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4)).cuda()
            target_x = target_x + off_x
            off_y = targetoff_y * (
                torch.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4)).cuda()
            target_y = target_y + off_y

        # target_size /= 2.0
        scale = target_size / current_patch_size

        scale = scale.view(anglesize)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        tx = 1 - 2 * target_x
        ty = 1 - 2 * target_y

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation,rescale matrix
        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = sin
        theta[:, 0, 2] = tx * cos + ty * sin
        theta[:, 1, 0] = -sin
        theta[:, 1, 1] = cos
        theta[:, 1, 2] = -tx * sin + ty * cos
        theta /= torch.cuda.FloatTensor(anglesize, 2,
                                        3).fill_(scale[0].float())

        grid = F.affine_grid(theta, adv_batch.shape)

        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)

        return adv_batch_t * msk_batch_t


class DPatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches. """

    def __init__(self):
        super(DPatchTransformer, self).__init__()

    def forward(self, adv_patch, img_size):
        # Determine size of padding
        pad = (img_size - adv_patch.size(-1))
        msk_batch = torch.cuda.FloatTensor(adv_patch.size()).fill_(1)

        # Pad patch and mask to image dimensions
        mypad = nn.ConstantPad2d((0, int(pad), 0, int(pad)), 0)
        adv_patch = mypad(adv_patch)
        msk_batch = mypad(msk_batch)
        adv_patch = torch.clamp(adv_patch, 0.000001, 0.999999)
        return adv_patch * msk_batch


class DPatchApplier(nn.Module):
    """DPatchApplier: applies adversarial patches to images.
"""

    def __init__(self):
        super(DPatchApplier, self).__init__()

    def forward(self, img_batch, patch):

        patch = patch.unsqueeze(0)
        patch = patch.expand(img_batch.size(0), -1, -1, -1)
        img_batch = torch.where((patch == 0), img_batch, patch)

        return img_batch


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a
    patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the
    total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(
            torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001),
            0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(
            torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001),
            0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the
    non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(
            self.get_printability_array(printability_file, patch_side),
            requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidean distance between colors in patch and
        # colors in printability_array
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array + 0.000001)
        color_dist = color_dist**2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(','))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa
