# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Union

import cv2
import numpy as np
import torch
from mmdet.models.detectors.rtmdet import RTMDet
from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from torch import Tensor

from .patch import (NPSCalculator, PatchApplier, PatchGenerator,
                    PatchTransformer, TotalVariation)


@MODELS.register_module()
class OBJRTMDet(RTMDet):

    def __init__(self,
                 *args,
                 patch_size,
                 printfile,
                 patch_dir=None,
                 adv_img_dir=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_genetator = PatchGenerator(patch_size, patch_dir=patch_dir)
        self.patch_applier = PatchApplier()
        self.patch_transformer = PatchTransformer()
        self.total_variation = TotalVariation()
        self.nps_calculator = NPSCalculator(printfile, patch_size)
        self.adv_img_dir = adv_img_dir
        if adv_img_dir is not None:
            if not os.path.exists(adv_img_dir):
                os.makedirs(adv_img_dir)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        gt_bboxes = [
            gt_instances.bboxes for gt_instances in batch_gt_instances
        ]
        img_shape = batch_img_metas[0]['batch_input_shape']
        patch = self.patch_genetator.patch
        adv_batch_t = self.patch_transformer(
            patch, gt_bboxes, img_shape[0], do_rotate=False, rand_loc=False)
        p_img_batch = self.patch_applier(batch_inputs, adv_batch_t)

        x = self.extract_feat(p_img_batch)

        # Add object loss
        cls_scores, bbox_preds = self.bbox_head.forward(x)
        obj_loss = torch.Tensor([]).cuda()
        for cls_scores_i in cls_scores:
            batch, cls_num, w, h = cls_scores_i.shape
            cls_scores_i = cls_scores_i.view(batch, cls_num, w * h)
            max_conf, max_conf_idx = torch.max(cls_scores_i, dim=2)
            max_conf = max_conf[0][0]
            max_conf = max_conf.view(1)
            obj_loss = torch.cat([obj_loss, max_conf], dim=0)
        obj_loss = obj_loss.max()

        # Add Patch loss
        tv = self.total_variation(patch)
        tv = torch.max(2.5 * tv, torch.tensor(0.1))
        nps = self.nps_calculator(patch)
        losses = {'tv_loss': tv, 'nps_loss': 0.01 * nps, 'obj_loss:': obj_loss}

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        gt_bboxes = [
            gt_instances.bboxes for gt_instances in batch_gt_instances
        ]

        img_name = batch_img_metas[0]['img_path'].split('/')[-1]
        mean = np.array([0., 0., 0.])
        std = np.array([255., 255., 255.])

        if len(gt_bboxes[0]) != 0:
            img_shape = batch_img_metas[0]['batch_input_shape']
            patch = self.patch_genetator()
            adv_batch_t = self.patch_transformer(
                patch,
                gt_bboxes,
                img_shape[0],
                do_rotate=False,
                rand_loc=False)
            p_img_batch = self.patch_applier(batch_inputs, adv_batch_t.cuda())
            img = p_img_batch[0, :, :, :].detach().cpu()
            img[0] = (img[0] * std[0] + mean[0])
            img[1] = (img[1] * std[1] + mean[1])
            img[2] = (img[2] * std[2] + mean[2])
            img = img.numpy().transpose(1, 2, 0)
            if self.adv_img_dir is not None:
                cv2.imwrite(f'{self.adv_img_dir}/{img_name}', img)
            x = self.extract_feat(p_img_batch)
        else:
            img = batch_inputs[0, :, :, :].detach().cpu()
            img[0] = (img[0] * std[0] + mean[0])
            img[1] = (img[1] * std[1] + mean[1])
            img[2] = (img[2] * std[2] + mean[2])
            img = img.numpy().transpose(1, 2, 0)
            if self.adv_img_dir is not None:
                cv2.imwrite(f'{self.adv_img_dir}/{img_name}', img)
            x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
