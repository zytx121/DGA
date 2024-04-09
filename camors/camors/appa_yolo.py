# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Union

import cv2
import numpy as np
import torch
from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmyolo.models.detectors import YOLODetector
from torch import Tensor

from .patch import (NPSCalculator, PatchApplier, PatchGenerator,
                    PatchTransformer, TotalVariation)


@MODELS.register_module()
class APPAYOLODetector(YOLODetector):

    def __init__(self,
                 *args,
                 patch_size,
                 printfile,
                 patch_dir=None,
                 adv_img_dir=None,
                 outside=False,
                 offset=16,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_genetator = PatchGenerator(patch_size, patch_dir=patch_dir)
        self.patch_applier = PatchApplier()
        self.patch_transformer = PatchTransformer(
            offset=offset)
        self.total_variation = TotalVariation()
        self.nps_calculator = NPSCalculator(printfile, patch_size)
        self.adv_img_dir = adv_img_dir
        if adv_img_dir is not None:
            if not os.path.exists(adv_img_dir):
                os.makedirs(adv_img_dir)

        # def set_bn_eval(m):
        #     classname = m.__class__.__name__
        #     if classname.find('BatchNorm2d') != -1:
        #       m.eval()

        # self.backbone.apply(set_bn_eval)
        # self.neck.apply(set_bn_eval)

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
        self.backbone.eval()
        self.neck.eval()
        self.bbox_head.eval()

        if isinstance(batch_data_samples, list):
            outputs = unpack_gt_instances(batch_data_samples)
            (batch_gt_instances, batch_gt_instances_ignore,
             batch_img_metas) = outputs
            gt_bboxes = [
                gt_instances.bboxes for gt_instances in batch_gt_instances
            ]
        else:
            gt_bboxes = [batch_data_samples['bboxes_labels'][..., 2:]]
            batch_img_metas = batch_data_samples['img_metas']

        # if no gt_bboxes, return 0 loss
        if gt_bboxes[0].shape[0] == 0:
            zero_tensor = torch.Tensor([0.]).cuda().requires_grad_(True)
            return {
                'tv_loss': zero_tensor,
                'nps_loss': zero_tensor,
                'obj_loss:': zero_tensor
            }

        img_shape = batch_img_metas[0]['batch_input_shape']
        patch = self.patch_genetator.patch
        adv_batch_t = self.patch_transformer(
            patch, gt_bboxes, img_shape[0], do_rotate=False, rand_loc=False)
        p_img_batch = self.patch_applier(batch_inputs, adv_batch_t)

        x = self.extract_feat(p_img_batch)
        #  # Add dispear loss
        # results_list = self.bbox_head.predict(
        #     x, batch_data_samples, rescale=True)
        # det_loss = torch.tensor(0.0).cuda()
        # for results in results_list:
        #     det_loss += results.scores.max(
        #     ) if len(results) > 0 else torch.tensor(0.0).cuda()

        results = self.bbox_head.forward(x)

        cls_scores = results[0]
        det_loss = torch.Tensor([]).cuda()
        for cls_scores_i in cls_scores:
            batch, num_base_priors, w, h = cls_scores_i.shape
            cls_scores_i = cls_scores_i.reshape(batch, -1)
            max_conf, max_conf_idx = torch.max(cls_scores_i, dim=1)
            det_loss = torch.cat([det_loss, max_conf], dim=0)
        det_loss = det_loss.max()

        # Add Patch loss
        tv = self.total_variation(patch)
        tv = torch.max(2.5 * tv, torch.tensor(0.1))
        nps = self.nps_calculator(patch)
        losses = {
            'tv_loss': tv,
            'nps_loss': 0.01 * nps,
            'det_loss': det_loss
        }

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
                rand_loc=False,
                test_mode=True)
            p_img_batch = self.patch_applier(batch_inputs, adv_batch_t.cuda())
            img = p_img_batch[0, :, :, :].detach().cpu().clone()
            img[0] = (img[0] * std[0] + mean[0])
            img[1] = (img[1] * std[1] + mean[1])
            img[2] = (img[2] * std[2] + mean[2])
            img = img[:, 16:img.shape[-1] - 16, 16:img.shape[-1] - 16]
            img = img.numpy().transpose(1, 2, 0)
            if self.adv_img_dir is not None:
                cv2.imwrite(f'{self.adv_img_dir}/{img_name}', img)
            x = self.extract_feat(p_img_batch)
        else:
            img = batch_inputs[0, :, :, :].detach().cpu().clone()
            img[0] = (img[0] * std[0] + mean[0])
            img[1] = (img[1] * std[1] + mean[1])
            img[2] = (img[2] * std[2] + mean[2])
            img = img[:, 16:img.shape[-1] - 17, 16:img.shape[-1] - 17]
            img = img.numpy().transpose(1, 2, 0)
            if self.adv_img_dir is not None:
                cv2.imwrite(f'{self.adv_img_dir}/{img_name}', img)
            x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
