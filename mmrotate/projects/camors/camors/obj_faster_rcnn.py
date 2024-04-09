# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os

import cv2
import numpy as np
import torch
from mmdet.models.detectors.faster_rcnn import FasterRCNN
from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from torch import Tensor

from .patch import (NPSCalculator, PatchApplier, PatchGenerator,
                    PatchTransformer, TotalVariation)


@MODELS.register_module()
class OBJFasterRCNN(FasterRCNN):

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
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        self.backbone.eval()
        self.neck.eval()

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

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            # losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        # Add object loss
        cls_scores, bbox_preds = self.roi_head.forward(x, rpn_results_list)
        obj_loss = torch.Tensor([]).cuda()
        for cls_scores_i in cls_scores:
            obj_loss = torch.cat([obj_loss, cls_scores_i[0].view(1)], dim=0)
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
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        gt_bboxes = [
            gt_instances.bboxes for gt_instances in batch_gt_instances
        ]
        img_shape = batch_img_metas[0]['batch_input_shape']

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

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
