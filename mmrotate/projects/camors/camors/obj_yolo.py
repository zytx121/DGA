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
class OBJYOLODetector(YOLODetector):

    def __init__(self,
                 *args,
                 patch_size,
                 printfile,
                 patch_dir=None,
                 adv_img_dir=None,
                 use_rbb=False,
                 offset=0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_genetator = PatchGenerator(patch_size, patch_dir=patch_dir)
        self.patch_applier = PatchApplier()
        self.patch_transformer = PatchTransformer()
        self.total_variation = TotalVariation()
        self.nps_calculator = NPSCalculator(printfile, patch_size)
        self.adv_img_dir = adv_img_dir
        self.use_rbb = use_rbb
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
        self.backbone.eval()
        self.neck.eval()
        self.bbox_head.eval()
        if isinstance(batch_data_samples, list):
            outputs = unpack_gt_instances(batch_data_samples)
            (batch_gt_instances, batch_gt_instances_ignore,
             batch_img_metas) = outputs
            if self.use_rbb:
                gt_rbboxes = []
                gt_masks = [
                    gt_instances.masks for gt_instances in batch_gt_instances
                ]
                masks = gt_masks[0]
                for idx, poly_per_obj in enumerate(masks.masks):
                    pts_per_obj = []
                    for p in poly_per_obj:
                        pts_per_obj.append(
                            np.array(p, dtype=np.float32).reshape(-1, 2))
                    pts_per_obj = np.concatenate(pts_per_obj, axis=0)
                    x1, y1 = pts_per_obj[0]
                    x2, y2 = pts_per_obj[1]
                    t= np.arctan2(y2 - y1, x2 - x1)
                    (x, y), (w, h), angle = cv2.minAreaRect(pts_per_obj)
                    t = t % (2 * np.pi)
                    t = t - 2 * np.pi if t >= np.pi else t
                    gt_rbboxes.append([x, y, w, h, t])
                gt_rbboxes = [torch.cuda.FloatTensor(gt_rbboxes)]
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
                'obj_loss': zero_tensor
            }
        img_shape = batch_img_metas[0]['batch_input_shape']
        ori_shape = batch_img_metas[0]['ori_shape']
        patch = self.patch_genetator.patch
        if self.use_rbb:
            adv_batch_t = self.patch_transformer(
                patch, gt_rbboxes, img_shape, ori_shape, do_rotate=False, rand_loc=False, use_rbb=self.use_rbb)
        else:
            adv_batch_t = self.patch_transformer(
                patch, gt_bboxes, img_shape, ori_shape, do_rotate=True, rand_loc=True, use_rbb=False)
        p_img_batch = self.patch_applier(batch_inputs, adv_batch_t)

        x = self.extract_feat(p_img_batch)
        results = self.bbox_head.forward(x)

        # APPA loss
        # cls_scores = results[0]
        # det_loss = torch.Tensor([]).cuda()
        # for cls_scores_i in cls_scores:
        #     batch, num_base_priors, w, h = cls_scores_i.shape
        #     cls_scores_i = cls_scores_i.reshape(batch, -1)
        #     max_conf, max_conf_idx = torch.max(cls_scores_i, dim=1)
        #     det_loss = torch.cat([det_loss, max_conf], dim=0)
        # det_loss = det_loss.max()
     
        # Add OBJ loss
        obj_flag = True if len(results) == 3 and results[0][
            0].shape == results[-1][0].shape else False
        if obj_flag:
            # yolov7 yolov5
            cls_scores, bbox_preds, objectness = results
            obj_loss = torch.Tensor([]).cuda()
            for cls_scores_i, objectness_i in zip(cls_scores, objectness):
                batch, num_base_priors, w, h = cls_scores_i.shape
                cls_scores_i = cls_scores_i.reshape(batch, -1)
                objectness_i = torch.sigmoid(objectness_i.reshape(batch, -1))
                confs_if_object = cls_scores_i
                max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)
                obj_loss = torch.cat([obj_loss, max_conf], dim=0)
            obj_loss = obj_loss.max()
        else:
            # yolov8
            cls_scores = results[0]
            obj_loss = torch.Tensor([]).cuda()
            for cls_scores_i in cls_scores:
                batch, num_base_priors, w, h = cls_scores_i.shape
                cls_scores_i = cls_scores_i.reshape(batch, -1)
                max_conf, max_conf_idx = torch.max(cls_scores_i, dim=1)
                obj_loss = torch.cat([obj_loss, max_conf], dim=0)
            obj_loss = obj_loss.max()

        # Add Patch loss
        tv = self.total_variation(patch)
        tv = torch.max(2.5 * tv, torch.tensor(0.1))
        nps = self.nps_calculator(patch)
        losses = {'tv_loss': tv, 'nps_loss': 0.01 * nps, 'obj_loss': obj_loss}

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

        if self.use_rbb:
            gt_rbboxes = []
            gt_masks = [
                gt_instances.masks for gt_instances in batch_gt_instances
            ]
            masks = gt_masks[0]
            for idx, poly_per_obj in enumerate(masks.masks):
                pts_per_obj = []
                for p in poly_per_obj:
                    pts_per_obj.append(
                        np.array(p, dtype=np.float32).reshape(-1, 2))
                pts_per_obj = np.concatenate(pts_per_obj, axis=0)
                x1, y1 = pts_per_obj[0]
                x2, y2 = pts_per_obj[1]
                t= np.arctan2(y2 - y1, x2 - x1)
                (x, y), (w, h), angle = cv2.minAreaRect(pts_per_obj)
                t = t % (2 * np.pi)
                t = t - 2 * np.pi if t >= np.pi else t
                gt_rbboxes.append([x, y, w, h, t])
            gt_rbboxes = [torch.cuda.FloatTensor(gt_rbboxes)]

        gt_bboxes = [
            gt_instances.bboxes for gt_instances in batch_gt_instances
        ]

        img_name = batch_img_metas[0]['img_path'].split('/')[-1]
        mean = np.array([0., 0., 0.])
        std = np.array([255., 255., 255.])
        img_shape = batch_img_metas[0]['batch_input_shape']
        ori_shape = batch_img_metas[0]['ori_shape']
        pad_pixel = int((img_shape[0] - ori_shape[0] / (ori_shape[1] / img_shape[0])) / 2)

        if len(gt_bboxes[0]) != 0:
            patch = self.patch_genetator()
            if self.use_rbb:
                adv_batch_t = self.patch_transformer(
                    patch, gt_rbboxes, img_shape, ori_shape, do_rotate=False, rand_loc=False, use_rbb=self.use_rbb)
            else:
                adv_batch_t = self.patch_transformer(
                    patch, gt_bboxes, img_shape, ori_shape, do_rotate=False, rand_loc=False, use_rbb=self.use_rbb)
            
            p_img_batch = self.patch_applier(batch_inputs, adv_batch_t.cuda())
            img = p_img_batch[0, :, :, :].detach().cpu().clone()
            img[0] = (img[0] * std[0] + mean[0])
            img[1] = (img[1] * std[1] + mean[1])
            img[2] = (img[2] * std[2] + mean[2])    
            img = img[:, pad_pixel:img.shape[-1] - pad_pixel, :] 
            img = img.numpy().transpose(1, 2, 0)
            img = img[:, :, ::-1]
            if self.adv_img_dir is not None:
                cv2.imwrite(f'{self.adv_img_dir}/{img_name}', img)
            x = self.extract_feat(p_img_batch)
        else:
            img = batch_inputs[0, :, :, :].detach().cpu().clone()
            img[0] = (img[0] * std[0] + mean[0])
            img[1] = (img[1] * std[1] + mean[1])
            img[2] = (img[2] * std[2] + mean[2])
            img = img[:, pad_pixel:img.shape[-1] - pad_pixel, :] 
            img = img.numpy().transpose(1, 2, 0)
            img = img[:, :, ::-1]
            if self.adv_img_dir is not None:
                cv2.imwrite(f'{self.adv_img_dir}/{img_name}', img)
            x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

@MODELS.register_module()
class APPAYOLODetector(OBJYOLODetector):

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
            if self.use_rbb:
                gt_rbboxes = []
                gt_masks = [
                    gt_instances.masks for gt_instances in batch_gt_instances
                ]
                masks = gt_masks[0]
                for idx, poly_per_obj in enumerate(masks.masks):
                    pts_per_obj = []
                    for p in poly_per_obj:
                        pts_per_obj.append(
                            np.array(p, dtype=np.float32).reshape(-1, 2))
                    pts_per_obj = np.concatenate(pts_per_obj, axis=0)
                    x1, y1 = pts_per_obj[0]
                    x2, y2 = pts_per_obj[1]
                    t= np.arctan2(y2 - y1, x2 - x1)
                    (x, y), (w, h), angle = cv2.minAreaRect(pts_per_obj)
                    t = t % (2 * np.pi)
                    t = t - 2 * np.pi if t >= np.pi else t
                    gt_rbboxes.append([x, y, w, h, t])
                gt_rbboxes = [torch.cuda.FloatTensor(gt_rbboxes)]
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
                'obj_loss': zero_tensor
            }
        img_shape = batch_img_metas[0]['batch_input_shape']
        ori_shape = batch_img_metas[0]['ori_shape']
        patch = self.patch_genetator.patch
        if self.use_rbb:
            adv_batch_t = self.patch_transformer(
                patch, gt_rbboxes, img_shape, ori_shape, do_rotate=False, rand_loc=False, use_rbb=self.use_rbb)
        else:
            adv_batch_t = self.patch_transformer(
                patch, gt_bboxes, img_shape, ori_shape, do_rotate=True, rand_loc=True, use_rbb=False)
        p_img_batch = self.patch_applier(batch_inputs, adv_batch_t)

        x = self.extract_feat(p_img_batch)
        results = self.bbox_head.forward(x)

        # APPA loss
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
        losses = {'tv_loss': tv, 'nps_loss': 0.01 * nps, 'appa_loss': det_loss}

        return losses