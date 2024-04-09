import os.path as osp
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.apis import DetInferencer
from mmdet.registry import DATASETS, METRICS
from mmdet.structures import DetDataSample
from rich.progress import track

InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = List[DetDataSample]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]


class BaseDigitalAttack(DetInferencer):

    def __call__(self,
                 inputs: InputsType,
                 batch_size: int = 1,
                 return_vis: bool = False,
                 show: bool = False,
                 wait_time: int = 0,
                 no_save_vis: bool = False,
                 draw_pred: bool = True,
                 pred_score_thr: float = 0.3,
                 return_datasample: bool = False,
                 print_result: bool = False,
                 no_save_pred: bool = True,
                 out_dir: str = '',
                 **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            batch_size (int): Inference batch size. Defaults to 1.
            show (bool): Whether to display the visualization results in a
                popup window. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            no_save_vis (bool): Whether to force not to save prediction
                vis results. Defaults to False.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            return_datasample (bool): Whether to return results as
                :obj:`DetDataSample`. Defaults to False.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            no_save_pred (bool): Whether to force not to save prediction
                results. Defaults to True.
            out_file: Dir to save the inference results or
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

            **kwargs: Other keyword arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        # load data from test_dataloader instead of inputs.
        dataset = DATASETS.build(self.cfg.test_dataloader.dataset)
        metric = METRICS.build(self.cfg.test_evaluator)
        metric.dataset_meta = {'classes': dataset._metainfo['classes']}

        # metric.dataset_meta['classes'] = dataset._metainfo['classes']
        results_dict = {
            'predictions': [],
            'visualization': [],
            'attack_rate': [],
            'psnr': [],
            'ssim': []
        }
        eval_results = []
        img_id = metric._coco_api.getImgIds()[0]
        for data in track(dataset, description='Inference'):
            ori_inputs = (data['data_samples'].img_path, )
            data['inputs'] = [data['inputs']]
            data['data_samples'] = [data['data_samples']]
            data = self.model.data_preprocessor(data, False)

            # attack
            data, info = self.attack(data)
            if info != {}:
                results_dict['attack_rate'].append(info['attack_rate'])
                results_dict['psnr'].append(info['psnr'])
                results_dict['ssim'].append(info['ssim'])

            # visualize
            preds = self.model._run_forward(data, mode='predict')
            visualization = self.visualize(
                ori_inputs,
                data,
                preds,
                return_vis=return_vis,
                show=show,
                wait_time=wait_time,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                no_save_vis=no_save_vis,
                img_out_dir=out_dir,
                **visualize_kwargs)
            results = self.postprocess(
                preds,
                visualization,
                return_datasample=return_datasample,
                print_result=print_result,
                no_save_pred=no_save_pred,
                pred_out_dir=out_dir,
                **postprocess_kwargs)
            results_dict['predictions'].extend(results['predictions'])
            if results['visualization'] is not None:
                results_dict['visualization'].extend(results['visualization'])

            # eval
            eval_results.append(
                self.eval_process(data['data_samples'][0], img_id))
            img_id += 1
            # if img_id == 11466:
            #     break

        metric.compute_metrics(eval_results)
        print('attack_rate: ', np.mean(results_dict['attack_rate']))
        print('psnr: ', np.mean(results_dict['psnr']))
        print('ssim: ', np.mean(results_dict['ssim']))
        return results_dict

    def visualize(self,
                  inputs: InputsType,
                  data,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  no_save_vis: bool = False,
                  img_out_dir: str = '',
                  **kwargs) -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
            preds (List[:obj:`DetDataSample`]): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            no_save_vis (bool): Whether to force not to save prediction
                vis results. Defaults to False.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if no_save_vis is True:
            img_out_dir = ''

        if not show and img_out_dir == '' and not return_vis:
            return None

        if self.visualizer is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):

            pred.pred_instances.bboxes = pred.pred_instances.bboxes.detach()
            img = data['inputs'].squeeze(0).cpu().detach().numpy()
            mean = np.array([0., 0., 0.])
            std = np.array([255., 255., 255.])
            img[0] = img[0] * std[0] + mean[0]  # B(0)
            img[1] = img[1] * std[1] + mean[1]  # G(1)
            img[2] = img[2] * std[2] + mean[2]  # R(2)
            img = img.transpose(1, 2, 0)
            img_name = osp.basename(single_input)

            out_file = osp.join(img_out_dir, 'vis',
                                img_name) if img_out_dir != '' else None

            self.visualizer.add_datasample(
                img_name,
                img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                out_file=out_file,
            )
            results.append(self.visualizer.get_image())
            self.num_visualized_imgs += 1

        return results

    def mask_generator(self, preds):
        img_shape = preds[0].img_shape
        attack_map = torch.zeros(img_shape)
        boxes = []
        for box, score in zip(preds[0].pred_instances.bboxes,
                              preds[0].pred_instances.scores):
            if score > 0.05:
                boxes.append([[int(box[1]), int(box[0])],
                              [int(box[3]), int(box[2])]])
        for box in boxes:
            y1 = max(box[0][0], 0)
            x1 = max(box[0][1], 0)
            y2 = min(box[1][0], img_shape[0] - 1)
            x2 = min(box[1][1], img_shape[1] - 1)
            attack_map[y1:y2, x1:x2] = 1
        mask = torch.stack((attack_map, attack_map, attack_map), axis=0)
        return mask

    def attack(self, data, *args, **kwargs):
        r"""
        It defines the attack at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def attack_loss(self, preds):
        results, cls_scores_list, bbox_preds_list = preds

        # support one stage and two stage
        try:
            self.model.roi_head
            cls_scores = cls_scores_list
        except:
            mlvl_scores = []
            for cls_score_lvl in cls_scores_list:
                mlvl_scores.append(torch.flatten(cls_score_lvl))
            cls_scores = torch.cat(mlvl_scores)
            results = results[0]
        det_scores = results.scores
        # det_bboxes = results.bboxes
        # det_labels = results.labels

        # attack loss
        if len(det_scores) == 0:
            scores = cls_scores[cls_scores >= 0.05]
        else:
            scores = det_scores[det_scores >= 0.05]
        loss = 1 - F.mse_loss(scores,
                              torch.zeros(scores.shape).to(scores.device))
        return loss

    def eval_process(self, data_sample, img_id):
        result = dict()
        pred = data_sample.pred_instances
        result['img_id'] = img_id
        result['bboxes'] = pred['bboxes'].cpu().detach().numpy()
        result['scores'] = pred['scores'].cpu().detach().numpy()
        result['labels'] = pred['labels'].cpu().detach().numpy()

        # parse gt
        gt = dict()
        gt['width'] = data_sample.ori_shape[1]
        gt['height'] = data_sample.ori_shape[0]
        gt['img_id'] = img_id
        return gt, result
