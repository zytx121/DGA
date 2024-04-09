from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes


@TRANSFORMS.register_module()
class DPatchEditBbox(BaseTransform):
    """Edit the GT bbox for DPatch attack.

    Required Keys:

    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_bboxes_labels (np.int64)

    Modified Keys:

    - gt_bboxes
    - gt_bboxes_labels

    Args:
        patch_width (float): The width of patch. Defaults to 40.
    """

    def __init__(self, patch_width: float = 40.) -> None:
        self.patch_width = patch_width

    def transform(self, results: dict) -> dict:
        """Transform function to random shift images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Shift results.
        """
        boxes = results['gt_bboxes'].clone().tensor
        # print('boxes: ', boxes)
        boxes[..., 0] = boxes[..., 0] * 0
        boxes[..., 1] = boxes[..., 1] * 0
        boxes[..., 2] = boxes.new_tensor(self.patch_width)
        boxes[..., 3] = boxes.new_tensor(self.patch_width)
        results['gt_bboxes'] = HorizontalBoxes(boxes)

        # print(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(patch_width={self.patch_width})'
        return repr_str
