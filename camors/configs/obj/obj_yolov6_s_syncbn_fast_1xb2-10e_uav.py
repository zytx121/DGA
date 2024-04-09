_base_ = ['../_base_/default_runtime_mmyolo.py']
pwd_dir = '/home/zytx121/mmrotate/'
custom_imports = dict(
    imports=['projects.camors.camors'], allow_failed_imports=False)
exp_name = 'obj_uav_yolov6'

# ======================= Frequently modified parameters =====================
# -----data related-----
data_root = '/home/zytx121/mmrotate/data/uav_track/'
# Path of train annotation file
train_ann_file = 'train/annotations/result.json'
train_data_prefix = 'train/images/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'val/annotations/result.json'
val_data_prefix = 'val/images/'  # Prefix of val image path
# val_ann_file = 'train/annotations/result.json'
# val_data_prefix = 'train/images/'


num_classes = 1  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 2
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# -----train val related-----
# Base learning rate for optim_wrapper
base_lr = 0.03
max_epochs = 10  # Maximum training epochs

# ======================= Possible modified parameters =======================
# -----data related-----
img_scale = (1920, 1920)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
# batch_shapes_cfg = dict(
#     type='BatchShapePolicy',
#     batch_size=val_batch_size_per_gpu,
#     img_size=img_scale[0],
#     size_divisor=32,
#     extra_pad_ratio=0.5)

batch_shapes_cfg = None

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
lr_factor = 0.01  # Learning rate scaling factor
weight_decay = 0.0005
# Save model checkpoint and validation intervals
save_epoch_intervals = 1
# The maximum checkpoints to keep.
max_keep_ckpts = 3
# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ============================== Unmodified in most cases ===================
model = dict(
    type='mmdet.OBJYOLODetector',
    patch_size=50,
    printfile=pwd_dir + 'projects/camors/camors/patch/30values.txt',
    # patch_dir=pwd_dir + exp_name + '/6.npy',
    adv_img_dir=pwd_dir + exp_name + '/adv_img/',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv6EfficientRep',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True)),
    neck=dict(
        type='YOLOv6RepPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[128, 256, 512],
        num_csp_blocks=12,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    bbox_head=dict(
        type='YOLOv6Head',
        head_module=dict(
            type='YOLOv6HeadModule',
            num_classes=num_classes,
            in_channels=[128, 256, 512],
            widen_factor=widen_factor,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[8, 16, 32]),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='giou',
            bbox_format='xyxy',
            reduction='mean',
            loss_weight=2.5,
            return_iou=False)),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(
            type='BatchATSSAssigner',
            num_classes=num_classes,
            topk=9,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            topk=13,
            alpha=1,
            beta=6),
    ),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))

# The training pipeline of YOLOv6 is basically the same as YOLOv5.
# The difference is that Mosaic and RandomAffine will be closed in the last 15 epochs. # noqa
pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True)
]

train_pipeline = [
    *pre_transform,
    # dict(
    #     type='Mosaic',
    #     img_scale=img_scale,
    #     pad_val=114.0,
    #     pre_transform=pre_transform),
    # dict(
    #     type='YOLOv5RandomAffine',
    #     max_rotate_degree=0.0,
    #     max_translate_ratio=0.1,
    #     scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
    #     # img_scale is (width, height)
    #     border=(-img_scale[0] // 2, -img_scale[1] // 2),
    #     border_val=(114, 114, 114),
    #     max_shear_degree=0.0),
    # dict(type='YOLOv5HSVRandomAug'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_shear_degree=0.0,
    ),
    # dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

metainfo = dict(classes=('uav', ))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    collate_fn=dict(type='yolov5_collate'),
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader

# Optimizer and learning rate scheduler of YOLOv6 are basically the same as YOLOv5. # noqa
# The difference is that the scheduler_type of YOLOv6 is cosine.
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='SGD',
#         lr=base_lr,
#         momentum=0.937,
#         weight_decay=weight_decay,
#         nesterov=True,
#         batch_size_per_gpu=train_batch_size_per_gpu),
#     constructor='YOLOv5OptimizerConstructor')
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.),
            'neck': dict(lr_mult=0.),
            'head': dict(lr_mult=0.)
        },
        bypass_duplicate=True),
    optimizer=dict(type='Adam', lr=base_lr))

default_hooks = dict(
    # param_scheduler=dict(
    #     type='YOLOv5ParamSchedulerHook',
    #     scheduler_type='cosine',
    #     lr_factor=lr_factor,
    #     max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=max_keep_ckpts,
        save_best='auto'))

# custom_hooks = [
#     dict(
#         type='EMAHook',
#         ema_type='ExpMomentumEMA',
#         momentum=0.0001,
#         update_buffers=True,
#         strict_load=False,
#         priority=49),
#     dict(
#         type='mmdet.PipelineSwitchHook',
#         switch_epoch=max_epochs - num_last_epochs,
#         switch_pipeline=train_pipeline_stage2)
# ]
custom_hooks = [
    dict(type='mmdet.SavePatchHook', patch_dir=pwd_dir + '/' + exp_name)
]

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

load_from = pwd_dir + 'work_dirs/yolov6_s_syncbn_fast_1xb2-100e_cowc/epoch_100.pth'  # noqa