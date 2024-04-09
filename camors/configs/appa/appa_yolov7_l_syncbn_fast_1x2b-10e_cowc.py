_base_ = '../_base_/default_runtime_mmyolo.py'
pwd_dir = '/home/zytx121/mmrotate/'
custom_imports = dict(
    imports=['projects.camors.camors'], allow_failed_imports=False)

# ========================Frequently modified parameters======================
# -----data related-----
data_root = '/home/zytx121/mmrotate/data/COWC/'
# Path of train annotation file
# train_ann_file = 'annotations/train.json'
train_ann_file = 'annotations/test.json'
train_data_prefix = 'images/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/test.json'
# val_ann_file = 'annotations/test_yolov5_obj.json'
val_data_prefix = 'images/'  # Prefix of val image path
# val_ann_file = 'Toronto/annotations/test.json'
# val_data_prefix = 'Toronto/images/'  # Prefix of val image path

num_classes = 1  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 2
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# -----model related-----
# Basic size of multi-scale prior box
anchors = [
    [(12, 16), (19, 36), (40, 28)],  # P3/8
    [(36, 75), (76, 55), (72, 146)],  # P4/16
    [(142, 110), (192, 243), (459, 401)]  # P5/32
]
# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs
base_lr = 0.03
max_epochs = 1  # Maximum training epochs

num_epoch_stage2 = 0  # The last 30 epochs switch evaluation interval
val_interval_stage2 = 1  # Evaluation intervalb

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS.
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.65),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (256, 256)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32,
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5)

# -----model related-----
strides = [8, 16, 32]  # Strides of multi-scale prior box
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)

# Data augmentation
max_translate_ratio = 0.2  # YOLOv5RandomAffine
scaling_ratio_range = (0.5, 2.0)  # YOLOv5RandomAffine
mixup_prob = 0.15  # YOLOv5MixUp
randchoice_mosaic_prob = [0.8, 0.2]
mixup_alpha = 8.0  # YOLOv5MixUp
mixup_beta = 8.0  # YOLOv5MixUp

# -----train val related-----
loss_cls_weight = 0.3
loss_bbox_weight = 0.05
loss_obj_weight = 0.7
# BatchYOLOv7Assigner params
simota_candidate_topk = 10
simota_iou_weight = 3.0
simota_cls_weight = 1.0
prior_match_thr = 4.  # Priori box matching threshold
obj_level_weights = [4., 1.,
                     0.4]  # The obj loss weights of the three output layers

lr_factor = 0.1  # Learning rate scaling factor
weight_decay = 0.0005
save_epoch_intervals = 10  # Save model checkpoint and validation intervals
max_keep_ckpts = 3  # The maximum checkpoints to keep.

# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
model = dict(
    type='mmdet.APPAYOLODetector',
    patch_size=50,
    printfile=pwd_dir + 'projects/camors/camors/patch/30values.txt',
    # patch_dir=pwd_dir + 'appa_cowc_yolov5_dr/10.npy',
    adv_img_dir=pwd_dir + 'appa_cowc_yolov7/adv_img/',
    # data_preprocessor=dict(
    #     type='YOLOv5DetDataPreprocessor',
    #     mean=[0., 0., 0.],
    #     std=[255., 255., 255.],
    #     bgr_to_rgb=True),
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv7Backbone',
        arch='L',
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv7PAFPN',
        block_cfg=dict(
            type='ELANBlock',
            middle_ratio=0.5,
            block_ratio=0.25,
            num_blocks=4,
            num_convs_in_block=1),
        upsample_feats_cat_first=False,
        in_channels=[512, 1024, 1024],
        # The real output channel will be multiplied by 2
        out_channels=[128, 256, 512],
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv7HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            featmap_strides=strides,
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=strides),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_cls_weight *
            (num_classes / 80 * 3 / num_det_layers)),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            reduction='mean',
            loss_weight=loss_bbox_weight * (3 / num_det_layers),
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_obj_weight *
            ((img_scale[0] / 256)**2 * 3 / num_det_layers)),
        prior_match_thr=prior_match_thr,
        obj_level_weights=obj_level_weights,
        # BatchYOLOv7Assigner params
        simota_candidate_topk=simota_candidate_topk,
        simota_iou_weight=simota_iou_weight,
        simota_cls_weight=simota_cls_weight),
    test_cfg=model_test_cfg)

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True)
]

# mosiac4_pipeline = [
#    dict(
#        type='Mosaic',
#        img_scale=img_scale,
#        pad_val=114.0,
#        pre_transform=pre_transform),
#    dict(
#        type='YOLOv5RandomAffine',
#        max_rotate_degree=0.0,
#        max_shear_degree=0.0,
#        max_translate_ratio=max_translate_ratio,  # note
#        scaling_ratio_range=scaling_ratio_range,  # note
#        # img_scale is (width, height)
#        border=(-img_scale[0] // 2, -img_scale[1] // 2),
#        border_val=(114, 114, 114)),
# ]
mosiac4_pipeline = None

# mosiac9_pipeline = [
#    dict(
#        type='Mosaic9',
#        img_scale=img_scale,
#        pad_val=114.0,
#        pre_transform=pre_transform),
#    dict(
#        type='YOLOv5RandomAffine',
#        max_rotate_degree=0.0,
#        max_shear_degree=0.0,
#        max_translate_ratio=max_translate_ratio,  # note
#       scaling_ratio_range=scaling_ratio_range,  # note
#      # img_scale is (width, height)
#        border=(-img_scale[0] // 2, -img_scale[1] // 2),
#        border_val=(114, 114, 114)),
# ]
mosiac9_pipeline = None

# randchoice_mosaic_pipeline = dict(
#    type='RandomChoice',
#    transforms=[mosiac4_pipeline, mosiac9_pipeline],
#    prob=randchoice_mosaic_prob)
randchoice_mosaic_pipeline = None

train_pipeline = [
    *pre_transform,
    # randchoice_mosaic_pipeline,
    # dict(
    #    type='YOLOv5MixUp',
    #    alpha=mixup_alpha,  # note
    #    beta=mixup_beta,  # note
    #    prob=mixup_prob,
    #    pre_transform=[*pre_transform, randchoice_mosaic_pipeline]),
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
                   'flip_direction', 'scale_factor'))
]

metainfo = dict(classes=('car', ))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # collate_fn=dict(type='yolov5_collate'),  # FASTER
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

param_scheduler = None
# optim_wrapper = dict(
#    type='OptimWrapper',
#    optimizer=dict(
#        type='SGD',
#        lr=base_lr,
#        momentum=0.937,
#        weight_decay=weight_decay,
#        nesterov=True,
#        batch_size_per_gpu=train_batch_size_per_gpu),
#    constructor='YOLOv7OptimWrapperConstructor')

optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.),
            'neck': dict(lr_mult=0.),
            'bbox_head': dict(lr_mult=0.)
        },
        bypass_duplicate=True),
    optimizer=dict(type='Adam', lr=base_lr))

default_hooks = dict(
    # param_scheduler=dict(
    #    type='YOLOv5ParamSchedulerHook',
    #    scheduler_type='cosine',
    #    lr_factor=lr_factor,  # note
    #    max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        save_param_scheduler=False,
        interval=save_epoch_intervals,
        save_best='auto',
        max_keep_ckpts=max_keep_ckpts))

# custom_hooks = [
#    dict(
#        type='EMAHook',
#        ema_type='ExpMomentumEMA',
#        momentum=0.0001,
#        update_buffers=True,
#        strict_load=False,
#        priority=49)
# ]

custom_hooks = [
    dict(type='mmdet.SavePatchHook', patch_dir=pwd_dir + 'appa_cowc_yolov7/')
]

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),  # Can be accelerated
    ann_file=data_root + val_ann_file,
    metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[(max_epochs - num_epoch_stage2, val_interval_stage2)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

load_from = pwd_dir + 'work_dirs/yolov7_l_syncbn_fast_1x2b-100e_cowc_2/epoch_100.pth'  # noqa
