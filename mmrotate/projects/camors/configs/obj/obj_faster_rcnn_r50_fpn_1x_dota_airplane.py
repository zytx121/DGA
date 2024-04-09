_base_ = [
    '../_base_/datasets/dota_airplane.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
pwd_dir = '/home/zytx121/mmrotate/'

custom_imports = dict(
    imports=['projects.camors.camors'], allow_failed_imports=False)

# model settings
model = dict(
    type='OBJFasterRCNN',
    patch_size=300,
    printfile=pwd_dir + 'projects/camors/camors/patch/30values.txt',
    # patch_dir=pwd_dir + 'obj_fr_plane/150.npy',
    adv_img_dir=pwd_dir + 'obj_fr_plane/adv_img/',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.1),
            max_per_img=2000)))

train_dataloader = dict(batch_size=1, num_workers=2)

max_epochs = 10
base_lr = 0.03

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=max_epochs)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={'patch_genetator.patch': dict(lr_mult=1e20)},
        bypass_duplicate=True),
    optimizer=dict(type='Adam', lr=base_lr * 1e-20))

param_scheduler = [
    dict(type='ReduceOnPlateauLR', monitor='loss', rule='less', patience=50),
]

custom_hooks = [
    dict(type='SavePatchHook', patch_dir=pwd_dir + 'obj_fr_plane/')
]

load_from = pwd_dir + 'work_dirs/faster_rcnn_r50_fpn_1x_dota_airplane/epoch_12.pth'  # noqa

default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=3))

test_evaluator = dict(
    type='VOCMetric',
    iou_thrs=0.5,
    metric='mAP',
    eval_mode='11points',
)
