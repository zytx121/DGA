_base_ = [
    '../_base_/datasets/dota_airplane.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
pwd_dir = '/home/zytx121/mmrotate/'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

custom_imports = dict(
    imports=['projects.camors.camors'], allow_failed_imports=False)

model = dict(
    type='OBJRTMDet',
    patch_size=300,
    printfile=pwd_dir + 'projects/camors/camors/patch/30values.txt',
    # patch_dir=pwd_dir + 'obj_rtm_plane/150.npy',
    adv_img_dir=pwd_dir + 'obj_rtm_plane/adv_img/',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=1,
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.1),
        max_per_img=2000),
)

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
    dict(type='SavePatchHook', patch_dir=pwd_dir + 'obj_rtm_plane/')
]

load_from = pwd_dir + 'work_dirs/rtmdet_tiny_3x_dota_airplane/epoch_36.pth'  # noqa

default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=3))

test_evaluator = dict(
    type='VOCMetric',
    iou_thrs=0.5,
    metric='mAP',
    eval_mode='11points',
)
