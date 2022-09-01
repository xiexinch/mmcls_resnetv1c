default_scope = 'mmcls'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ClsVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False
optim_wrapper = dict(
    optimizer=dict(type='Lamb', lr=0.005, weight_decay=0.01),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=595,
        eta_min=1e-06,
        by_epoch=True,
        begin=5,
        end=600)
]
train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=2048)
dataset_type = 'ImageNet'
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
bgr_mean = [103.53, 116.28, 123.675]
bgr_std = [57.375, 57.12, 58.395]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=7,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(type='PackClsInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=236,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]
train_dataloader = dict(
    batch_size=256,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                scale=224,
                backend='pillow',
                interpolation='bicubic'),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='RandAugment',
                policies='timm_increasing',
                num_policies=2,
                total_level=10,
                magnitude_level=7,
                magnitude_std=0.5,
                hparams=dict(pad_val=[104, 116, 124],
                             interpolation='bicubic')),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='RepeatAugSampler', shuffle=True),
    persistent_workers=True)
val_dataloader = dict(
    batch_size=256,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeEdge',
                scale=236,
                edge='short',
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)
val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_dataloader = dict(
    batch_size=256,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeEdge',
                scale=236,
                edge='short',
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)
test_evaluator = dict(type='Accuracy', topk=(1, 5))
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MMSegResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        topk=(1, 5)),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.2, num_classes=1000),
        dict(type='CutMix', alpha=1.0, num_classes=1000)
    ]))
launcher = 'none'
work_dir = './work_dirs/resnetv1c-d8_rsb'
