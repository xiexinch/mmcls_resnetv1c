_base_ = ['default_runtime.py', 'schedules/imagenet_bs2048_rsb.py', 'datasets/imagenet_bs256_rsb_a12.py']

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MMSegResNetV1d',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        contract_dilation=False),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        topk=(1, 5),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.2, num_classes=1000),
        dict(type='CutMix', alpha=1.0, num_classes=1000)
    ]))
# dataset settings
train_dataloader = dict(sampler=dict(type='RepeatAugSampler', shuffle=True))

# schedule settings
optim_wrapper = dict(
    optimizer=dict(weight_decay=0.01),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=595,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=600)
]

train_cfg = dict(by_epoch=True, max_epochs=600)