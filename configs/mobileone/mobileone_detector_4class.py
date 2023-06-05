model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileOne',
        arch='s0',
        out_indices=(3, ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=4,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original',
        ),
        topk=(1, 5),
    ))
# model settings
# import mmcv.runner.hooks.evaluation

# model = dict(
#     type='ImageClassifier',
#     backbone=dict(type='VGGNet',
#                   num_classes=4,
#                   stem_channels=12,
#                   stage_channels=(12, 16, 24, 32),
#                   block_per_stage=(1, 1, 1, 3),
#                   kernel_size=[3, 3, 3, 3],
#                   num_out=1
#                   ),
#     neck=dict(type='GlobalAveragePooling'),
#     head=dict(
#         type='StackedLinearCosineClsHead',
#         num_classes=4,
#         in_channels=32,
#         mid_channels=[32, 32],
#         dropout_rate=0,
#         loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
#         topk=(1, 5)))


# dataset settings
dataset_type = 'ImageNet'
img_lighting_cfg = dict(
    eigval=[55.4625, 4.7940, 1.1475],
    eigvec=[[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203]],
    alphastd=0.1,
    to_rgb=True)
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128, 128, 128], to_rgb=True)
data_root = '/media/traindata_ro/users/yl4203/coco_crop_data'
policies = [
    dict(type='AutoContrast', prob=0.5),
    dict(type='Equalize', prob=0.5),
    dict(type='Invert', prob=0.5),
    dict(
        type='Rotate',
        magnitude_key='angle',
        magnitude_range=(0,15),
        pad_val=0,
        prob=0.5,
        random_negative_prob=0.5),
    dict(
        type='Posterize',
        magnitude_key='bits',
        magnitude_range=(0, 4),
        prob=0.5),
    dict(
        type='Solarize',
        magnitude_key='thr',
        magnitude_range=(0, 256),
        prob=0.5),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110),
        thr=128,
        prob=0.5),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.),
    dict(
        type='Contrast',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.),
    dict(
        type='Brightness',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.),
    dict(
        type='Sharpness',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.5,
        direction='horizontal',
        random_negative_prob=0.5),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.5,
        direction='vertical',
        random_negative_prob=0.5),
    dict(
        type='Cutout',
        magnitude_key='shape',
        magnitude_range=(1, 41),
        pad_val=0,
        prob=0.5)
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=128, backend='pillow'),
    # dict(
    #     type='RandAugment',
    #     policies=policies,
    #     num_policies=2,
    #     magnitude_level=12),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    # dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    # dict(type='Lighting', **img_lighting_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=128, backend='pillow'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=data_root + '/train_add',
        classes=['bottle', 'chair', 'person', 'potted_plant'],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_root + '/real_val',
        ann_file=data_root + '/meta/real_val.txt',
        classes=['bottle', 'chair', 'person', 'potted_plant'],
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_root + '/real_val',
        ann_file=data_root + '/meta/real_val.txt',
        classes=['bottle', 'chair', 'person', 'potted_plant'],
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy', metric_options={'topk': (1, )}, save_best='accuracy_top-1')

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.005,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999))
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='step',
#                  step=[30, 60, 90],
#                  warmup='linear',
#                  warmup_iters=1000,)

lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20 * 1252,
    warmup_by_epoch=False)

# lr_config = dict(
#         policy='Cyclic',
#         by_epoch=False,
#         cyclic_times=3)
runner = dict(type='EpochBasedRunner', max_epochs=120) #120

# checkpoint saving
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/mobileone_detector_4class/mobileone_detector_4class'
