# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='RepVGGNet',
                  num_classes=2,
                  stem_channels=8,
                  stage_channels=(8, 16, 24, 32),
                  block_per_stage=(1, 1, 1, 3),
                  kernel_size=[3, 3, 3, 3],
                  num_out=1,
                  conv_type="RepVGGBlock"),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[1, 8]),
        topk=(1, 5),
    ))

# model = dict(
#     type='ImageClassifier',
#     backbone=dict(type='VGG', depth=11, num_classes=2),
#     neck=None,
#     head=dict(
#         type='ClsHead',
#         loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
#         topk=(1, 5),
#     ))


# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128, 128, 128], to_rgb=True)
data_root = '/media/traindata/coco_crop_data'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Pad', pad_to_square=True),
    dict(type='Resize', size=128, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Pad', pad_to_square=True),
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
        data_prefix=data_root + '/train_crop_hands_state_filted_lengthen',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_root + '/val_crop_hands_state_filted',
        ann_file=data_root + '/meta/hands_val_filted.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_root + '/val_crop_hands_state_filted',
        ann_file=data_root + '/meta/hands_val_filted.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy', save_best='accuracy_top-1')

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.005,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999))
# optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20 * 1252,
    warmup_by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=120)

# checkpoint saving
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/waving_hands_filted/repvgg_0.25_repvggnet_1113_nopad_AdamW_0.005_18_add_crop_aug_lengthen_very_easy'
