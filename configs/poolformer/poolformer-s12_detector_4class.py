_base_ = [
    '../_base_/datasets/imagenet_bs128_poolformer_small_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PoolFormer',
        arch='s12',
        drop_path_rate=0.1,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['GroupNorm'], val=1., bias=0.),
        ],
        _scope_='mmpretrain',),
    neck=dict(type='GlobalAveragePooling', _scope_='mmpretrain',),
    head=dict(
        type='LinearClsHead',
        num_classes=4,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        _scope_='mmpretrain',
    ),
    _scope_='mmpretrain',)



img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128, 128, 128], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224, backend='pillow'),
    # dict(
    #     type='RandAugment',
    #     policies=policies,
    #     num_policies=2,
    #     magnitude_level=12),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    # dict(type='Lighting', **img_lighting_cfg),
    dict(type='PackInputs'),
]

dataset_type = 'ImageNet'
data_root = '/media/traindata_ro/users/yl4203/coco_crop_data'
train_dataloader = dict(
    _delete_=True,
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root + '/train_add',
        metainfo={'classes': ['bottle', 'chair', 'person', 'potted_plant']},
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root + '/real_val',
        ann_file=data_root + '/meta/real_val.txt',
),)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/mobileone_detector_4class/efficientformer-l1_detector_4class'
