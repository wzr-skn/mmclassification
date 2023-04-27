# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV3', arch='small'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='StackedLinearClsHead',
        num_classes=6,
        in_channels=576,
        mid_channels=[1024],
        dropout_rate=0.2,
        act_cfg=dict(type='HSwish'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128, 128, 128], to_rgb=True)
data_root = '/home/ubuntu/my_datasets'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Pad', pad_to_square=True),
    dict(type='Resize', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Pad', pad_to_square=True),
    dict(type='Resize', size=224, backend='pillow'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=data_root + '/self_waving_hands_classfiaction/JPEGImages/',
        ann_file=data_root + '/self_waving_hands_classfiaction/txt_annotations/self_waving_hands_classfication_trainlabel.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_root + '/Stanford40/JPEGImages/',
        ann_file=data_root + '/Stanford40/txt_annotations/self_waving_hands_classfication_testlabel/self_waving_hands_classfication_testlabel.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_root + '/Stanford40/JPEGImages/',
        ann_file=data_root + '/Stanford40/txt_annotations/self_waving_hands_classfication_testlabel/self_waving_hands_classfication_testlabel.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=60)

# checkpoint saving
checkpoint_config = dict(interval=1)
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
resume_from = './work_dirs/waving_hand/mobilenet_selfdataset_6class_baseline/latest.pth'
workflow = [('train', 1)]
work_dir = './work_dirs/waving_hand/mobilenet_selfdataset_6class_baseline'
