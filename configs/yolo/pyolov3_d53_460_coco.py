_base_ = './yolov3_d53_mstrain-608_273e_coco.py'
# dataset settings
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data_root = '/home/SENSETIME/huanghaian/dataset/project/'

classes = ('out-ok', 'out-ng')

model = dict(bbox_head=dict(num_classes=2))

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        classes=classes,
        pipeline=train_pipeline,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/'),
    val=dict(
        classes=classes,
        pipeline=test_pipeline,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/'),
    test=dict(
        classes=classes,
        pipeline=test_pipeline,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/'))

load_from = '/home/SENSETIME/huanghaian/Downloads/yolov3_d53_320_273e_coco-421362b6.pth'

optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001)
lr_config = dict(policy='step', step=[300, 330])
total_epochs = 360
