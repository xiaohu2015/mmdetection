_base_ = '../_base_/default_runtime.py'

dataset_type = 'CocoDataset'
data_root = '/home/hha/dataset/project/'
classes = ('out-ok', 'out-ng')

model = dict(
    type='SingleStageDetector',
    backbone=dict(type='TinyYolov4Backbone'),
    neck=None,
    bbox_head=dict(
        type='TinyYolov4Head',
        num_classes=2,
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[[29, 28], [31, 30], [30, 35], [31, 35], [35, 32],
                         [36, 35]]],
            strides=[16]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[16],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))

# dataset settings
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)


albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=2,
        interpolation=1,
        p=0.5),
    # dict(
    #     type='RandomBrightnessContrast',
    #     brightness_limit=0.9,
    #     contrast_limit=0.9,
    #     p=0.2),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         # dict(
    #         #     type='RGBShift',
    #         #     r_shift_limit=10,
    #         #     g_shift_limit=10,
    #         #     b_shift_limit=10,
    #         #     p=1.0),
    #         dict(
    #             type='HueSaturationValue',
    #             hue_shift_limit=20,
    #             sat_shift_limit=30,
    #             val_shift_limit=20,
    #             p=1.0)
    #     ],
    #     p=0.1),
    dict(
       type='Blur', blur_limit=1, p=0.3)
]


train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(448, 320), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    # dict(
    #     type='MinIoURandomCrop',
    #     min_ious=(0.8, 0.9),
    #     min_crop_size=0.9),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(448, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        pipeline=train_pipeline,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/'),
    val=dict(
        type=dataset_type,
        classes=classes,
        pipeline=test_pipeline,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        pipeline=test_pipeline,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/'))

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  # importent
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=600,  # same as burn-in in darknet importment
    warmup_ratio=0.1,
    step=[80, 100])
# runtime settings
total_epochs = 120
evaluation = dict(interval=5, metric=['bbox'])

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=3,max_keep_ckpts=5)

