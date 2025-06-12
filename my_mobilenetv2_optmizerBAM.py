# model settings  mobilenet-v2_8xb32_in1k.py

model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2BAM', widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    #neck=dict(type='GeneralizedMeanPooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        #loss=dict(type='FocalLoss', loss_weight=1.0),
        topk=(1, 3),
    ))

#模型分类是五类
train_cfg = dict(mixup=dict(alpha=0.2, num_classes=5))
#从训练好的模型微调，顺序不重要
load_from = 'mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'

# dataset settings  imagenet_bs32_pil_bicubic.py
dataset_type = 'WBCNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='E:\WBC_detection\pycharmUsing\SiChuanPH_MaiRuiFiveCells_TrainAndTest_lowRecongnize/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='E:\WBC_detection\pycharmUsing\SiChuanPH_MaiRuiFiveCells_TrainAndTest_lowRecongnize/val',
       # ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='E:\WBC_detection\pycharmUsing\SiChuanPH_MaiRuiFiveCells_TrainAndTest_lowRecongnize/test',
        #ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
#修改
#evaluation = dict(interval=1, save_best='best_model', metric='accuracy', metric_options={'topk': (1, )})
evaluation = dict(interval=1, metric='accuracy')
# optimizer  imagenet_bs256_epochstep.py
# 学习率人家用的8卡，我们要除以8，且还要再小一点

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.00004)
#optimizer = dict(type='Adam', lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='step', gamma=0.98, step=1)
#lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=200)


checkpoint_config = dict(interval=5)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_param = dict(backend='nccl')
log_level = 'INFO'

resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/my_mobilenet_v2_1x'
gpu_ids = [0]


