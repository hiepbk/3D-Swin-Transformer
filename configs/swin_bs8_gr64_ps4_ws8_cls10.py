# Training parameters
grid_size = 64
patch_size = 4
window_size = 8
num_feat = 6
num_classes = 10

train_batch_size = 8
val_batch_size = 1

dataset = dict(
    name = "build_modelnet_dataset",
    data_root = "data",
    num_classes = num_classes,
    num_feat = num_feat,
    grid_size = grid_size,
    pc_range = [-1.0,-1.0,-1.0,1.0,1.0,1.0],
    label_mode = "single",
    train = dict(
        batch_size = train_batch_size,
        shuffle = True,
        num_workers = 2,
        pin_memory = True,
        drop_last = True,
        split = "train",
        transforms = [
            dict(type = "MeanGridVoxelize", 
                 grid_size = grid_size, 
                 pc_range = [-1.0,-1.0,-1.0,1.0,1.0,1.0], 
                 num_feat = num_feat)
        ]
    ),
    val = dict(
        batch_size = val_batch_size,
        shuffle = False,
        num_workers = 2,
        pin_memory = True,
        drop_last = True,
        split = "test",
        transforms = [
            dict(type = "MeanGridVoxelize", 
                 grid_size = grid_size, 
                 pc_range = [-1.0,-1.0,-1.0,1.0,1.0,1.0], 
                 num_feat = num_feat)
        ]
    )
)

model = dict(
    name = "ImageClassifier",
    backbone = dict(
        name = "SwinTransformer3D",
        grid_size = grid_size,
        patch_size = patch_size,
        in_chans = num_feat,
        embed_dim = 96,
        depths = [2, 2, 6, 2],
        num_heads = [3, 6, 12, 24],
        window_size = window_size,
        mlp_ratio = 4.,
        qkv_bias = True,
        drop_rate = 0.1,
        attn_drop_rate = 0.1,
        drop_path_rate = 0.1,
        ape = False,
        patch_norm = True,
        frozen_stages=-1,
        use_checkpoint=False
    ),
    neck = None,
    head = dict(
        name = "ClsHead3D",
        num_classes = num_classes,
        in_channels = 768,
        dropout = 0.1
    ),
    loss = dict(
        name = "CrossEntropyLoss",
        weight = 1.0
    )
)

optimizer = dict(
    name = "Adam",
    lr = 0.0001,
    weight_decay = 0.01,
    num_epochs = 30,
    seed = 42,
    deterministic = True,
    early_stopping_patience = 10,
    early_stopping_delta = 0.001
)

lr_config = dict(
    policy = "CosineAnnealingLR",
    warmup = "linear",
    warmup_iters = 500,
    warmup_ratio = 0.001,
    min_lr = 1e-6,
    by_epoch = True
)

load_from = None
resume_from = None

# Hooks configuration
hooks = [
    dict(
        type='LoggerHook',
        save_dir='logs',
        log_freq=10,
        val_epoch_interval=1
    ),
    dict(
        type='CheckpointHook',
        save_dir='ckpts',
        save_freq=10
    ),
    dict(type='LRSchedulerHook'),
    dict(type='OptimizerHook')
]


