# Training parameters
grid_size = 64
patch_size = 4
window_size = 8
num_feat = 6
num_classes = 10

train_batch_size = 8
val_batch_size = 1

dataset_cfg = dict(
    root_dir = "data",
    num_classes = num_classes,
    num_feat = num_feat,
    grid_size = grid_size,
    pc_range = [-1.0,-1.0,-1.0,1.0,1.0,1.0],

    train_cfg = dict(
        batch_size = train_batch_size,
        shuffle = True,
        num_workers = 2,
        pin_memory = True,
        drop_last = True,
        split = "train",
    ),
    val_cfg = dict(
        batch_size = val_batch_size,
        shuffle = False,
        num_workers = 2,
        pin_memory = True,
        drop_last = True,
        split = "test",
    ),
)

model_cfg = dict(
        grid_size = grid_size,
        patch_size = patch_size,
        in_chans = num_feat,
        num_classes = num_classes,
        embed_dim = 96,
        depths = [2, 2, 6, 2],
        num_heads = [3, 6, 12, 24],
        window_size = window_size,
        mlp_ratio = 4.,
        qkv_bias = True,
        drop_rate = 0.,
        attn_drop_rate = 0.,
        drop_path_rate = 0.
)

optimizer_cfg = dict(
    name = "Adam",
    lr = 0.0001,
    weight_decay = 0.01,
    warmup_iterations = 500,
    min_lr = 1e-6,
    num_epochs = 30,
    # Random seed configuration
    seed = 42,
    deterministic = True,
    early_stopping_patience = 10,
    early_stopping_delta = 0.001,
)

# Loss function parameters
loss_cfg = dict(
    ce_smoothing = 0.1,
    focal_alpha = 0.25,
    focal_gamma = 2.0,
    focal_weight = 0.1
)

# Logging parameters
log_cfg = dict(
    log_dir = "logs",
    ckpt_dir = "ckpt",
    save_freq = 1,
    resume = False,
    resume_path = None,
    resume_epoch = None,
    resume_optimizer = False,
    resume_optimizer_path = None,
    resume_optimizer_epoch = None,
    log_interval = 100,
    save_interval = 10000,
)
