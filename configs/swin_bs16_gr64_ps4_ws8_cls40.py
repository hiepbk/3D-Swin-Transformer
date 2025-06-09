# Training parameters
grid_size = 64
patch_size = 4
window_size = 8
num_feat = 6

class_names = [ "airplane",
                "bathtub",
                "bed",
                "bench",
                "bookshelf",
                "bottle",
                "bowl",
                "car",
                "chair",
                "cone",
                "cup",
                "curtain",
                "desk",
                "door",
                "dresser",
                "flower_pot",
                "glass_box",
                "guitar",
                "keyboard",
                "lamp",
                "laptop",
                "mantel",
                "monitor",
                "night_stand",
                "person",
                "piano",
                "plant",
                "radio",
                "range_hood",
                "sink",
                "sofa",
                "stairs",
                "stool",
                "table",
                "tent",
                "toilet",
                "tv_stand",
                "vase",
                "wardrobe",
                "xbox",
                    ]
num_classes = len(class_names)

# Performance optimizations
train_batch_size = 16  # Reduced from 32 for more stable training
val_batch_size = 32    # Reduced from 64

dataset = dict(
    name = "build_modelnet_dataset",
    data_root = "data/modelnet",
    num_classes = num_classes,
    class_names = class_names,
    num_feat = num_feat,
    grid_size = grid_size,
    pc_range = [-1.0,-1.0,-1.0,1.0,1.0,1.0],
    label_mode = "single",
    train = dict(
        batch_size = train_batch_size,
        shuffle = True,
        num_workers = 6,        # Reduced from 8 to avoid bottlenecks
        pin_memory = True,
        drop_last = True,
        split = "train",
        persistent_workers = True,  # Keep workers alive between epochs
        prefetch_factor = 2,        # Reduced from 4
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
        num_workers = 4,        # Increased from 2
        pin_memory = True,
        drop_last = False,      # Don't drop last for validation
        split = "test",
        persistent_workers = True,
        prefetch_factor = 2,
        transforms = [
            dict(type = "MeanGridVoxelize", 
                 grid_size = grid_size, 
                 pc_range = [-1.0,-1.0,-1.0,1.0,1.0,1.0], 
                 num_feat = num_feat)
        ]
    )
)

model = dict(
    name = "Classifier",
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
    loss = [
        dict(
            name = "CrossEntropyLoss",
            loss_weight = 1.0,
            # Use more balanced class weights
            class_weight = [1.5, 3.0, 1.5, 2.5, 1.5, 2.0, 4.0, 2.0, 1.0, 2.5, 
                           4.0, 3.0, 2.0, 3.0, 2.0, 2.5, 2.5, 2.5, 2.5, 3.0,
                           2.5, 1.5, 1.2, 2.0, 3.5, 1.8, 1.8, 3.0, 2.8, 3.0,
                           1.0, 3.0, 3.5, 1.5, 2.5, 1.8, 1.8, 1.2, 3.5, 3.0],
            label_mode = "single"
        )
        # dict(
        #     name = "FocalLoss",
        #     loss_weight = 1.0,
        #     # Alpha weights calculated based on class distribution (normalized inverse frequency)
        #     alpha = [
        #                 0.617,
        #                 1.234,
        #                 0.617,
        #                 3.085,
        #                 0.617,
        #                 0.617,
        #                 3.085,
        #                 0.617,
        #                 0.617,
        #                 3.085,
        #                 3.085,
        #                 3.085,
        #                 0.717,
        #                 3.085,
        #                 0.717,
        #                 3.085,
        #                 0.617,
        #                 0.617,
        #                 3.085,
        #                 3.085,
        #                 3.085,
        #                 0.617,
        #                 0.617,
        #                 0.717,
        #                 3.085,
        #                 0.617,
        #                 0.617,
        #                 3.085,
        #                 0.617,
        #                 3.085,
        #                 0.617,
        #                 3.085,
        #                 3.085,
        #                 0.617,
        #                 3.085,
        #                 0.617,
        #                 0.617,
        #                 0.617,
        #                 3.085,
        #                 3.085
        #                 ],
        #     gamma = 2.0,
        #     reduction = "mean",
        #     label_mode = "single"
        # )
        ]
)

optimizer = dict(
    name = "AdamW",
    lr = 1e-4,              # Reduced from 6e-4 for more stable training
    weight_decay = 0.01,    # Reduced from 0.05 to prevent over-regularization
    num_epochs = 100,       # Increased from 40 for better convergence
    seed = 42,
    deterministic = True,
    early_stopping_patience = 15,  # Increased patience
    early_stopping_delta = 0.001
)

# Training optimizations
use_amp = True                    # Enable mixed precision training
gradient_accumulation_steps = 1   # Accumulate gradients over multiple steps
compile_model = False             # Disable for debugging, enable later for speed

# Gradient clipping for stability
grad_clip = dict(
    max_norm = 0.5,    # Reduced from 1.0 for more aggressive clipping
    norm_type = 2
)

lr_config = dict(
    policy = "CosineAnnealingLR",
    warmup = "linear",
    warmup_iters = 500,     # Increased from 150 for better warmup
    warmup_ratio = 0.01,    # Reduced from 0.1 for gentler warmup
    min_lr = 1e-6,          # Reduced minimum learning rate
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
        save_freq=1
    ),
    dict(type='LRSchedulerHook'),
    dict(type='OptimizerHook')
]


