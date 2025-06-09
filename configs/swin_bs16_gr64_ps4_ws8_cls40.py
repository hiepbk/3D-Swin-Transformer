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
train_batch_size = 32  # Increased from 16 for better GPU utilization
val_batch_size = 64    # Increased from 1 for faster validation

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
        num_workers = 8,        # Increased from 2 for faster data loading
        pin_memory = True,
        drop_last = True,
        split = "train",
        persistent_workers = True,  # Keep workers alive between epochs
        prefetch_factor = 4,        # Prefetch more batches
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
        # dict(
        #     name = "CrossEntropyLoss",
        #     loss_weight = 1.0,
        #     class_weight = [3.77, 0.77, 0.45, 2.00, 2.00, 0.86, 2.00, 0.59, 1.02, 1.16],
        #     label_mode = "single"
        #     ),
        dict(
            name = "FocalLoss",
            loss_weight = 1.0,
            # alpha = [3.77, 0.77, 0.45, 2.00, 2.00, 0.86, 2.00, 0.59, 1.02, 1.16],  # Class-specific weights - removed due to mismatch with 40 classes
            # To use class-specific weights, provide alpha values for all 40 classes:
            # alpha = [1.0] * 40,  # or calculate proper weights based on class frequencies
            # Alpha weights calculated based on class distribution (normalized inverse frequency)
            alpha = [
                        0.617,
                        1.234,
                        0.617,
                        3.085,
                        0.617,
                        0.617,
                        3.085,
                        0.617,
                        0.617,
                        3.085,
                        3.085,
                        3.085,
                        0.717,
                        3.085,
                        0.717,
                        3.085,
                        0.617,
                        0.617,
                        3.085,
                        3.085,
                        3.085,
                        0.617,
                        0.617,
                        0.717,
                        3.085,
                        0.617,
                        0.617,
                        3.085,
                        0.617,
                        3.085,
                        0.617,
                        3.085,
                        3.085,
                        0.617,
                        3.085,
                        0.617,
                        0.617,
                        0.617,
                        3.085,
                        3.085
                        ],
            gamma = 2.0,
            reduction = "mean",
            label_mode = "single"
        )
        ]
)

optimizer = dict(
    name = "AdamW",
    # lr = 0.0001,
    lr = 1e-3,
    weight_decay = 0.05,
    num_epochs = 100,
    seed = 42,
    deterministic = True,
    early_stopping_patience = 10,
    early_stopping_delta = 0.001
)

# Training optimizations
use_amp = True                    # Enable mixed precision training
gradient_accumulation_steps = 1   # Accumulate gradients over multiple steps
compile_model = True              # Use torch.compile for faster training (PyTorch 2.0+)

# Gradient clipping for stability
grad_clip = dict(
    max_norm = 1.0,
    norm_type = 2
)

lr_config = dict(
    policy = "CosineAnnealingLR",
    warmup = "linear",
    warmup_iters = 100,
    warmup_ratio = 0.05,
    # min_lr = 1e-6,
    min_lr = 3e-5,
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


