from deeplib.utils.registry import Registry
from .coco_cls_dataset import CocoClsDataset
from .modelnet_dataset import ModelNetDataset
DATASET_REGISTRY = Registry('dataset')

def build_dataset(cfg):
    """Build dataset from config."""
    dataset_cfg = cfg.dataset
    dataset_name = dataset_cfg.name
    
    dataset = DATASET_REGISTRY.get(dataset_name)(dataset_cfg)
    return dataset

@DATASET_REGISTRY.register_module()
def build_coco_cls_dataset(dataset_cfg):
    data_root = dataset_cfg.data_root
    train_cfg = dataset_cfg.train
    val_cfg = dataset_cfg.val

    train_set = CocoClsDataset(data_root=data_root, 
                               ann_file=train_cfg.ann_file, 
                               img_prefix=train_cfg.img_prefix, 
                               label_mode=dataset_cfg.label_mode, 
                               transforms=train_cfg.transforms)
    
    # Use val transforms if available, otherwise use the same as train
    val_transforms = getattr(val_cfg, 'transforms', train_cfg.transforms)
    val_set = CocoClsDataset(data_root=data_root, 
                             ann_file=val_cfg.ann_file, 
                             img_prefix=val_cfg.img_prefix, 
                             label_mode=dataset_cfg.label_mode, 
                             transforms=val_transforms)
    return train_set, val_set

@DATASET_REGISTRY.register_module()
def build_modelnet_dataset(dataset_cfg):
    train_set = ModelNetDataset(dataset_cfg,
                               split=dataset_cfg.train.split,
                               transforms=dataset_cfg.train.transforms)
    val_set = ModelNetDataset(dataset_cfg,
                             split=dataset_cfg.val.split,
                             transforms=dataset_cfg.val.transforms)
    return train_set, val_set