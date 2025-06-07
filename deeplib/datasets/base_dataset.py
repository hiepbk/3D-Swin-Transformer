import torch
from torch.utils.data import Dataset
from .transforms import build_transform

class BaseDataset(Dataset):
    """
    Base class for all datasets.
    """
    def __init__(self, dataset_cfg, transforms=None):
        self.cfg = dataset_cfg
        self.transforms = self.build_transforms(transforms)

    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def get_annotations(self):
        """
        Get dataset annotations.
        """
        raise NotImplementedError 
    
    def build_transforms(self, transforms_cfg):
        """Build transforms from config."""
        if transforms_cfg is None:
            return None
        
        transforms = []
        for transform_cfg in transforms_cfg:
            transform = build_transform(transform_cfg)
            transforms.append(transform)
        
        return transforms
    
    