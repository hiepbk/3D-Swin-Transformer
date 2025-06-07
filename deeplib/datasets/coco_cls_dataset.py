import os
import torch
import numpy as np
from pycocotools.coco import COCO
from .base_dataset import BaseDataset
import cv2

class CocoClsDataset(BaseDataset):
    """
    COCO dataset class for classification.
    
    This dataset treats each image as having a single class label,
    which is determined by the most prominent object in the image.
    """
    def __init__(self, 
                 data_root, 
                 ann_file, 
                 img_prefix='', 
                 transforms=None,
                 label_mode='multi'):  # 'dominant', 'first', or 'multi'
        super().__init__(transforms)
        self.data_root = data_root
        self.ann_file = os.path.join(data_root, ann_file)
        self.img_prefix = os.path.join(data_root, img_prefix)
        self.label_mode = label_mode
        
        # Load COCO annotations
        self.coco = COCO(self.ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        
        # Filter images without annotations
        self.img_ids = self._filter_imgs()
        
        # Prepare image-to-label mapping
        self._prepare_labels()
            
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_prefix, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get label(s)
        if self.label_mode == 'multi':
            label = self.img_labels[img_id]  # One-hot encoded tensor
        else:
            label = torch.tensor(self.img_labels[img_id], dtype=torch.int64)
        

        
        # Apply transforms
        if self.transforms is not None:
            transformed_img = self.transforms(img)
            img = transformed_img
        
        data = {
            'img': img,
            'gt_label': label,
            'img_id': img_id,
            'file_name': img_info['file_name']
        }
        
        return data
    
    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_img_ids = []
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                valid_img_ids.append(img_id)
        return valid_img_ids
    
    def _prepare_labels(self):
        """Prepare image labels based on the label mode."""
        self.img_labels = {}
        
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            if self.label_mode == 'dominant':
                # Use the category of the largest object as the label
                max_area = -1
                dominant_cat = -1
                for ann in anns:
                    if ann['area'] > max_area:
                        max_area = ann['area']
                        dominant_cat = ann['category_id']
                self.img_labels[img_id] = self.cat2label[dominant_cat]
                
            elif self.label_mode == 'first':
                # Use the first annotation's category as the label
                self.img_labels[img_id] = self.cat2label[anns[0]['category_id']]
                
            elif self.label_mode == 'multi':
                # Multi-label classification: create one-hot encoded tensor
                label = torch.zeros(len(self.cat_ids), dtype=torch.float32)
                for ann in anns:
                    label[self.cat2label[ann['category_id']]] = 1.0
                self.img_labels[img_id] = label
    
    def get_annotations(self):
        """Get COCO annotations."""
        return self.coco
    
    def collate_fn(self, batch):
        """
        Collate function for DataLoader.
        
        Args:
            batch (list): List of samples from __getitem__
            
        Returns:
            dict: Collated batch with stacked tensors
        """
        imgs = torch.stack([sample['img'] for sample in batch])
        
        if self.label_mode == 'multi':
            labels = torch.stack([sample['gt_label'] for sample in batch])
        else:
            labels = torch.tensor([sample['gt_label'] for sample in batch])
            
        img_ids = [sample['img_id'] for sample in batch]
        file_names = [sample['file_name'] for sample in batch]
        
        batch_data = {
            'img': imgs,
            'gt_label': labels,
            'img_id': img_ids,
            'file_name': file_names
        }
        
        return batch_data



        
        