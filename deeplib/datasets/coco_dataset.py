# This for loading the coco dataset using pytorch

import os
import torch
import numpy as np
from pycocotools.coco import COCO
from .base_dataset import BaseDataset
import cv2

class CocoDataset(BaseDataset):
    """
    COCO dataset class.
    """
    def __init__(self, 
                 data_root, 
                 ann_file, 
                 img_prefix='', 
                 transforms=None,
                 filter_empty_gt=True):
        super().__init__(transforms)
        self.data_root = data_root
        self.ann_file = os.path.join(data_root, ann_file)
        self.img_prefix = os.path.join(data_root, img_prefix)
        self.filter_empty_gt = filter_empty_gt
        
        # Load COCO annotations
        self.coco = COCO(self.ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        
        # Filter images without annotations if needed
        if self.filter_empty_gt:
            self.img_ids = self._filter_imgs()
            
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_info = self.coco.loadAnns(ann_ids)
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_prefix, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process annotations
        gt_bboxes = []
        gt_labels = []
        gt_masks = []
        
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            bbox = [x1, y1, x1 + w, y1 + h]
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
            
            if 'segmentation' in ann:
                mask = self.coco.annToMask(ann)
                gt_masks.append(mask)
        
        # Convert to tensors
        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32)
        gt_labels = torch.tensor(gt_labels, dtype=torch.int64)
        
        data = {
            'img': img,
            'img_shape': img.shape,
            'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels,
            'img_id': img_id,
            'file_name': img_info['file_name']
        }
        
        if len(gt_masks) > 0:
            data['gt_masks'] = np.stack(gt_masks)
        
        # Apply transforms
        if self.transforms is not None:
            data = self.transforms(data)
            
        return data
    
    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_img_ids = []
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                valid_img_ids.append(img_id)
        return valid_img_ids
    
    def get_annotations(self):
        """Get COCO annotations."""
        return self.coco
        