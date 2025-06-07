import torch.nn as nn
import torch.nn.functional as F
from deeplib.models.backbones.build import build_backbone
from deeplib.models.necks.build import build_neck
from deeplib.models.heads.build import build_head
from deeplib.models.losses.build import build_loss
import torch
from .build import ARCHITECTURE_REGISTRY

@ARCHITECTURE_REGISTRY.register_module()
class ImageClassifier(nn.Module):
    """
    Basic image classifier architecture with backbone, neck, and head.
    
    Args:
        backbone (dict): Config dict for backbone
        neck (dict, optional): Config dict for neck
        head (dict): Config dict for classification head
        pretrained (bool): Whether to load pretrained weights
    """
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        
        self.cfg = cfg
        
        # Build backbone
        self.backbone = build_backbone(cfg)
        
        # Build neck
        self.neck = build_neck(cfg)
        
        # Build head
        self.head = build_head(cfg)
        
        # Build loss
        self.loss_func = build_loss(cfg)
        
        # Initialize weights
        self.init_weights(pretrained)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def init_weights(self, pretrained=False):
        """Initialize the weights."""
        # Backbone weights are initialized in the backbone class
        # No need to initialize neck and head here as they're already initialized
        pass
    
    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x
    
    def forward_train(self, batch_data):
        """Forward computation during training."""
        img = batch_data['tensor_data'].to(self.device)
        label = batch_data['gt_label'].to(self.device)
        
        x = self.extract_feat(img)
        cls_score = self.head(x)
        
        # compute loss
        loss = self.loss_func(cls_score, label)
        
        return loss
    
    def forward_test(self, batch_data):
        """Forward computation during testing."""
        batch_data['tensor_data'] = batch_data['tensor_data'].to(self.device)
        batch_data['gt_label'] = batch_data['gt_label'].to(self.device)
        x = self.extract_feat(batch_data['tensor_data'])
        cls_score = self.head(x)
        pred = F.softmax(cls_score, dim=1)

        #compute loss
        loss = self.loss_func(cls_score, batch_data['gt_label'])
        return pred, loss
    
    def forward(self, batch_data, istrain=True):
        """
        Args:
            tensor_data (torch.Tensor): Input point cloud of shape (N, C, D, H, W)
            gt_labels (torch.Tensor): Ground-truth labels
            return_loss (bool): Whether to return loss or prediction
            
        Returns:
            dict or torch.Tensor: Loss dict or prediction
        """
        if istrain:
            return self.forward_train(batch_data)
        else:
            return self.forward_test(batch_data) 
        