# Swin Transformer for 3D Point Cloud Classification

This project implements a Swin Transformer architecture for 3D point cloud classification, specifically designed for the ModelNet10 dataset. The implementation adapts the Swin Transformer's hierarchical design for 3D point cloud data, leveraging its powerful feature extraction capabilities.

## Features

- 3D Swin Transformer architecture for point cloud processing
- Multi-scale feature extraction and hierarchical representation
- Support for ModelNet10 dataset
- Balanced training with class weights
- Early stopping and model checkpointing
- Comprehensive logging and visualization
- Learning rate scheduling with warmup

## Requirements

- Python 3.8+
- CUDA 11.7+ (for GPU acceleration)
- PyTorch 1.13.1+ with CUDA support
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Swin-Transformer-Point-Cloud.git
cd Swin-Transformer-Point-Cloud
```

2. Create and activate a conda environment:
```bash
conda create -n swin python=3.8
conda activate swin
```

3. Install PyTorch with CUDA support:
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

4. Install other dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the ModelNet10 dataset, which contains 10 categories of 3D CAD models. The dataset is organized as follows:

### ModelNet10 Classes
- bathtub (106 samples)
- bed (515 samples)
- chair (889 samples)
- desk (200 samples)
- dresser (200 samples)
- monitor (465 samples)
- night_stand (200 samples)
- sofa (680 samples)
- table (392 samples)
- toilet (344 samples)

### Data Structure
```
data/
├── modelnet10_shape_names.txt    # List of ModelNet10 class names
├── modelnet10_train.txt          # Training set file list
├── modelnet10_test.txt           # Test set file list
├── modelnet40_shape_names.txt    # List of ModelNet40 class names
├── modelnet40_train.txt          # ModelNet40 training set file list
├── modelnet40_test.txt           # ModelNet40 test set file list
├── filelist.txt                  # Complete file list
├── bathtub/                      # ModelNet10 class directory
├── bed/                          # ModelNet10 class directory
├── chair/                        # ModelNet10 class directory
├── desk/                         # ModelNet10 class directory
├── dresser/                      # ModelNet10 class directory
├── monitor/                      # ModelNet10 class directory
├── night_stand/                  # ModelNet10 class directory
├── sofa/                         # ModelNet10 class directory
├── table/                        # ModelNet10 class directory
├── toilet/                       # ModelNet10 class directory
└── [other ModelNet40 classes]/   # Additional ModelNet40 classes
```

### Dataset Files
- `modelnet10_shape_names.txt`: Contains the 10 class names for ModelNet10
- `modelnet10_train.txt`: List of training samples for ModelNet10
- `modelnet10_test.txt`: List of test samples for ModelNet10
- `filelist.txt`: Complete list of all point cloud files

### Class Distribution
The dataset shows significant class imbalance:
- Most frequent: chair (889 samples)
- Least frequent: bathtub (106 samples)
- Average samples per class: ~399 samples

## Configuration

The project uses a configuration system defined in `config.py`. Key parameters include:

### Model Architecture
```python
model_cfg = dict(
    grid_size = 64,
    patch_size = 4,
    in_chans = 6,
    num_classes = 10,
    embed_dim = 96,
    depths = [2, 2, 6, 2],
    num_heads = [3, 6, 12, 24],
    window_size = 8
)
```

### Training Parameters
```python
optimizer_cfg = dict(
    lr = 0.0001,
    weight_decay = 0.01,
    warmup_iterations = 500,
    num_epochs = 30,
    early_stopping_patience = 10
)
```

### Loss Function
```python
loss_cfg = dict(
    ce_smoothing = 0.1,
    focal_alpha = 0.25,
    focal_gamma = 2.0,
    focal_weight = 0.1
)
```

## Training

To train the model:

```bash
python train.py
```

The training process includes:
- Learning rate warmup (500 iterations)
- Class-balanced training with weights
- Early stopping (patience=10)
- Model checkpointing
- Comprehensive logging

### Monitoring Training

View training progress using TensorBoard:
```bash
tensorboard --logdir logs
```

## Model Architecture

The 3D Swin Transformer consists of:
- Patch embedding layer (64x64x64 → 16x16x16)
- Hierarchical Swin Transformer blocks
- Multi-scale feature extraction
- Classification head

Key components:
- 3D patch partitioning (4x4x4)
- Window-based self-attention (8x8x8)
- Shifted window mechanism
- Hierarchical feature maps

## Logging and Visualization

Training progress is logged to:
- TensorBoard logs in `logs/`
- Checkpoints in `ckpt/`
- Training metrics in JSON format

Metrics tracked:
- Loss (CE, Focal, Total)
- Accuracy
- F1 Score
- Per-class metrics
- Learning rate

## Results

The model achieves competitive performance on ModelNet10:
- Accuracy: [Your best accuracy]
- F1 Score: [Your best F1 score]
- Per-class metrics available in logs

## License

[Your chosen license]

## Citation

If you use this code in your research, please cite:
```bibtex
@article{swin-transformer,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

## Acknowledgments

- Original Swin Transformer paper and implementation
- ModelNet dataset creators
- PyTorch community