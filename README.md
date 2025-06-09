# Swin Transformer for 3D Point Cloud Classification

This project implements a Swin Transformer architecture for 3D point cloud classification, specifically designed for the ModelNet10 dataset. The implementation adapts the Swin Transformer's hierarchical design for 3D point cloud data, leveraging its powerful feature extraction capabilities.

## DeepLib Framework

I built **DeepLib**, a comprehensive end-to-end deep learning framework built from scratch for computer vision tasks. DeepLib is designed with modularity, extensibility, and ease of use in mind, drawing inspiration from established frameworks such as:

- **Detectron2** (Facebook AI Research)
- **MMDetection** (OpenMMLab)
- **MMClassification** (OpenMMLab)
- **PaddlePaddle** (Baidu)

### Framework Philosophy

DeepLib follows modern deep learning framework design principles:
- **Modular Architecture**: Separate components for models, datasets, losses, and training logic
- **Registry System**: Dynamic module registration for easy extensibility
- **Configuration-Driven**: Flexible configuration system for experiments
- **Hook-Based Training**: Customizable training pipeline with hooks for logging, checkpointing, and scheduling
- **Unified Interface**: Consistent APIs across different vision tasks

### Core Components

```
deeplib/
├── config/          # Configuration management
├── core/           # Core utilities (evaluators, hooks)
├── datasets/       # Dataset implementations and transforms
├── engine/         # Training and inference engines
├── models/         # Model architectures, losses, and components
│   ├── architectures/  # Complete model definitions
│   ├── backbones/      # Feature extractors
│   ├── heads/          # Task-specific heads
│   ├── losses/         # Loss functions
│   └── necks/          # Feature pyramid networks
└── utils/          # Utilities and registry system
```

The framework demonstrates how to build a production-ready deep learning system with:
- **Flexible Model Construction**: Modular backbone + neck + head architecture
- **Advanced Loss Functions**: Custom implementations of CrossEntropy, Focal Loss, etc.
- **Comprehensive Training Pipeline**: Multi-hook system for logging, checkpointing, and scheduling
- **Experiment Management**: Organized output structure with versioning
- **Extensible Design**: Easy to add new models, datasets, and tasks



## Features

- 3D Swin Transformer architecture for point cloud processing
- Multi-scale feature extraction and hierarchical representation
- Support for ModelNet10 dataset
- Balanced training with class weights
- Early stopping and model checkpointing
- Comprehensive logging and visualization
- Learning rate scheduling with warmup
- Flexible configuration system
- Organized experiment management with work directories

## Requirements

- Python 3.8+
- CUDA 11.7+ (for GPU acceleration)
- PyTorch 1.13.1+ with CUDA support
- Cython (required for pycocotools)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hiepbk/3D-Swin-Transformer.git
cd 3D-Swin-Transformer
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

4. Install Cython (required for pycocotools):
```bash
conda install -y cython
```

5. Install other dependencies:
```bash
pip install -r requirements.txt
```

6. Install deeplib in development mode:
```bash
python setup.py develop
```

## Dataset

The project uses the ModelNet10 dataset, which contains 10 categories of 3D CAD models. You can download the dataset from the following link:

[Download ModelNet10 Dataset](https://www.kaggle.com/datasets/chenxaoyu/modelnet-normal-resampled?resource=download)

From Kaggle with resampled to shape 10,000 points

After downloading:
1. Extract the zip file
2. Place the extracted folder in the `data` directory of this project
3. The data structure should look like this:

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

## Configuration

The project uses a flexible configuration system. Configuration files are stored in the `configs` directory. Each configuration file defines parameters for:

- Dataset configuration
- Model architecture
- Training parameters
- Loss function
- Logging settings

Example configuration file (`configs/swin_bs8_gr64_ps4_ws8_cls10.py`):
```python
# Training parameters
grid_size = 64
patch_size = 4
window_size = 8
num_feat = 6
num_classes = 10

# Dataset configuration
dataset_cfg = dict(
    root_dir = "data",
    num_classes = num_classes,
    num_feat = num_feat,
    grid_size = grid_size,
    pc_range = [-1.0,-1.0,-1.0,1.0,1.0,1.0],
    ...
)

# Model configuration
model_cfg = dict(
    grid_size = grid_size,
    patch_size = patch_size,
    in_chans = num_feat,
    num_classes = num_classes,
    ...
)

# Training configuration
optimizer_cfg = dict(
    lr = 0.0001,
    weight_decay = 0.01,
    ...
)
```

## Training

To train the model, use the following command:

```bash
python tools/train.py/configs/swin_bs16_gr64_ps4_ws8_cls10.py [options]
```

### Command Line Arguments

- `config`: Path to the configuration file (required)
- `--work-dir`: Directory to save logs and models (optional)
- `--extra-tag`: Extra tag for the experiment (optional)

### Examples

```bash
# Basic training with default work directory
python tools/train.py configs/swin_bs16_gr64_ps4_ws8_cls10.py

# Training with experiment tag
python tools/train.py configs/swin_bs16_gr64_ps4_ws8_cls10.py --resume-from work_dirs/swin_bs16_gr64_ps4_ws8_cls10/experiment2/ckpts/epoch_0.pth --extra-tag experiment2


python tools/train.py configs/swin_bs16_gr64_ps4_ws8_cls10.py --extra-tag experiment2


python tools/train.py configs/swin_bs16_gr64_ps4_ws8_cls40.py --extra-tag experiment1


```

### Output Structure

The training outputs are organized as follows:
```
work_dirs/
└── config_name/
    └── [extra_tag]/
        └── YYYYMMDD_HHMMSS/
            ├── logs/
            │   ├── events.out.tfevents.*
            │   └── training.log
            └── ckpts/
                ├── latest.pth
                ├── epoch_*.pth
                └── best.pth
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
- TensorBoard logs in `{work_dir}/logs/`
- Checkpoints in `{work_dir}/ckpts/`
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