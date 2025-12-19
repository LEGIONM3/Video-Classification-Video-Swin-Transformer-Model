# Swin Transformer for Video

## Model Architecture
- **Type**: Video Swin Transformer (Tiny - Swin3D-T)
- **Source**: Torchvision `models.swin3d_t`.
- **Modifications**: Classification head (Linear) modified to output 2 classes (Binary).
- **Features**: Hierarchical transformer with shifted windows, adapted for 3D Video processing.

## Dataset Structure
Expects `Dataset` folder in parent directory.
```
Dataset/
├── violence/
└── no-violence/
```

## How to Run
1. Install dependencies: `torch`, `opencv-python`, `scikit-learn`, `numpy`, `torchvision`.
2. Run `python train.py`.
