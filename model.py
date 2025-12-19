import torch.nn as nn
import torchvision.models.video as models

def build_swin_model():
    print("Initializing Video Swin Transformer...")
    # Using torchvision's Swin3D-T (Tiny)
    # Weights=None for scratch
    model = models.swin3d_t(weights=None) 
    
    # Modify Head for Binary Classification
    # Original head is model.head (Linear)
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, 2)
    
    return model
