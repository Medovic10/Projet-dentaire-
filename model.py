import torch
import torch.nn as nn
from torchvision import models

def get_efficientnet_model(num_classes=7, dropout_prob=0.5):
    """
    EfficientNet-B0 pré-entraîné avec couche de régularisation Dropout.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_prob),
        nn.Linear(in_features, num_classes)
    )
    return model