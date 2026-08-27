import torch.nn as nn
from torchvision import models


def get_model(architecture: str, num_classes: int) -> nn.Module:
    """
    Instantiates and returns a PyTorch model based on the specified architecture.
    Uses pre-trained weights for faster convergence.
    """
    if architecture == "resnet18":
        # Load a ResNet-18 model with pre-trained weights
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        # Modify the final fully connected layer to match the number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    # Add other architectures here if needed
    else:
        raise ValueError(f"Unknown model architecture: {architecture}")
