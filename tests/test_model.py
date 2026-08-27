import torch
import pytest
from src.model import get_model


def test_get_model_resnet18():
    """Test if the resnet18 model initializes correctly and outputs the right shape."""
    num_classes = 10
    model = get_model(architecture="resnet18", num_classes=num_classes)
    
    # Check if the model has the right number of output features
    assert model.fc.out_features == num_classes
    
    # Test forward pass with a dummy tensor (batch_size, channels, height, width)
    # CIFAR-10 images are 3x32x32
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    output = model(dummy_input)
    
    # Output should have shape (batch_size, num_classes)
    assert output.shape == (batch_size, num_classes)


def test_get_model_unknown_architecture():
    """Test if the correct exception is raised for unknown architecture."""
    with pytest.raises(ValueError, match="Unknown model architecture: unknown_arch"):
        get_model(architecture="unknown_arch", num_classes=10)
