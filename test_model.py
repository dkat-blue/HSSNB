import torch
from src.models import HSSNB

def test_model():
    # Create model
    model = HSSNB(num_classes=16)
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 1, 30, 25, 25)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Print shapes
    print("\nInput shape:", x.shape)
    print("Output shape:", output.shape)
    
    # Verify output dimensions
    assert output.shape == (batch_size, 16), "Incorrect output shape"
    
    # Test model on GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        with torch.no_grad():
            output = model(x)
        print("\nModel successfully ran on GPU")
    
    print("\nAll tests passed!")
    
    return model

if __name__ == "__main__":
    model = test_model()