import torch
from src.preprocessor import HSIPreprocessor
from pathlib import Path

def test_indian_pines():
    # Setup paths
    data_dir = Path('data/ip')
    data_path = data_dir / 'Indian_pines_corrected.mat'
    gt_path = data_dir / 'Indian_pines_gt.mat'
    
    # Create preprocessor
    preprocessor = HSIPreprocessor(
        data_path=data_path,
        gt_path=gt_path,
        dataset_name='IP',
        n_components=30,
        window_size=25
    )
    
    # Run preprocessing pipeline
    split_data = preprocessor.preprocess_pipeline(train_size=0.3)
    
    # Print information about the results
    print("\nPreprocessing Results:")
    print(f"Training samples: {split_data['X_train'].shape}")
    print(f"Testing samples: {split_data['X_test'].shape}")
    print(f"Training labels shape: {split_data['y_train'].shape}")
    print(f"Testing labels shape: {split_data['y_test'].shape}")
    
    # Print class distribution
    print("\nClass distribution in training set:")
    for i in range(16):  # 16 classes
        count = (split_data['y_train'] == i).sum().item()
        print(f"Class {i+1}: {count} samples")
    
    # Verify data properties
    train_data = split_data['X_train']
    print("\nVerifying data properties:")
    print(f"Number of dimensions: {len(train_data.shape)}")
    print(f"Batch size (number of samples): {train_data.shape[0]}")
    print(f"Number of channels: {train_data.shape[1]}")
    print(f"Number of spectral bands: {train_data.shape[2]}")
    print(f"Spatial dimensions: {train_data.shape[3]}x{train_data.shape[4]}")
    
    # Assertions
    assert len(train_data.shape) == 5, "Expected 5D tensor (batch, channel, spectral, height, width)"
    assert train_data.shape[1] == 1, "Expected 1 channel"
    assert train_data.shape[2] == 30, "Expected 30 spectral bands"
    assert train_data.shape[3] == train_data.shape[4] == 25, "Expected 25x25 spatial dimensions"
    
    # Data quality checks
    assert not torch.isnan(train_data).any(), "Found NaN values in training data"
    assert not torch.isinf(train_data).any(), "Found infinite values in training data"
    
    print("\nAll checks passed!")
    
    return split_data

if __name__ == "__main__":
    split_data = test_indian_pines()