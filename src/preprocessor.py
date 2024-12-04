import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import torch
from typing import Tuple, List, Dict

class HSIPreprocessor:
    """Preprocessor for Hyperspectral Image data"""
    
    # Dictionary defining water absorption bands to remove for each dataset
    WATER_ABSORPTION_BANDS = {
        'IP': list(range(104,108)) + list(range(150,163)),  # Indian Pines
        'PU': [],  # Pavia University - no removal needed
        'SA': list(range(108,112)) + list(range(154,167))   # Salinas
    }

    def __init__(self, 
                 data_path: str, 
                 gt_path: str, 
                 dataset_name: str,
                 n_components: int = 30,
                 window_size: int = 25,
                 remove_water_bands: bool = True):
        """
        Args:
            data_path: Path to the .mat file containing HSI data
            gt_path: Path to the .mat file containing ground truth
            dataset_name: Name of dataset ('IP', 'PU', or 'SA')
            n_components: Number of components for PCA reduction
            window_size: Size of spatial window for patch extraction
            remove_water_bands: Whether to remove water absorption bands
        """
        self.data_path = Path(data_path)
        self.gt_path = Path(gt_path)
        self.dataset_name = dataset_name
        self.n_components = n_components
        self.window_size = window_size
        self.remove_water_bands = remove_water_bands
        
        # Load and preprocess data
        self.data, self.gt = self._load_data()
        if self.remove_water_bands:
            self.data = self._remove_water_bands()
        self.data_pca = None  # Will be set after PCA
        
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data and ground truth from .mat files"""
        try:
            data_dict = loadmat(self.data_path)
            gt_dict = loadmat(self.gt_path)
            
            # Get the actual data arrays (last key in dict)
            data = data_dict[list(data_dict.keys())[-1]]
            gt = gt_dict[list(gt_dict.keys())[-1]]
            
            print(f"Loaded data shape: {data.shape}")
            print(f"Loaded ground truth shape: {gt.shape}")
            
            return data, gt
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _remove_water_bands(self) -> np.ndarray:
        """Remove water absorption bands"""
        if self.dataset_name not in self.WATER_ABSORPTION_BANDS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        bands_to_remove = self.WATER_ABSORPTION_BANDS[self.dataset_name]
        if not bands_to_remove:
            return self.data
            
        mask = np.ones(self.data.shape[-1], dtype=bool)
        mask[bands_to_remove] = False
        data_filtered = self.data[:, :, mask]
        
        print(f"Removed {len(bands_to_remove)} water absorption bands")
        print(f"New data shape: {data_filtered.shape}")
        
        return data_filtered
    
    def apply_pca(self) -> np.ndarray:
        """Apply PCA dimensionality reduction"""
        # Reshape to 2D array
        h, w, c = self.data.shape
        data_2d = self.data.reshape(-1, c)
        
        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_2d)
        
        # Apply PCA
        pca = PCA(n_components=self.n_components)
        data_pca = pca.fit_transform(data_scaled)
        
        # Reshape back to 3D
        self.data_pca = data_pca.reshape(h, w, self.n_components)
        
        print(f"Applied PCA: {c} bands → {self.n_components} components")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        return self.data_pca
    
    def create_patches(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create patches around each labeled pixel"""
        if self.data_pca is None:
            raise ValueError("Must run apply_pca() before creating patches")
            
        pad_size = self.window_size // 2
        padded_data = np.pad(
            self.data_pca,
            ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode='reflect'
        )
        
        patches, labels = [], []
        h, w = self.gt.shape
        
        for i in range(h):
            for j in range(w):
                if self.gt[i, j] != 0:  # Skip background
                    patch = padded_data[
                        i:i+self.window_size,
                        j:j+self.window_size,
                        :
                    ]
                    patches.append(patch)
                    labels.append(self.gt[i, j] - 1)  # Convert to 0-based indexing
        
        patches = np.array(patches)
        labels = np.array(labels)
        
        print(f"Created {len(patches)} patches of shape {patches.shape}")
        return patches, labels
    
    def create_train_test_split(self, 
                              patches: np.ndarray, 
                              labels: np.ndarray,
                              train_size: float = 0.3,
                              random_state: int = 42) -> Dict[str, np.ndarray]:
        """Create stratified train/test split"""
        np.random.seed(random_state)
        
        # Split indices by class
        classes = np.unique(labels)
        train_idx, test_idx = [], []
        
        for c in classes:
            idx = np.where(labels == c)[0]
            n_train = int(len(idx) * train_size)
            
            # Shuffle indices
            np.random.shuffle(idx)
            train_idx.extend(idx[:n_train])
            test_idx.extend(idx[n_train:])
        
        # Create final splits
        return {
            'X_train': patches[train_idx],
            'X_test': patches[test_idx],
            'y_train': labels[train_idx],
            'y_test': labels[test_idx]
        }
    
    def preprocess_pipeline(self, train_size: float = 0.3) -> Dict[str, np.ndarray]:
        """Run the complete preprocessing pipeline"""
        self.apply_pca()
        patches, labels = self.create_patches()
        split_data = self.create_train_test_split(patches, labels, train_size)
        
        # Convert to torch tensors
        split_data = {
            k: torch.from_numpy(v).float() for k, v in split_data.items()
        }
        
        # Rearrange dimensions to (batch, channel=1, spectral=30, height=25, width=25)
        split_data['X_train'] = split_data['X_train'].permute(0, 3, 1, 2).unsqueeze(1)
        split_data['X_test'] = split_data['X_test'].permute(0, 3, 1, 2).unsqueeze(1)
        
        return split_data