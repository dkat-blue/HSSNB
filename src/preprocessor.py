import numpy as np
from scipy.io import loadmat
import rasterio
import spectral.io.envi as envi
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from pathlib import Path

class HSIPreprocessor:
    """Preprocessor for Hyperspectral Image data."""
    
    def __init__(self, data_path, gt_path, dataset_name, n_components=30,
                 window_size=25, remove_water_bands=False):
        """Initialize preprocessor with data paths and parameters."""
        self.data_path = data_path
        self.gt_path = gt_path
        self.dataset_name = dataset_name.upper()
        self.n_components = n_components
        self.window_size = window_size
        self.remove_water_bands = remove_water_bands

        # Dataset-specific configurations
        if self.dataset_name == 'HO':  # Houston GRSS dataset
            self.data_key = None  # ENVI format doesn't use keys
            self.gt_key = None
            self.water_bands = []  # GRSS data already has water bands removed
        else:  # IP/PU/SA datasets
            if self.dataset_name == 'IP':
                self.data_key = 'indian_pines_corrected'
                self.gt_key = 'indian_pines_gt'
            elif self.dataset_name == 'PU':
                self.data_key = 'paviaU'
                self.gt_key = 'paviaU_gt'
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
                
            # Water absorption bands (only for IP/PU/SA)
            self.water_bands = [104-1, 105-1, 106-1, 107-1, 108-1, 109-1, 110-1, 111-1, 112-1,
                              113-1, 114-1, 115-1, 116-1, 117-1, 118-1, 119-1, 120-1, 121-1, 
                              122-1, 123-1, 124-1, 125-1, 126-1, 127-1, 128-1, 129-1, 130-1,
                              131-1, 132-1, 133-1, 134-1, 135-1, 136-1, 137-1, 138-1, 139-1,
                              140-1, 141-1, 142-1, 143-1, 144-1, 145-1, 146-1, 147-1, 148-1,
                              149-1, 150-1, 151-1, 152-1, 153-1, 154-1, 155-1, 156-1, 157-1,
                              158-1, 159-1, 160-1, 161-1, 162-1, 163-1, 164-1, 165-1, 166-1,
                              167-1, 168-1, 169-1, 170-1, 171-1, 172-1, 173-1, 174-1, 175-1,
                              176-1, 177-1, 178-1, 179-1, 180-1, 181-1, 182-1, 183-1, 184-1,
                              185-1, 186-1, 187-1, 188-1, 189-1, 190-1, 191-1, 192-1, 193-1,
                              194-1, 195-1, 196-1, 197-1, 198-1, 199-1, 200-1]

        # Load data
        self.data, self.gt = self._load_data()
        if self.remove_water_bands and self.dataset_name != 'HO':
            self.data = self._remove_water_bands()

    def _load_data(self):
        """Load data based on dataset type."""
        try:
            if self.dataset_name == 'HO':
                return self._load_houston_data()
            else:
                return self._load_mat_data()
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def _load_houston_data(self):
        """Load Houston GRSS dataset."""
        try:
            # Load HSI data
            hsi_header = os.path.splitext(self.data_path)[0] + '.hdr'
            hsi_data = envi.open(hsi_header, image=self.data_path)
            data = np.array(hsi_data.read_bands(range(48)))  # Exclude auxiliary bands
            
            # Load ground truth
            with rasterio.open(self.gt_path) as src:
                gt = src.read(1)
                
            # Align spatial dimensions
            data = self._align_spatial_dimensions(data, gt)
            
            return data, gt
        except Exception as e:
            raise Exception(f"Error loading Houston data: {str(e)}")

    def _load_mat_data(self):
        """Load .mat format datasets."""
        try:
            data = loadmat(self.data_path)[self.data_key]
            gt = loadmat(self.gt_path)[self.gt_key]
            return data, gt
        except Exception as e:
            raise Exception(f"Error loading .mat data: {str(e)}")

    def _align_spatial_dimensions(self, data, gt):
        """
        Align HSI data spatial dimensions with ground truth by finding the largest 
        common region between both images.
        
        Returns:
            data: Aligned HSI data
            valid_mask: Boolean mask indicating valid (non-padded) regions
        """
        # Ensure data is in (H, W, C) format
        if data.shape[-1] != min(data.shape):
            data = np.transpose(data, (1, 2, 0))
            
        h_x, w_x, c = data.shape
        h_y, w_y = gt.shape
        
        # Find intersection dimensions
        h_out = min(h_x, h_y)
        w_out = min(w_x, w_y)
        
        # Calculate offsets to center the crop
        h_start_x = (h_x - h_out) // 2
        w_start_x = (w_x - w_out) // 2
        h_start_y = (h_y - h_out) // 2
        w_start_y = (w_y - w_out) // 2
        
        # Crop both images to intersection
        data_cropped = data[h_start_x:h_start_x+h_out, 
                          w_start_x:w_start_x+w_out, :]
        gt_cropped = gt[h_start_y:h_start_y+h_out,
                       w_start_y:w_start_y+w_out]
        
        # Create validity mask (all True since we're cropping)
        valid_mask = np.ones((h_out, w_out), dtype=bool)
        
        self.valid_mask = valid_mask
        self.gt = gt_cropped
        return data_cropped

    def _remove_water_bands(self):
        """Remove water absorption bands."""
        bands = list(range(self.data.shape[2]))
        for band in sorted(self.water_bands, reverse=True):
            del bands[band]
        return self.data[:, :, bands]

    def preprocess_pipeline(self, train_size=0.3):
        """Run full preprocessing pipeline."""
        raise NotImplementedError("Method not implemented yet")

    def _apply_pca(self):
        """Apply PCA to reduce dimensionality."""
        raise NotImplementedError("Method not implemented yet")

    def _create_patches(self, data, gt, valid_mask, patch_size=25):
        """
        Create patches for training, only from valid regions.
        
        Args:
            data: HSI data of shape (H, W, C)
            gt: Ground truth of shape (H, W)
            valid_mask: Boolean mask of shape (H, W)
            patch_size: Size of patches to extract
            
        Returns:
            patches: List of (patch_size, patch_size, C) arrays
            labels: List of corresponding labels
        """
        h, w = valid_mask.shape
        half_patch = patch_size // 2
        
        patches = []
        labels = []
        
        # Only extract patches centered on valid pixels with valid neighborhoods
        for i in range(half_patch, h - half_patch):
            for j in range(half_patch, w - half_patch):
                # Check if center pixel and neighborhood are valid
                patch_mask = valid_mask[i-half_patch:i+half_patch+1,
                                      j-half_patch:j+half_patch+1]
                if np.all(patch_mask) and gt[i,j] > 0:  # Exclude background class
                    patch = data[i-half_patch:i+half_patch+1,
                               j-half_patch:j+half_patch+1, :]
                    patches.append(patch)
                    labels.append(gt[i,j])
        
        return np.array(patches), np.array(labels)