import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class CNN3DBlock(nn.Module):
    """3D CNN block with ReLU activation"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int]):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))

class CNN2DBlock(nn.Module):
    """2D CNN block with ReLU activation"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))

class HSSNB(nn.Module):
    """
    Hybrid Spectral-Spatial Network with Bi-LSTM (HSSNB)
    
    Architecture from the paper:
    "Hybrid CNN Bi-LSTM neural network for Hyperspectral image classification"
    """
    def __init__(self, num_classes: int = 16):
        super().__init__()
        
        # 3D CNN path
        self.conv3d_1 = CNN3DBlock(1, 8, kernel_size=(7, 3, 3))
        self.conv3d_2 = CNN3DBlock(8, 16, kernel_size=(5, 3, 3))
        self.conv3d_3 = CNN3DBlock(16, 32, kernel_size=(3, 3, 3))
        
        # 2D CNN path
        # After 3D CNN, the spatial dimensions are reduced and channels are reshaped
        self.conv2d_1 = CNN2DBlock(576, 64)  # 576 = 32 * 18 (from last 3D conv)
        self.conv2d_2 = CNN2DBlock(64, 128)
        
        # Bi-LSTM path
        self.bilstm_1 = nn.LSTM(
            input_size=1920,  # 128 * 15 (from last 2D conv)
            hidden_size=64,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.25)
        
        self.bilstm_2 = nn.LSTM(
            input_size=128,  # 64*2 (bidirectional)
            hidden_size=64,
            bidirectional=True,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Linear(128, num_classes)  # 128 = 64*2 (bidirectional)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            x: Input tensor of shape (batch_size, 1, spectral_bands, height, width)
               where spectral_bands=30, height=width=25
        
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # 3D CNN path: extract spectral-spatial features
        x = self.conv3d_1(x)       # -> (batch, 8, 24, 23, 23)
        x = self.conv3d_2(x)       # -> (batch, 16, 20, 21, 21)
        x = self.conv3d_3(x)       # -> (batch, 32, 18, 19, 19)
        
        # Reshape for 2D CNN
        # Combine spectral and channel dimensions
        x = x.view(batch_size, -1, x.size(-2), x.size(-1))  # -> (batch, 576, 19, 19)
        
        # 2D CNN path: further spatial feature extraction
        x = self.conv2d_1(x)       # -> (batch, 64, 17, 17)
        x = self.conv2d_2(x)       # -> (batch, 128, 15, 15)
        
        # Reshape for Bi-LSTM
        x = x.view(batch_size, 15, -1)  # -> (batch, 15, 1920)
        
        # First Bi-LSTM
        x, _ = self.bilstm_1(x)    # -> (batch, 15, 128)
        x = self.dropout(x)
        
        # Second Bi-LSTM
        x, _ = self.bilstm_2(x)    # -> (batch, 15, 128)
        
        # Take the last output
        x = x[:, -1, :]           # -> (batch, 128)
        
        # Classification
        x = self.classifier(x)     # -> (batch, num_classes)
        
        return x
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)