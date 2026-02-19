"""
Model Definition for AI Voice Deepfake Detection
This file contains the DeepfakeCNN model architecture.
"""

import torch
import torch.nn as nn


class DeepfakeCNN(nn.Module):
    """
    1D CNN model for binary classification.
    
    Architecture:
    - Conv1D (32 filters, kernel size 3, relu)
    - MaxPool1d
    - Conv1D (64 filters, kernel size 3, relu)
    - MaxPool1d
    - Flatten
    - Linear (64, relu)
    - Dropout (0.3)
    - Linear (1)
    """
    def __init__(self, input_features):
        super(DeepfakeCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        
        # Calculate the size after convolutions and pooling
        # After pool1: features // 2
        # After pool2: features // 4
        conv_output_size = (input_features // 4) * 64
        
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # x shape: (batch, features, 1) -> need (batch, 1, features) for Conv1d
        x = x.permute(0, 2, 1)
        
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
