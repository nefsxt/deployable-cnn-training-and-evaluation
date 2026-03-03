# Model Architecture (CNN)


import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    """
        A simple CNN to classify images from the CIFAR-10 dataset
        
    """
    
    def __init__(self):
        super().__init__()

        # -------------------------
        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=3, # Input channels: 3  (RGB)
            out_channels=32, # Output channels: 32 (number of learned filters)
            kernel_size=3, # Kernel size: 3x3 
            padding=1 # Padding: 1 (maintains spatial size)
        )
        # Output shape: (batch_size, 32, 32, 32)
        # -------------------------

        
        # -------------------------
        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=32, # Input channels: 32 (from previous layer)
            out_channels=64, # Output channels: 64 (more complex features)
            kernel_size=3, # Kernel size: 3x3 
            padding=1 # Padding: 1 (maintains spatial size)
        )
        # Output shape:(batch_size, 64, 16, 16) after pooling
        # -------------------------

        
        # -------------------------
        # Fully connected layer
        #
        # After two rounds of pooling: 32x32 → 16x16 → 8x8
        #
        # Feature map size: 64 channels × 8 × 8 = 4096 features
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        # -------------------------

        # -------------------------
        # Final classification layer
        # Maps to the 10 CIFAR-10 classes
        self.fc2 = nn.Linear(256, 10)
        # -------------------------

        
    def forward(self, x):
        """
        Defines the forward pass of the network.

        x shape: (batch_size, 3, 32, 32)
        """
        
        # (1) Apply first convolution
        x = self.conv1(x)

        # (2) Apply nonlinearity via ReLU
        x = F.relu(x)

        # (3) Max pooling (2x2): 32x32 -> 16x16
        x = F.max_pool2d(x, kernel_size=2)

        # (4) Apply second convolution
        x = self.conv2(x)
        
        # (5) Apply nonlinearity via ReLU 
        x = F.relu(x)

        # (6) Max pooling (2x2): 16x16 -> 8x8
        x = F.max_pool2d(x, kernel_size=2)

        # (7) Flatten -> Convert from (B, C, H, W) to (B, C*H*W)
        x = x.view(x.size(0), -1) 

        # (8) Fully connected layer 
        x = self.fc1(x)
        x = F.relu(x)

        # Output logits (no softmax here because CrossEntropyLoss applies softmax internally)
        x = self.fc2(x)

        return x
