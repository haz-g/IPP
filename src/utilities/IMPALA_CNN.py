import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A simple residual block. 
    - It applies two convolutional layers (with ReLU activations), and, 
    - Adds the input (or a transformed version of it) back into the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=padding)
        # If the number of input channels differs from the output channels,
        # we transform the input so that we can add it.
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                                      kernel_size=1,
                                      stride=stride)
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        return F.relu(out)

class ImpalaCNN(nn.Module):
    """
    A simplified IMPALA-style CNN.
    - Input is expected to have shape (batch_size, channels, height, width).
    - The network processes the input with convolutional layers and residual blocks.
    - The final fully connected layer produces a feature vector (latent representation).
    """
    def __init__(self, in_channels=5, feature_dim=128):
        super(ImpalaCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16,
                               kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(16, 16)
        
        self.conv2 = nn.Conv2d(16, 32,
                               kernel_size=3, stride=1, padding=1)
        self.res2 = ResidualBlock(32, 32)
        
        self.fc = nn.Linear(32 * 5 * 5, feature_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x