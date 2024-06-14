from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

class ConvNet(nn.Module):
    def __init__(self, input_size: int = 2, output_size: int = 10):
        """Standard Convolutional Network layers for the MNIST dataset.

        Args:
            input_size (int): input size of the model.
            output_size (int): The number of output classes.

        """
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(3136, 1024)
        self.fc2 = torch.nn.Linear(1024, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

##################################################################
#                   TODO: Implement other models                 #
##################################################################





##################################################################