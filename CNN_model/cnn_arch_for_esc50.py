import torch 
import torch.nn as nn
import torch.nn.functional as f 


class ESC50_CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout2d(0.3)
        self.dropout2d = nn.Dropout2d(0.3)

        # conv blocks
        # block 1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(16)

        # block 2
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)

        # block 3
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)

        # fully connected 
        self.fc1 = nn.Linear(
            in_features= 64,
            out_features=256
        )
        self.fc2 = nn.Linear(
            in_features=256,
            out_features=50
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = self.dropout2d(x)
        
        # block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = self.dropout2d(x)

        # blcok 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = self.dropout2d(x)

        x = f.adaptive_avg_pool2d(x, (1, 1))
        # Flatten
        x = torch.flatten(x, 1)

        # FC
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
