import torch
import torch.nn as nn


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If input and output dimensions are not equal, use a 1x1 convolution
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If dimensions change, apply 1x1 convolution on identity
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add residual and pass through ReLU
        out += identity
        out = self.relu(out)

        return out


class DnnrNet(nn.Module):
    def __init__(self, in_channels=216):
        super(DnnrNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.residual1 = ResidualBlock(64, 64, kernel_size=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.residual2 = ResidualBlock(128, 128, kernel_size=1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.residual3 = ResidualBlock(256, 256, kernel_size=1)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        # Convolutional layers with residuals
        x = self.conv1(x)  # 216*1*1->64*1*1
        x = self.residual1(x)  # 64*1*1
        x = self.conv2(x)  # 64*1*1->128*1*1
        x = self.residual2(x)  # 128*1*1
        x = self.conv3(x)  # 128*1*1->256*1*1
        x = self.residual3(x)  # 256*1*1

        # Flatten before fully connected layers
        x = torch.flatten(x, 1)

        # Dropout layer
        x = self.dropout(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x
