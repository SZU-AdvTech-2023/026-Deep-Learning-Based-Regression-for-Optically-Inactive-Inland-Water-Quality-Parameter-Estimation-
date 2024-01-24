import torch
import torch.nn as nn
import acmix


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If input and output dimensions are not equal, use a 1x1 convolution
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

        self.se_block = SEBlock(out_channels)

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

        out = self.se_block(out)
        # Add residual and pass through ReLU
        out += identity
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        squeeze_output = self.squeeze(x)
        batch_size, channels, _, _ = squeeze_output.size()
        squeeze_output = squeeze_output.view(batch_size, channels)

        excitation_output = self.excitation(squeeze_output)
        excitation_output = excitation_output.view(batch_size, channels, 1, 1)

        weighted_x = x * excitation_output.expand_as(x)
        return weighted_x


class DnnrNet(nn.Module):
    def __init__(self, in_channels=216):
        super(DnnrNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.residual1 = ResidualBlock(64, 64, kernel_size=3)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.residual2 = ResidualBlock(128, 128, kernel_size=3)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.residual3 = ResidualBlock(256, 256, kernel_size=3)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 30 * 30, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(128, 1)
        self.acmix1 = acmix.WindowAttention_acmix(64, (4, 4), 8)
        self.acmix2 = acmix.WindowAttention_acmix(256,(2, 2), 4)

    def forward(self, x):
        # Convolutional layers with residuals
        x = self.conv1(x)  # 216*32*32->64*32*32
        x = x.permute(0, 2, 3, 1)  # 将维度从 BCHW 转换为 BHWC
        x = self.acmix1(x, 32, 32)
        x = x.permute(0, 3, 1, 2)  # 将维度从 BHWC 转换为 BCHW
        x = self.residual1(x)  # 64*32*32

        x = self.conv2(x)  # 64*32*32->128*31*31
        x = self.residual2(x)  # 128*31*31
        x = self.conv3(x)  # 128*31*31->256*30*30
        x = x.permute(0, 2, 3, 1)  # 将维度从 BCHW 转换为 BHWC
        x = self.acmix2(x, 30, 30)
        x = x.permute(0, 3, 1, 2)  # 将维度从 BHWC 转换为 BCHW
        x = self.residual3(x)  # 256*30*30

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


# 自定义 R² 的倒数作为损失函数
class R2ReciprocalLoss(nn.Module):
    def __init__(self):
        super(R2ReciprocalLoss, self).__init__()

    def forward(self, y_pred, y_true):
        SS_res = torch.sum((y_true - y_pred) ** 2)
        SS_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r_squared = 1 - SS_res / SS_tot
        r2_reciprocal = 1 / (r_squared + 1e-10)  # 避免分母为零，添加一个小的常数
        return r2_reciprocal
