import torch
import numpy as np
from torch import nn

device = torch.device("cuda:0")

class LongConnectFourModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.channels = 64
        self.conv = ConvBlock(self.channels)
        self.r1 = ResidualBlock(self.channels)
        self.r2 = ResidualBlock(self.channels)
        self.r3 = ResidualBlock(self.channels)
        self.r4 = ResidualBlock(self.channels)
        self.r5 = ResidualBlock(self.channels)
        self.r6 = ResidualBlock(self.channels)
        self.ph = PolicyHead(self.channels)
        self.vh = ValueHead(self.channels)

    def forward(self, X):
        X = self.conv.forward(X)
        for residual in [self.r1, self.r2, self.r3, self.r4, self.r5, self.r6]:
            X = residual.forward(X)
        policy = self.ph.forward(X)
        value = self.vh.forward(X)
        return policy, value


class ConvBlock(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same').to(device)
        self.bn = torch.nn.BatchNorm2d(out_channels).to(device)
        self.relu = torch.nn.ReLU().to(device)

    def forward(self, X):
        X = np.reshape(X, (-1, 1, 6, 7))
        X = torch.from_numpy(X).float().to(device)
        X = self.conv(X)
        X = self.bn(X)
        X = self.relu(X)
        return X


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding='same').to(device)
        self.bn = torch.nn.BatchNorm2d(channels).to(device)
        self.relu = torch.nn.ReLU().to(device)
        self.conv2 = torch.nn.Conv2d(channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding='same').to(device)

    def forward(self, X):
        outputs = self.conv1(X)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        outputs = self.bn(outputs)
        outputs = torch.add(X, outputs)
        outputs = self.relu(outputs)
        return outputs


class PolicyHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels=2, kernel_size=(1, 1), stride=(1, 1)).to(device)
        self.bn = torch.nn.BatchNorm2d(2).to(device)
        self.relu = torch.nn.ReLU().to(device)
        self.flatten = torch.nn.Flatten().to(device)
        self.linear = torch.nn.Linear(84, 7).to(device)
        self.softmax = torch.nn.Softmax(dim=1).to(device)

    def forward(self, X):
        X = self.conv(X)
        X = self.bn(X)
        X = self.relu(X)
        X = self.flatten(X)
        X = self.linear(X)
        X = self.softmax(X)
        return X


class ValueHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels=1, kernel_size=(1, 1), stride=(1, 1)).to(device)
        self.bn = torch.nn.BatchNorm2d(1).to(device)
        self.relu = torch.nn.ReLU().to(device)
        self.flatten = torch.nn.Flatten().to(device)
        self.linear1 = torch.nn.Linear(42, 256).to(device)
        self.linear2 = torch.nn.Linear(256, 1).to(device)
        self.tanh = torch.nn.Tanh().to(device)

    def forward(self, X):
        X = self.conv(X)
        X = self.bn(X)
        X = self.relu(X)
        X = self.flatten(X)
        X = self.linear1(X)
        X = self.relu(X)
        X = self.linear2(X)
        X = self.tanh(X)
        return X
