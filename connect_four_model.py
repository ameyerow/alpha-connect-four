import torch
from torch import nn
import numpy as np


class ConnectFourModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.softmax = torch.nn.Softmax(dim=1)
        self.tanh = torch.nn.Tanh()
        self.pi1 = torch.nn.Conv2d(1, out_channels=42, kernel_size=(3, 3), stride=(1, 1))
        self.pi2 = torch.nn.Conv2d(42, out_channels=84, kernel_size=(3, 3), stride=(1, 1))
        self.pi3 = torch.nn.Linear(504, 64)
        self.pi4 = torch.nn.Linear(64, 7)
        self.values1 = torch.nn.Conv2d(1, out_channels=42, kernel_size=(3, 3), stride=(1, 1))
        self.values2 = torch.nn.Conv2d(42, out_channels=84, kernel_size=(3, 3), stride=(1, 1))
        self.values3 = torch.nn.Linear(504, 64)
        self.values4 = torch.nn.Linear(64, 1)

    def forward(self, X):
        X = np.reshape(X, (-1, 1, 6, 7))
        X = torch.from_numpy(X).float()
        pi_outputs = self.pi1(X)
        pi_outputs = self.relu(pi_outputs)
        pi_outputs = self.pi2(pi_outputs)
        pi_outputs = self.relu(pi_outputs)
        pi_outputs = self.flatten(pi_outputs)
        pi_outputs = self.pi3(pi_outputs)
        pi_outputs = self.pi4(pi_outputs)
        pi_outputs = self.softmax(pi_outputs)

        values_outputs = self.values1(X)
        values_outputs = self.relu(values_outputs)
        values_outputs = self.values2(values_outputs)
        values_outputs = self.relu(values_outputs)
        values_outputs = self.flatten(values_outputs)
        values_outputs = self.values3(values_outputs)
        values_outputs = self.relu(values_outputs)
        values_outputs = self.values4(values_outputs)
        values_outputs = self.tanh(values_outputs)
        return pi_outputs, values_outputs
