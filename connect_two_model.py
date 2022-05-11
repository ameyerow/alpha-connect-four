import torch
import numpy as np
from torch import nn


class ConnectTwoModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.softmax = torch.nn.Softmax(dim=1)
        self.tanh = torch.nn.Tanh()

        self.pi1 = nn.Linear(4, 16)
        self.pi2 = nn.Linear(16, 4)
        self.values1 = nn.Linear(4, 16)
        self.values2 = nn.Linear(16, 1)

    def forward(self, X):
        X = np.reshape(X, (-1, 4))
        X = torch.from_numpy(X).float()
        X = self.flatten(X)

        pi_outputs = self.pi1(X)
        pi_outputs = self.relu(pi_outputs)
        pi_outputs = self.pi2(pi_outputs)
        pi_outputs = self.softmax(pi_outputs)

        values_outputs = self.values1(X)
        values_outputs = self.relu(values_outputs)
        values_outputs = self.values2(values_outputs)
        values_outputs = self.tanh(values_outputs)
        
        return pi_outputs, values_outputs
