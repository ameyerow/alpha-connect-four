import torch
from torch import nn
import numpy as np


class ConnectFourModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.pi1 = torch.nn.Linear(42, 7)
        self.pi2 = torch.nn.Softmax()
        self.values1 = torch.nn.Linear(42, 1)
        self.values2 = torch.nn.Tanh()

    def forward(self, X):
        pi_outputs = self.pi1(X)
        pi_outputs = self.pi2(pi_outputs)
        values_outputs = self.values1(X)
        values_outputs = self.values(values_outputs)
        return pi_outputs, values_outputs
