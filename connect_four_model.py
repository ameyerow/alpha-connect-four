import torch
from torch import nn


class ConnectFourModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.pi1 = torch.nn.Linear(42, 7)
        self.pi2 = torch.nn.Softmax()
        self.values1 = torch.nn.Linear(42, 1)
        self.values2 = torch.nn.Tanh()
        self.pi = None
        self.values = None

    def forward(self, X):
        pi_outputs = self.pi1(X)
        self.pi = self.pi2(pi_outputs)
        values_outputs = self.values1(X)
        self.values = self.values(values_outputs)

