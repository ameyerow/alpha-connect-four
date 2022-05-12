import torch
import numpy as np
from torch import nn


class RandomModel(nn.Module):

    def __init__(self, action_space_shape: int):
        super().__init__()
        self.n: int = action_space_shape

    def forward(self, X):
        return torch.tensor(np.reshape(np.ones(self.n) / self.n, (1, self.n))), 0
