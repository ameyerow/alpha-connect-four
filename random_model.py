import torch
import numpy as np


class RandomModel:

    def __init__(self, action_space_shape: int):
        self.n: int = action_space_shape

    def forward(self, X):
        return torch.tensor(np.reshape(np.ones(self.n) / self.n, (1, self.n))), 0
