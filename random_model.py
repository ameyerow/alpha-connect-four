import torch
import numpy as np


class RandomModel:

    def forward(self, X):
        return torch.tensor(np.reshape(np.ones(7) / 7, (1, 7))), 0
