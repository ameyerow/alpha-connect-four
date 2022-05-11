import torch
import numpy as np
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BigConnectFourModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.softmax = torch.nn.Softmax(dim=1)
        self.tanh = torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.bn2 = torch.nn.BatchNorm2d(33)
        self.bn3 = torch.nn.BatchNorm2d(97)
        self.bn4 = torch.nn.BatchNorm1d(4138)
        self.pi1 = torch.nn.Conv2d(1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.pi2 = torch.nn.Conv2d(33, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.pi3 = torch.nn.Linear(4074, 64)
        self.pi4 = torch.nn.Linear(4138, 7)
        self.values1 = torch.nn.Conv2d(1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.values2 = torch.nn.Conv2d(33, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.values3 = torch.nn.Linear(4074, 64)
        self.values4 = torch.nn.Linear(4138, 1)

    def forward(self, X):
        X = np.reshape(X, (-1, 1, 6, 7))
        X = torch.from_numpy(X).float().to(device)
        X = self.bn1(X)

        pi_outputs = torch.cat((self.pi1(X), X), 1)
        pi_outputs = self.bn2(pi_outputs)
        pi_outputs = self.relu(pi_outputs)
        pi_outputs = torch.cat((self.pi2(pi_outputs), pi_outputs), 1)
        pi_outputs = self.bn3(pi_outputs)
        pi_outputs = self.relu(pi_outputs)
        pi_outputs = self.flatten(pi_outputs)
        pi_outputs = torch.cat((self.pi3(pi_outputs), pi_outputs), 1)
        pi_outputs = self.bn4(pi_outputs)
        pi_outputs = self.relu(pi_outputs)
        pi_outputs = self.pi4(pi_outputs)
        pi_outputs = self.softmax(pi_outputs)

        values_outputs = torch.cat((self.values1(X), X), 1)
        values_outputs = self.bn2(values_outputs)
        values_outputs = self.relu(values_outputs)
        values_outputs = torch.cat((self.values2(values_outputs), values_outputs), 1)
        values_outputs = self.bn3(values_outputs)
        values_outputs = self.relu(values_outputs)
        values_outputs = self.flatten(values_outputs)
        values_outputs = torch.cat((self.values3(values_outputs), values_outputs), 1)
        values_outputs = self.bn4(values_outputs)
        values_outputs = self.relu(values_outputs)
        values_outputs = self.values4(values_outputs)
        values_outputs = self.tanh(values_outputs)
        return pi_outputs, values_outputs
