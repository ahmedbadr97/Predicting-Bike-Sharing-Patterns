import torch.nn as nn
from torch import sigmoid
import torch.nn.functional as F


class BikeSharingModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = sigmoid(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = sigmoid(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = F.relu(x)

        return x
