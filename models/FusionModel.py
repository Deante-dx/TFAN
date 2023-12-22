import torch
from torch import nn


class FusionModel(nn.Module):
    def __init__(self, seq, dim):
        super(FusionModel, self).__init__()
        self.fc = nn.Linear(2*dim, 2)
        self.act_last = nn.Sigmoid()

    def forward(self, b, p):
        bp = torch.cat([b, p], dim=-1)
        s = self.act_last(self.fc(bp))
        return s