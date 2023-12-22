from torch.nn.parameter import Parameter
import torch
from torch import nn
from einops.layers.torch import Rearrange


class LN_mid(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.beta = Parameter(torch.zeros([1, dim, 1, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class MAN(nn.Module):
    def __init__(self, seq, numberJ, residual=True):
        super(MAN, self).__init__()
        self.fc = nn.Linear(seq, seq)
        self.norm = LN_mid(numberJ)
        self.reset_parameters()
        self.residual = residual
        self.numberJ = numberJ

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x_ = self.fc(x)
        x_ = Rearrange('b (j c) n -> b j c n', j=self.numberJ)(x_)
        x_ = self.norm(x_)
        x_ = Rearrange('b j c n -> b (j c) n')(x_)
        if self.residual:
            x = x + x_
        else:
            x = x_
        return x


class Predictor(nn.Module):
    def __init__(self, num_layer, seq, dim, numberJ):
        super(Predictor, self).__init__()
        self.fc_in = nn.Linear(dim, dim)
        self.fc_out = nn.Linear(dim, dim)
        self.fc = nn.Sequential(*[MAN(seq, numberJ) for _ in range(num_layer)])
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1e-8)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, x):
        # x:[b,n,d]
        x_ = self.fc_in(x)
        x_ = Rearrange('b n d->b d n')(x_)
        x_ = self.fc(x_)
        x_ = Rearrange('b d n->b n d')(x_)
        x_ = self.fc_out(x_)
        return x_
