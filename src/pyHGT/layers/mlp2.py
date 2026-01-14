import torch
import torch.nn as nn


class MLP2(nn.Module):
    def __init__(self, n_h, out):
        super(MLP2, self).__init__()
        self.linear = nn.Linear(n_h, out)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        out_1 = self.linear(x)
        out2 = self.sigm(out_1)

        return out2