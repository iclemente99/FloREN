import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_h, out_h):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(n_h, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, out_h)
        #self.linear4 = nn.Linear(32, 2)
        self.silu = nn.SiLU()

    def forward(self, x):
        l_128 = self.linear1(x)
        g_128 = self.silu(l_128)
        l_64 = self.linear2(g_128)
        g_64 = self.silu(l_64)
        l_32 = self.linear3(g_64)
        g_32 = self.silu(l_32)
        classifier = self.linear4(g_32)

        return classifier, g_32, l_32, g_64, l_64, g_128, l_128
