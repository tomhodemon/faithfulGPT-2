import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.WK = nn.Linear(config.d_model, config.d_k, bias=False)
        self.WQ = nn.Linear(config.d_model, config.d_k, bias=False)
        self.WV = nn.Linear(config.d_model, config.d_v, bias=False)

        self.config = config

    def forward(self, x):

        K = self.WK(x)
        Q = self.WQ(x)

        scores = (Q @ K.transpose(-2, -1)) * (self.config.d_k**-0.5) 
        tril = torch.tril(torch.ones(self.config.context_size, self.config.context_size))
        scores = scores.masked_fill(tril == 0, float('-inf'))
        scores = F.softmax(scores, -1)

        V = self.WV(x)

        out = scores @ V

        return out

