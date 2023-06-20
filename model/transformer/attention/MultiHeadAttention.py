import torch
import torch.nn as nn

from .Head import Head

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.W0 = nn.Linear(config.n_head*config.d_v, config.d_model)
        self.dropout = nn.Dropout(config.p_dropout)

        self.config = config

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        out = self.dropout(self.W0(out))

        return out