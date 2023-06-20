import torch
import torch.nn as nn
from .TokenEmbedding import TokenEmbedding
from .PositionEmbedding import PositionEmbedding

class GPTEmbedding(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.token_embed = TokenEmbedding(config.vocab_size, config.d_model)
    self.position_embed = PositionEmbedding(config.context_size, config.d_model)
    self.embedding_dropout = nn.Dropout(config.p_dropout)

    self.config = config

  def forward(self, sequence):
    device = sequence.device
    pos = torch.arange(sequence.size(1), device=device).unsqueeze(0)
    print(pos)

    x = self.token_embed(sequence)# + self.position_embed(sequence)

    return self.embedding_dropout(x)