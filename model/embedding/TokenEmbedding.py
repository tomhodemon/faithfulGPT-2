import torch.nn as nn

class TokenEmbedding(nn.Embedding):
  def __init__(self, vocab_size, d_model):
    super().__init__(vocab_size, d_model)