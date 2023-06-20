import torch.nn as nn

class PositionEmbedding(nn.Embedding):
  def __init__(self, context_size, d_model):
    super().__init__(context_size, d_model)