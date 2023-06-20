import torch.nn as nn 

class FFN(nn.Module):
  def __init__(self, config):
    super().__init__()
 
    self.c_fc = nn.Linear(config.d_model, config.d_model*4)
    self.gelu = nn.GELU()
    self.c_proj = nn.Linear(config.d_model*4, config.d_model)
    self.dropout = nn.Dropout(config.p_dropout)

    self.config = config

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    x = self.dropout(x)
    return x