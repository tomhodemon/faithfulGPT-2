import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .FFN import FFN

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
      super().__init__()
      assert config.d_model % config.n_head == 0
      # key, query, value projections for all heads, but in a batch
      self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
      # output projection
      self.c_proj = nn.Linear(config.d_model, config.d_model)
      # regularization
      self.attn_dropout = nn.Dropout(config.p_dropout)
      self.resid_dropout = nn.Dropout(config.p_dropout)

      # causal mask to ensure that attention is only applied to the left in the input sequence
      self.register_buffer("bias", torch.tril(torch.ones(config.context_size, config.context_size)).view(1, 1, config.context_size, config.context_size))
      self.n_head = config.n_head
      self.d_model = config.d_model

    def forward(self, x):
      B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_model)

      # calculate query, key, values for all heads in batch and move head forward to be the batch dim
      q, k ,v  = self.c_attn(x).split(self.d_model, dim=2)
      k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
      q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
      v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

      # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
      att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
      att = F.softmax(att, dim=-1)
      att = self.attn_dropout(att)
      y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
      y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

      # output projection
      y = self.resid_dropout(self.c_proj(y))
      return y

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.ln1 = nn.LayerNorm(config.d_model)
    self.attn = MultiHeadAttention(config)
    self.ln2 = nn.LayerNorm(config.d_model)
    self.FFN = FFN(config)
    
  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.FFN(self.ln2(x))
    return x