import math
from dataclasses import dataclass
import torch.nn as nn
from .embedding import GPTEmbedding
from .transformer import Block

@dataclass
class GPTConfig:
    context_size = 1024
    vocab_size = 50257
    n_layer = 12
    n_head = 12
    d_model = 768
    p_dropout = 0.1

    d_k = d_v = d_model // n_head # from https://arxiv.org/pdf/1706.03762.pdf

class GPTLanguageModel(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.embedding = GPTEmbedding(config)
    self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

    self.f_ln = nn.LayerNorm(config.d_model)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    
    self.embedding.token_embed.weight = self.lm_head.weight

    self._init_weights()

    # overwrite c_proj's weight initialization
    for p_name, p in self.named_parameters():
      if p_name.endswith('c_proj.weight'):
        nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*config.n_layer)) # from GPT-2 paper

    self.config = config

  def _init_weights(self):
    for child in self.children():
      if(isinstance(child, nn.Linear)):
        nn.init.normal_(child.weight, mean=0.0, std=0.02) # std=0.02 is from GPT paper
        if(child.bias is not None):
          nn.init.zeros_(child.bias) # common practice
      elif(isinstance(child, nn.Embedding)):
        nn.init.normal_(child.weight, mean=0.0, std=0.02)

  def summary(self):
    print(self.state_dict)
    print(f'(n_params={sum(p.numel() for p in self.parameters())/1e6:.2f}M)')

  def forward(self, sequence):
    x = self.embedding(sequence)

    for block in self.blocks:
      x = block(x)

    x = self.f_ln(x)
    x = self.lm_head(x)
    return x