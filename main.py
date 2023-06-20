import torch
from model import GPTConfig, GPTLanguageModel
from transformers import GPT2Tokenizer

# TODO:
# 1. residual dropout
# 2. attention dropout
# 3. CausalSelfAttention from Scratchx

config = GPTConfig()
model = GPTLanguageModel(config)

input_ids = torch.randint(0, config.vocab_size, (2, config.context_size))

print( input_ids.shape )
print( input_ids )
out = model(input_ids)

print(out.shape)

# model.summary()