import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter


class WordEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size,
                 device='cpu', dtype=torch.float32):
        super().__init__()

        # Register parameters
        self.W_embed = Parameter(torch.randn(vocab_size, embed_size,
                                             device=device, dtype=dtype).div(math.sqrt(vocab_size)))

    def forward(self, x):
        out = self.W_embed[x]
        return out
