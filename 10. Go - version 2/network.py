import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random
from einops.layers.torch import Rearrange
from einops import rearrange


class ResidualBlock(nn.Module):
    def __init__(self, block_width):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = block_width, out_channels = block_width, kernel_size=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features = block_width)
        
    def forward(self, x):
        preds = self.conv(x)
        preds = F.leaky_relu(x)
        preds = self.batch_norm(x)
        return preds + x
    

class AlphaGoZeroBatch(nn.Module):
    def __init__(self, n_residual_blocks, block_width):
        super().__init__()
        self.stem = nn.Sequential(
            Rearrange('b w h -> b 1 w h'),
            nn.Conv2d(in_channels=1, out_channels=block_width, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.res_blocks = [ResidualBlock(block_width) for _ in range(n_residual_blocks)]
        
        self.tower1 = nn.Sequential(
            nn.Linear(block_width*81,82)
        )
        
        self.tower2 = nn.Sequential(
            nn.Linear(block_width*81,1),
            nn.Tanh()
        ) 


    def forward(self, x, legal_moves):            
        illegal = lambda legal: [move not in legal for move in range(82)]
        mask = torch.stack([torch.as_tensor(illegal(lm)) for lm in legal_moves])
        # remove option for pass, unless only move:
        mask[[len(lm) != 1 for lm in legal_moves], 81] = 1
        x = self.stem(x)
        for block in self.res_blocks:
            x = block(x)
        x = rearrange(x, 'b block_width w h -> b (block_width w h)') 
        x1 = self.tower1(x)
        x1 = x1.masked_fill(mask, -torch.inf)
        x1 = F.softmax(x1, dim=-1)
        x2 = self.tower2(x)
        return x1, x2