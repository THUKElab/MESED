import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from einops import repeat

class Transformer(nn.Module):
    def __init__(self, dim, depth=6, head=12, mlp_dim=3072, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadedAttention(dim, head = head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, **kwargs):
        x = self.fn(x, **kwargs)
        x = self.norm(x)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, dim, head=12, dropout=0.):
        super().__init__()
        assert dim % head == 0

        self.d_k = dim // head
        self.head = head

        self.linear_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.output_linear = nn.Linear(dim, dim)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, qkv, mask=None):
        batch_size = qkv.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (qkv, qkv, qkv))]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        return self.output_linear(x)


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=0.):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        
        if mask is not None:
            mask = repeat(mask, 'b l -> b h c l', h=scores.shape[1], c=scores.shape[2])
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
