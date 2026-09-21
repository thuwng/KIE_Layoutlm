import math
import torch
import torch.nn as nn

class RelationalGraphAttention(nn.Module):
    def __init__(self, hidden, n_heads, n_rel, dropout):
        super().__init__()
        self.h, self.d = n_heads, hidden // n_heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)
        self.rel_bias = nn.Embedding(n_rel, n_heads)
        nn.init.zeros_(self.rel_bias.weight)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x, rel, mask):
        B, S, H = x.shape
        sh = lambda t: t.view(B, S, self.h, self.d).transpose(1, 2)
        q, k, v = sh(self.q(x)), sh(self.k(x)), sh(self.v(x))

        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d)
        att = att + self.rel_bias(rel).permute(0, 3, 1, 2)

        no_edge = (rel == 0) | (~mask)[:, None, :]
        att = att.masked_fill(no_edge[:, None, :, :], float("-inf"))
        att = torch.nan_to_num(att.softmax(-1))

        out = (self.drop(att) @ v).transpose(1, 2).reshape(B, S, H)
        return self.ln(x + self.o(out))


class SpatialGraphEncoder(nn.Module):
    def __init__(self, hidden, n_layers=2, n_heads=4, n_rel=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            RelationalGraphAttention(hidden, n_heads, n_rel, dropout)
            for _ in range(n_layers)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.GELU(),
                nn.Linear(hidden * 2, hidden), nn.Dropout(dropout)
            )
            for _ in range(n_layers)
        ])
        self.ln = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

    def forward(self, x, rel, mask):
        for att, ffn, ln in zip(self.layers, self.ffn, self.ln):
            x = att(x, rel, mask)
            x = ln(x + ffn(x))
        return x * mask.unsqueeze(-1)