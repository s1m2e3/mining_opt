import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                    # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)                                  # (1, T, D)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    """
    A minimal Transformer encoder:
      input:  (B, T, in_dim)
      output: (B, T, 256) by default (per-token), or (B, 256) if pool='mean'/'cls'.
    """
    def __init__(
        self,
        in_dim: int,
        d_model: int = 256,          # embedding dim (argument)
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        out_dim: int = 1,          # fixed output projection to 256 (can override if desired)
        use_posenc: bool = True,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.posenc = PositionalEncoding(d_model, dropout=dropout) if use_posenc else None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, out_dim)

        # optional CLS token if you want 'cls' pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,                      # (B, T, in_dim)
        src_key_padding_mask: torch.Tensor = None,  # (B, T) with True at PAD positions
        attn_mask: torch.Tensor = None,             # (T, T) or (B*nhead, T, T)
        pool: str = None,                            # None | 'mean' | 'cls'
    ):
        B, T, _ = x.shape
        h = self.in_proj(x)                 # (B, T, d_model)

        if pool == "cls":
            cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
            h = torch.cat([cls, h], dim=1)             # prepend CLS
            if src_key_padding_mask is not None:
                # pad mask needs a leading False for CLS
                cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=src_key_padding_mask.device)
                src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        if self.posenc is not None:
            h = self.posenc(h)                  # add positions
        h = self.encoder(
            h,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )                                   # (B, T or T+1, d_model)

        if pool is None:
            y = self.out_proj(h)            # (B, T, 256)
        elif pool == "mean":
            if src_key_padding_mask is None:
                y = h.mean(dim=1)           # (B, d_model)
            else:
                # masked mean over valid tokens
                valid = (~src_key_padding_mask).float()           # (B, T)
                denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                y = (h * valid.unsqueeze(-1)).sum(dim=1) / denom  # (B, d_model)
            y = self.out_proj(y)            # (B, 256)
        elif pool == "cls":
            y = self.out_proj(h[:, 0])      # (B, 256) — project CLS
        else:
            raise ValueError("pool must be None, 'mean', or 'cls'.")

        return y
    
def sinkhorn(logits, iters=8, tau=0.5, eps=1e-12):
    """
    logits: (B, T, T) unnormalized scores
    returns: P ~ doubly-stochastic (B, T, T)
    """
    # softmax across columns for numerical stability
    P = torch.softmax(logits / tau, dim=-1)  # row-normalized to start
    for _ in range(iters):
        P = P / (P.sum(dim=-1, keepdim=True) + eps)   # rows -> 1
        P = P / (P.sum(dim=-2, keepdim=True) + eps)   # cols -> 1
    return P

import torch

def sinkhorn_rectangular(
    logits, iters=8, tau=0.5, eps=1e-12,
    row_sum=1.0, col_sum=1.5, row_marg=None, col_marg=None
):
    """
    logits: (B, m, n) or (m, n)
    You can specify either scalars row_sum/col_sum (uniform marginals) or
    full vectors row_marg (m,) and col_marg (n,) that sum to the same total.
    """
    single = (logits.dim() == 2)
    if single:
        logits = logits.unsqueeze(0)
    B, m, n = logits.shape

    # positive init
    A = torch.softmax(logits / tau, dim=-1)

    if row_marg is None:
        row_marg = torch.full((m,), float(row_sum), device=logits.device)
    if col_marg is None:
        # default: make totals match if only row_sum is given
        total = row_marg.sum()
        col_marg = torch.full((n,), (total / n), device=logits.device)
    # normalize to same total mass
    row_marg = row_marg / row_marg.sum()
    col_marg = col_marg / col_marg.sum()
    total_mass = 1.0
    row_marg = row_marg * total_mass
    col_marg = col_marg * total_mass

    r = row_marg.view(1, m, 1)
    c = col_marg.view(1, 1, n)

    for _ in range(iters):
        A = A / (A.sum(dim=-1, keepdim=True) + eps) * r    # rows → row_marg
        A = A / (A.sum(dim=-2, keepdim=True) + eps) * c    # cols → col_marg

    return A.squeeze(0) if single else A
