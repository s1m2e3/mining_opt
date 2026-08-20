import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MambaConfig, MambaModel


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


class BinaryOutput(NamedTuple):
    score: torch.Tensor    # (B, T, out_dim) real-valued, carries gradients
    binary: torch.Tensor   # same shape, values in {0,1} (or {+1,-1,0} for kind='haar')


class HaarBinaryHead(nn.Module):
    """
    Final layer that turns a real score into a per-token decision by comparing it
    against learned thresholds. The map is the Haar family of indicator bases:

      kind='step'  Heaviside                1[s >= b]                 -> {0,1}
      kind='box'   Haar scaling fn (phi)    1[a <= s < c]             -> {0,1}
      kind='haar'  Haar mother wavelet      +1 on [a,b), -1 on [b,c)  -> {+1,-1,0}

    The thresholds are the parameters; they relocate the breakpoints but never
    change the output values, so the map is piecewise constant and its true
    derivative is 0 almost everywhere. `mode` selects what backward sees:

      'hard'  exact step. No gradient at all -- the score upstream gets nothing.
              Use when the binaries are handed to a MIP / Benders master.
      'ste'   straight-through: hard values forward, sigmoid slope backward.
      'soft'  sigmoid((s - b) / tau) both ways. Differentiable everywhere, but
              only binary in the limit tau -> 0; anneal `tau` down during training.

    Widths are stored as softplus pre-activations so a < b < c holds by
    construction and never has to be enforced by a constraint.
    """

    def __init__(
        self,
        out_dim: int = 1,
        kind: str = "step",
        mode: str = "hard",
        tau: float = 0.1,
        threshold_init: float = 0.0,
        width_init: float = 1.0,
    ):
        super().__init__()
        if kind not in ("step", "box", "haar"):
            raise ValueError("kind must be 'step', 'box', or 'haar'.")
        if mode not in ("hard", "ste", "soft"):
            raise ValueError("mode must be 'hard', 'ste', or 'soft'.")
        if tau <= 0:
            raise ValueError("tau must be positive.")

        self.kind = kind
        self.mode = mode
        # buffer, not a constant, so it can be annealed in the training loop
        self.register_buffer("tau", torch.tensor(float(tau)))

        # leftmost breakpoint; the only threshold when kind='step'
        self.a = nn.Parameter(torch.full((out_dim,), float(threshold_init)))

        inv_softplus = math.log(math.expm1(width_init))
        if kind in ("box", "haar"):
            self.log_w1 = nn.Parameter(torch.full((out_dim,), inv_softplus))
        if kind == "haar":
            self.log_w2 = nn.Parameter(torch.full((out_dim,), inv_softplus))

    def edges(self, threshold: torch.Tensor = None):
        """Breakpoint locations. `threshold` overrides the leftmost one, keeping widths."""
        a = self.a if threshold is None else threshold
        if self.kind == "step":
            return (a,)
        b = a + F.softplus(self.log_w1)
        if self.kind == "box":
            return (a, b)
        return (a, b, b + F.softplus(self.log_w2))

    def _heaviside(self, z: torch.Tensor) -> torch.Tensor:
        hard = (z >= 0).to(z.dtype)
        if self.mode == "hard":
            return hard
        soft = torch.sigmoid(z / self.tau)
        if self.mode == "soft":
            return soft
        return soft + (hard - soft).detach()

    def forward(self, score: torch.Tensor, threshold: torch.Tensor = None) -> torch.Tensor:
        e = self.edges(threshold)
        if self.kind == "step":
            return self._heaviside(score - e[0])
        if self.kind == "box":
            return self._heaviside(score - e[0]) - self._heaviside(score - e[1])
        return (
            self._heaviside(score - e[0])
            - 2.0 * self._heaviside(score - e[1])
            + self._heaviside(score - e[2])
        )


class SimpleTransformer(nn.Module):
    """
    A minimal Transformer encoder:
      input:  (B, T, in_dim)
      output: (B, T, out_dim) by default (per-token), or (B, out_dim) if pool='mean'/'cls'.

    With binary_head=True the forward returns a BinaryOutput(score, binary): the
    encoder emits a real score per token and HaarBinaryHead thresholds it into a
    per-token 0/1 decision. Intended for pool=None, where each token is one
    candidate and each binary is that candidate's include/exclude flag.
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
        binary_head: bool = False,   # append HaarBinaryHead; changes the return type
        binary_kind: str = "step",
        binary_mode: str = "hard",
        binary_tau: float = 0.1,
        threshold_init: float = 0.0,
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

        self.binary = (
            HaarBinaryHead(
                out_dim=out_dim,
                kind=binary_kind,
                mode=binary_mode,
                tau=binary_tau,
                threshold_init=threshold_init,
            )
            if binary_head
            else None
        )

    def forward(
        self,
        x: torch.Tensor,                      # (B, T, in_dim)
        src_key_padding_mask: torch.Tensor = None,  # (B, T) with True at PAD positions
        attn_mask: torch.Tensor = None,             # (T, T) or (B*nhead, T, T)
        pool: str = None,                            # None | 'mean' | 'cls'
        threshold: torch.Tensor = None,              # overrides the learned threshold
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

        if self.binary is not None:
            return BinaryOutput(score=y, binary=self.binary(y, threshold=threshold))
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



class MiningMamba(nn.Module):
    """
    Uses Hugging Face's MambaModel as the backbone, fed via inputs_embeds.
    Input:  x  shape (B, L, in_dim)
    Output: y  shape (B, L, out_dim)
    """
    def __init__(self, in_dim: int, d_model: int, num_layers: int, out_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)

        # HF MambaModel expects a config; vocab_size is required even if you only use inputs_embeds.
        config = MambaConfig(
            hidden_size=d_model,
            n_layer=num_layers,
            vocab_size=1,                # dummy; unused when passing inputs_embeds
            tie_word_embeddings=False,   # avoids tying to embeddings you aren't using
            # You can also set d_state, d_conv, expand, etc. via config if desired.
        )
        self.backbone = MambaModel(config)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # x: (B, L, in_dim)
        h0 = self.in_proj(x)  # (B, L, d_model)

        out = self.backbone(
            inputs_embeds=h0,           # (B, L, d_model)
            attention_mask=attention_mask,
            use_cache=False,            # usually off for training
            return_dict=True,
        )
        h = out.last_hidden_state      # (B, L, d_model) :contentReference[oaicite:1]{index=1}
        h = self.norm(h)
        y = self.head(h)               # (B, L, out_dim)
        return y
    

import math
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewSparseMHA_3a(nn.Module):
    """
    Multi-view sparse multi-head attention (Option 3a):

      - Shared Q projection across all views/heads.
      - K,V projections are:
          * shared across "shared views"
          * specialized per view for "specialized views"
      - Each view uses its own neighbor indices idx[v] of shape (B, L, K_v),
        so we never build an LxL attention matrix.

    Inputs:
      x:    (B, L, d_model)
      idxs: list/tuple length V; idxs[v] is LongTensor (B, L, K_v), neighbor indices in [0, L)

    Output:
      y:    (B, L, d_model)
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_heads: int,
        n_views: int,
        d_out:int,
        heads_per_view: Optional[Sequence[int]] = None,
        specialized_views: Optional[Iterable[int]] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")


        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.n_views = n_views
        self.dropout = dropout
        self.in_proj = nn.Linear(d_in, d_model, bias=bias)
        # ---- Heads allocation across views ----
        if heads_per_view is None:
            # Even split (last view gets the remainder)
            base = n_heads // n_views
            rem = n_heads % n_views
            heads_per_view = [base + (1 if v < rem else 0) for v in range(n_views)]
        if len(heads_per_view) != n_views:
            raise ValueError("heads_per_view must have length n_views.")
        if sum(heads_per_view) != n_heads:
            raise ValueError("sum(heads_per_view) must equal n_heads.")

        self.heads_per_view = list(heads_per_view)

        # Map view -> head indices slice [start, end)
        self.view_head_slices: List[Tuple[int, int]] = []
        s = 0
        for hv in self.heads_per_view:
            self.view_head_slices.append((s, s + hv))
            s += hv

        # ---- Specialized vs shared views ----
        spec: Set[int] = set(specialized_views) if specialized_views is not None else set()
        if any((v < 0 or v >= n_views) for v in spec):
            raise ValueError("specialized_views contains invalid view index.")
        self.specialized_views = spec
        self.shared_views = [v for v in range(n_views) if v not in self.specialized_views]

        # ---- Projections ----
        # Shared Q for all heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)

        # Shared K,V that can serve any head group (we'll slice per view)
        # Only needed if there exists at least one shared view.
        self.k_proj_shared = nn.Linear(d_model, d_model, bias=bias) if len(self.shared_views) > 0 else None
        self.v_proj_shared = nn.Linear(d_model, d_model, bias=bias) if len(self.shared_views) > 0 else None

        # Specialized K,V per view (output dimension = heads_in_view * dh)
        self.k_proj_spec = nn.ModuleDict()
        self.v_proj_spec = nn.ModuleDict()
        for v in self.specialized_views:
            hs, he = self.view_head_slices[v]
            hv = he - hs
            out_dim = hv * self.dh
            self.k_proj_spec[str(v)] = nn.Linear(d_model, out_dim, bias=bias)
            self.v_proj_spec[str(v)] = nn.Linear(d_model, out_dim, bias=bias)

        self.out_proj = nn.Linear(d_model, d_out, bias=bias)

    @staticmethod
    def _gather_kv(
        tensor_bhld: torch.Tensor,  # (B, H, L, dh)
        idx_blk: torch.Tensor,      # (B, L, K)
    ) -> torch.Tensor:
        """
        Gathers tensor along the sequence dimension using idx.

        Returns: (B, H, L, K, dh) where out[b,h,i,k,:] = tensor[b,h, idx[b,i,k], :].
        """
        B, H, L, dh = tensor_bhld.shape
        if idx_blk.dtype != torch.long:
            raise TypeError("idx must be a LongTensor.")
        if idx_blk.shape[0] != B or idx_blk.shape[1] != L:
            raise ValueError(f"idx shape must be (B,L,K). Got {tuple(idx_blk.shape)} for B={B}, L={L}.")

        K = idx_blk.shape[2]
        # Expand indices to (B, H, L, K, dh) so gather works
        idx = idx_blk.unsqueeze(1).unsqueeze(-1).expand(B, H, L, K, dh)  # (B,H,L,K,dh)
        # Expand source to (B, H, L, K, dh) but gather needs the source dim=2 to be indexable.
        src = tensor_bhld.unsqueeze(3).expand(B, H, L, K, dh)            # (B,H,L,K,dh)
        # We want to gather along dim=2 (the L dimension). However src currently has L at dim=2,
        # and we want index into that dimension. For gather, src and idx must have same shape.
        out = torch.gather(src, dim=2, index=idx)  # (B,H,L,K,dh)
        return out

    def forward(
        self,
        x: torch.Tensor,
        idxs: Sequence[torch.Tensor],
        masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        x:     (B, L, d_model)
        idxs:  length V; idxs[v] is (B, L, K_v) neighbor indices
        masks: optional, length V; masks[v] is a (B, L, K_v) bool tensor, True
               where idxs[v] points at a genuine neighbour. Slots that are False
               are excluded from the softmax rather than competing in it.

               Without a mask, callers with ragged neighbourhoods have to pad
               the index rows with something, and whatever they pick takes real
               attention mass -- padding a 2-ancestor row out to K=9 with the
               node's own index gives self seven duplicate logits that no
               learned weight can separate from the genuine one. Masking is the
               only way to make a short row mean "few neighbours" instead of
               "mostly myself".

               Rows with no valid slot at all (a surface block has no ancestors)
               contribute zero rather than NaN: an all -inf softmax is undefined,
               so the scores are neutralised first and the output zeroed after.
        """
        if len(idxs) != self.n_views:
            raise ValueError(f"Expected idxs length {self.n_views}, got {len(idxs)}.")
        if masks is not None and len(masks) != self.n_views:
            raise ValueError(f"Expected masks length {self.n_views}, got {len(masks)}.")

        x = self.in_proj(x)
        B, L, d = x.shape
        if d != self.d_model:
            raise ValueError(f"Expected x last dim {self.d_model}, got {d}.")

        # Shared Q across all heads: (B, H, L, dh)
        q = self.q_proj(x).view(B, L, self.n_heads, self.dh).transpose(1, 2)

        # Shared K,V for shared views (if any): (B, H, L, dh)
        if self.k_proj_shared is not None:
            k_shared = self.k_proj_shared(x).view(B, L, self.n_heads, self.dh).transpose(1, 2)
            v_shared = self.v_proj_shared(x).view(B, L, self.n_heads, self.dh).transpose(1, 2)
        else:
            k_shared, v_shared = None, None

        # Collect per-view outputs into the right head slots
        out_heads = torch.zeros((B, self.n_heads, L, self.dh), device=x.device, dtype=x.dtype)

        for v in range(self.n_views):
            hs, he = self.view_head_slices[v]
            hv = he - hs
            idx = idxs[v]  # (B, L, K_v)

            qv = q[:, hs:he, :, :]  # (B, hv, L, dh)

            if v in self.specialized_views:
                kv = self.k_proj_spec[str(v)](x).view(B, L, hv, self.dh).transpose(1, 2)  # (B,hv,L,dh)
                vv = self.v_proj_spec[str(v)](x).view(B, L, hv, self.dh).transpose(1, 2)  # (B,hv,L,dh)
            else:
                if k_shared is None:
                    raise RuntimeError("No shared K/V projections available but a view is marked shared.")
                kv = k_shared[:, hs:he, :, :]  # (B,hv,L,dh)
                vv = v_shared[:, hs:he, :, :]  # (B,hv,L,dh)

            # Gather neighbors: (B, hv, L, K_v, dh)
            k_sel = self._gather_kv(kv, idx)
            v_sel = self._gather_kv(vv, idx)

            # Sparse attention weights: (B, hv, L, K_v)
            scores = (qv.unsqueeze(3) * k_sel).sum(-1) / math.sqrt(self.dh)

            mv = None if masks is None else masks[v]
            if mv is not None:
                m = mv.unsqueeze(1)                                # (B,1,L,K_v)
                valid = m.any(dim=-1, keepdim=True)                # (B,1,L,1)
                scores = scores.masked_fill(~m, float("-inf"))
                # neutralise fully-invalid rows so the softmax stays finite;
                # their output is zeroed below, so the uniform weights they get
                # here never reach the residual stream
                scores = torch.where(valid, scores, torch.zeros_like(scores))

            attn = F.softmax(scores, dim=-1)
            if mv is not None:
                attn = torch.where(valid, attn, torch.zeros_like(attn))
            attn = F.dropout(attn, p=self.dropout, training=self.training)

            # Weighted sum: (B, hv, L, dh)
            ov = (attn.unsqueeze(-1) * v_sel).sum(3)

            out_heads[:, hs:he, :, :] = ov

        # Merge heads: (B, L, d_model)
        out = out_heads.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)