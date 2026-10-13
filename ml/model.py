"""A small GPT-2 style character-level transformer, written out by hand.

Architecture follows SPEC.md exactly: learned token + position embeddings, pre-LayerNorm
blocks (LN -> causal self-attention -> residual, LN -> MLP(GELU tanh) -> residual),
final LayerNorm, and logits computed with the token embedding matrix (weight tying).
All Linear layers and LayerNorms have biases; LayerNorm eps = 1e-5.

Module names are chosen so state_dict keys equal the tensor names in the export
format (wte, wpe, h.{i}.ln_1.weight, h.{i}.attn.c_attn.weight, ..., ln_f.bias).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        # Output columns are [q | k | v], each n_embd wide; head h is the slice
        # [h*head_dim, (h+1)*head_dim) inside each of them.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, T, C) -> (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Explicit attention (not the fused kernel) so the weights can be inspected.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        probs = att
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return (y, probs) if return_attn else (y, None)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.mlp = MLP(cfg)

    def forward(self, x, return_attn: bool = False):
        a, att = self.attn(self.ln_1(x), return_attn)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, att


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        # No lm_head: logits = ln_f(x) @ wte.weight^T (weight tying).
        self.apply(self._init)
        # GPT-2 scaled init for residual projections.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_attn: bool = False):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        attns = []
        for block in self.h:
            x, att = block(x, return_attn)
            attns.append(att)
        x = self.ln_f(x)
        logits = x @ self.wte.weight.t()
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_attn:
            return logits, loss, attns
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            nxt = torch.multinomial(probs.cpu(), 1, generator=generator).to(idx.device)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
