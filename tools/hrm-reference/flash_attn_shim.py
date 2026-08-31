"""Minimal flash_attn stand-in for CPU/MPS.

flash-attn is a CUDA performance kernel, not a different algorithm:
its output is standard scaled dot-product attention. torch's own SDPA
computes the same function, so substituting it changes speed and
nothing else.

Layout is the only real difference. flash-attn takes [B, S, H, D];
SDPA takes [B, H, S, D]. Transpose in, transpose out.
"""
import torch.nn.functional as F


def flash_attn_func(q, k, v, causal=False, **_):
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal
    )
    return out.transpose(1, 2)
