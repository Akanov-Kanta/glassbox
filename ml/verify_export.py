"""Independent check of the exported files, using NumPy only (no torch, no model.py).

Reads web/public/model/model.json + model.bin, re-implements the forward pass exactly
as SPEC.md describes it, and asserts that it reproduces reference.json to 1e-4. If this
passes, the binary layout (offsets in floats, [in, out] Linear weights, q|k|v column
order, weight tying) is what the TypeScript engine should expect.

  python ml/verify_export.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "web" / "public" / "model"
TOL = 1e-4


def load():
    meta = json.loads((OUT / "model.json").read_text())
    raw = (OUT / "model.bin").read_bytes()
    np_dtype = {"f32": "<f4", "f16": "<f2"}[meta["dtype"]]
    assert len(raw) % np.dtype(np_dtype).itemsize == 0
    blob = np.frombuffer(raw, dtype=np_dtype)
    W, expected_offset = {}, 0
    for p in meta["params"]:
        n = int(np.prod(p["shape"]))
        assert p["offset"] == expected_offset, f"{p['name']}: offset {p['offset']} != {expected_offset}"
        W[p["name"]] = blob[p["offset"]:p["offset"] + n].reshape(p["shape"]).astype(np.float64)
        expected_offset += n
    assert expected_offset == blob.size, f"file has {blob.size} values, table covers {expected_offset}"
    return meta, W


def check_shapes(meta, W):
    V, C, T, L = len(meta["vocab"]), meta["n_embd"], meta["block_size"], meta["n_layer"]
    want = {"wte": (V, C), "wpe": (T, C), "ln_f.weight": (C,), "ln_f.bias": (C,)}
    for i in range(L):
        p = f"h.{i}."
        want |= {
            p + "ln_1.weight": (C,), p + "ln_1.bias": (C,),
            p + "attn.c_attn.weight": (C, 3 * C), p + "attn.c_attn.bias": (3 * C,),
            p + "attn.c_proj.weight": (C, C), p + "attn.c_proj.bias": (C,),
            p + "ln_2.weight": (C,), p + "ln_2.bias": (C,),
            p + "mlp.c_fc.weight": (C, 4 * C), p + "mlp.c_fc.bias": (4 * C,),
            p + "mlp.c_proj.weight": (4 * C, C), p + "mlp.c_proj.bias": (C,),
        }
    assert set(W) == set(want), f"tensor names differ: {set(W) ^ set(want)}"
    for k, s in want.items():
        assert W[k].shape == s, f"{k}: {W[k].shape} != {s}"


def layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)  # biased variance
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def forward(meta, W, ids):
    C, H, L = meta["n_embd"], meta["n_head"], meta["n_layer"]
    hd = C // H
    T = len(ids)
    x = W["wte"][ids] + W["wpe"][:T]
    causal = np.tril(np.ones((T, T), dtype=bool))
    last_attn = None
    for i in range(L):
        p = f"h.{i}."
        a = layernorm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        qkv = a @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
        q, k, v = qkv[:, :C], qkv[:, C:2 * C], qkv[:, 2 * C:]
        y = np.zeros((T, C))
        for h in range(H):
            s = slice(h * hd, (h + 1) * hd)
            att = (q[:, s] @ k[:, s].T) / np.sqrt(hd)
            att = np.where(causal, att, -np.inf)
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            y[:, s] = att @ v[:, s]
            if i == L - 1 and h == 0:
                last_attn = att
        x = x + y @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        m = layernorm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        x = x + gelu(m @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"]) @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    x = layernorm(x, W["ln_f.weight"], W["ln_f.bias"])
    logits = x @ W["wte"].T  # weight tying
    return logits, last_attn


def main():
    meta, W = load()
    check_shapes(meta, W)
    ref = json.loads((OUT / "reference.json").read_text())
    ids = ref["input_ids"]
    T, V = meta["block_size"], len(meta["vocab"])
    assert len(ids) == T, f"input_ids has {len(ids)} tokens, block_size is {T}"
    assert len(ref["logits_last"]) == V and len(ref["attn_last_layer_head0"]) == T * T
    if "text" in ref:
        assert "".join(meta["vocab"][i] for i in ids) == ref["text"], "input_ids do not decode to ref text"

    logits, attn = forward(meta, W, ids)
    d_logits = np.abs(logits[-1] - np.array(ref["logits_last"])).max()
    d_attn = np.abs(attn.reshape(-1) - np.array(ref["attn_last_layer_head0"])).max()
    upper = np.array(ref["attn_last_layer_head0"]).reshape(T, T)[np.triu_indices(T, 1)]
    print(f"tensors: {len(meta['params'])}, floats: {sum(int(np.prod(p['shape'])) for p in meta['params']):,}")
    print(f"max |logits - ref| = {d_logits:.3e}")
    print(f"max |attn   - ref| = {d_attn:.3e}")
    print(f"ref attention above diagonal all zero: {bool(np.all(upper == 0))}")
    ok = d_logits < TOL and d_attn < TOL and np.all(upper == 0)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
