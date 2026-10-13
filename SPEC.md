# Glassbox: shared spec

A tiny character-level GPT, trained from scratch in PyTorch (`ml/`), exported to a flat binary, and run in
the browser by an inference engine written from scratch in TypeScript (`web/src/engine/`), with no ML
libraries. The UI (`web/src/`) makes the model transparent: live attention heads and next-character
probabilities while it writes.

## Ownership (parallel agents: touch only your own paths)

| Path | Owner |
|---|---|
| `ml/` | training agent |
| `web/public/model/` | training agent writes the exported artifacts here |
| `web/src/engine/` (except `types.ts`, which is frozen) | engine agent |
| everything else in `web/` | UI agent |
| `SPEC.md`, `web/src/engine/types.ts` | frozen; ask the coordinator to change them |

## Architecture (GPT-2 style, pre-LayerNorm)

- Character vocabulary `vocab: string[]`; token id = index into it.
- `x = wte[id] + wpe[pos]`
- For each layer `i`:
  - `a = LN(x; h.i.ln_1)`; `qkv = a @ h.i.attn.c_attn.weight + h.i.attn.c_attn.bias` (split into q, k, v, each `n_embd`; heads are contiguous slices of `head_dim = n_embd / n_head`)
  - causal softmax attention per head, scale `1/sqrt(head_dim)`; concat heads; `x += y @ h.i.attn.c_proj.weight + bias`
  - `m = LN(x; h.i.ln_2)`; `x += GELU(m @ h.i.mlp.c_fc.weight + b) @ h.i.mlp.c_proj.weight + b`
- `x = LN(x; ln_f)`; `logits = x @ wte^T` (weight tying, no separate lm_head)
- LayerNorm eps `1e-5` (biased variance). GELU is the tanh approximation:
  `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`. No dropout at inference.

## Weight file format

`web/public/model/model.json`:

```json
{
  "vocab": ["\n", " ", "!", "..."],
  "n_layer": 4, "n_head": 4, "n_embd": 192, "block_size": 128,
  "dtype": "f16",
  "params": [{ "name": "wte", "shape": [65, 192], "offset": 0 }, "..."]
}
```

`web/public/model/model.bin`: the tensors concatenated, little-endian, as `dtype`: `"f32"` (float32) or
`"f16"` (IEEE binary16; the shipped model, half the download). The engine widens f16 to float32 on load, which
is exact. `offset` is in **elements** (not bytes) from the start of the file. Every matrix is row-major. **Linear weights are stored `[in, out]`**
(so `y = x @ W`), i.e. the transpose of PyTorch's `nn.Linear.weight`.

Tensor names and shapes (`C = n_embd`, `V = vocab.length`, `T = block_size`):

| name | shape |
|---|---|
| `wte` | `[V, C]` |
| `wpe` | `[T, C]` |
| `h.{i}.ln_1.weight`, `h.{i}.ln_1.bias` | `[C]` |
| `h.{i}.attn.c_attn.weight` / `.bias` | `[C, 3C]` / `[3C]` (q, then k, then v along the output axis) |
| `h.{i}.attn.c_proj.weight` / `.bias` | `[C, C]` / `[C]` |
| `h.{i}.ln_2.weight`, `h.{i}.ln_2.bias` | `[C]` |
| `h.{i}.mlp.c_fc.weight` / `.bias` | `[C, 4C]` / `[4C]` |
| `h.{i}.mlp.c_proj.weight` / `.bias` | `[4C, C]` / `[C]` |
| `ln_f.weight`, `ln_f.bias` | `[C]` |

## Parity fixture

`web/public/model/reference.json`, produced by the PyTorch model in eval mode from the exported weights
(for f16: the checkpoint with every weight rounded to float16, i.e. exactly what the engine computes with):

```json
{
  "input_ids": [12, 40, 3, "..."],
  "logits_last": ["V floats: logits at the last position"],
  "attn_last_layer_head0": ["T*T floats, row-major, causal (zeros above the diagonal) softmax weights of the last layer, head 0, for input_ids"]
}
```

The engine must match `logits_last` and `attn_last_layer_head0` to within `1e-4` absolute.

## Training report

`web/public/model/training.json`:

```json
{
  "corpus": { "name": "...", "source_url": "...", "license": "...", "chars": 1115394 },
  "params": 1800000,
  "device": "mps", "train_seconds": 540,
  "steps": [0, 100, "..."], "train_loss": [4.2, "..."], "val_loss": [4.2, "..."],
  "final": { "train_loss": 1.3, "val_loss": 1.45 },
  "samples": ["a few unconditional samples from the final model"]
}
```
