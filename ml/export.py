"""Export the best checkpoint to the browser format described in SPEC.md.

Writes to web/public/model/:
  model.json      vocab, config, and the tensor table (name, shape, offset in floats)
  model.bin       little-endian float32 tensors, concatenated; Linear weights stored [in, out]
  reference.json  parity fixture (logits at the last position + last-layer head-0 attention)
  training.json   training report (corpus, loss curves, timing, samples)

  python ml/export.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
DATA = ROOT / "data"
OUT = ROOT.parent / "web" / "public" / "model"

SAMPLE_SEED = 1904
N_SAMPLES = 4
SAMPLE_LEN = 200
TEMPERATURE = 0.8


def tensor_table(model: GPT) -> list[tuple[str, np.ndarray]]:
    """Tensors in SPEC.md order, as float32 numpy arrays in the export layout."""
    sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    out = [("wte", sd["wte.weight"]), ("wpe", sd["wpe.weight"])]
    for i in range(model.cfg.n_layer):
        p = f"h.{i}."
        out += [
            (p + "ln_1.weight", sd[p + "ln_1.weight"]),
            (p + "ln_1.bias", sd[p + "ln_1.bias"]),
            # nn.Linear stores [out, in]; the export format is [in, out] so y = x @ W.
            (p + "attn.c_attn.weight", sd[p + "attn.c_attn.weight"].T),
            (p + "attn.c_attn.bias", sd[p + "attn.c_attn.bias"]),
            (p + "attn.c_proj.weight", sd[p + "attn.c_proj.weight"].T),
            (p + "attn.c_proj.bias", sd[p + "attn.c_proj.bias"]),
            (p + "ln_2.weight", sd[p + "ln_2.weight"]),
            (p + "ln_2.bias", sd[p + "ln_2.bias"]),
            (p + "mlp.c_fc.weight", sd[p + "mlp.c_fc.weight"].T),
            (p + "mlp.c_fc.bias", sd[p + "mlp.c_fc.bias"]),
            (p + "mlp.c_proj.weight", sd[p + "mlp.c_proj.weight"].T),
            (p + "mlp.c_proj.bias", sd[p + "mlp.c_proj.bias"]),
        ]
    out += [("ln_f.weight", sd["ln_f.weight"]), ("ln_f.bias", sd["ln_f.bias"])]
    return [(n, np.ascontiguousarray(a, dtype="<f4")) for n, a in out]


def write_weights(model: GPT, vocab: list[str]) -> None:
    cfg = model.cfg
    params, chunks, offset = [], [], 0
    for name, arr in tensor_table(model):
        params.append({"name": name, "shape": list(arr.shape), "offset": offset})
        chunks.append(arr.ravel())
        offset += arr.size
    blob = np.concatenate(chunks).astype("<f4")
    assert blob.size == offset
    (OUT / "model.bin").write_bytes(blob.tobytes())
    meta = {
        "vocab": vocab,
        "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd, "block_size": cfg.block_size,
        "dtype": "f32",
        "params": params,
    }
    (OUT / "model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"model.bin: {blob.size:,} floats ({blob.nbytes / 1e6:.2f} MB), {len(params)} tensors")


def reload_from_export(cfg: GPTConfig) -> GPT:
    """Build a fresh model from model.bin, so the fixture comes from exactly the exported weights."""
    meta = json.loads((OUT / "model.json").read_text())
    blob = np.fromfile(OUT / "model.bin", dtype="<f4")
    model = GPT(cfg)
    sd = model.state_dict()
    for p in meta["params"]:
        a = blob[p["offset"]: p["offset"] + int(np.prod(p["shape"]))].reshape(p["shape"])
        key = p["name"] + (".weight" if p["name"] in ("wte", "wpe") else "")
        t = torch.from_numpy(a.copy())
        if key.endswith(".weight") and t.dim() == 2 and key not in ("wte.weight", "wpe.weight"):
            t = t.t()  # back to nn.Linear's [out, in]
        assert sd[key].shape == t.shape, (key, sd[key].shape, t.shape)
        sd[key] = t
    model.load_state_dict(sd)
    return model.eval()


def write_reference(model: GPT, vocab: list[str]) -> None:
    T = model.cfg.block_size
    text = (DATA / "corpus.txt").read_text()
    stoi = {c: i for i, c in enumerate(vocab)}
    val_start = int(0.9 * len(text))
    # A real validation snippet that starts at the beginning of a line.
    start = text.index("\n", val_start + 2000) + 1
    snippet = text[start:start + T]
    assert len(snippet) == T
    ids = [stoi[c] for c in snippet]
    m = model.double().eval()  # float64 forward over the float32 weights: the most exact reference
    with torch.no_grad():
        logits, _, attns = m(torch.tensor([ids]), return_attn=True)
    ref = {
        "text": snippet,
        "input_ids": ids,
        "logits_last": [float(x) for x in logits[0, -1]],
        "attn_last_layer_head0": [float(x) for x in attns[-1][0, 0].reshape(-1)],
    }
    (OUT / "reference.json").write_text(json.dumps(ref, ensure_ascii=False))
    model.float()
    print(f"reference.json: {T} tokens from validation text: {snippet[:60]!r}...")


def samples(model: GPT, vocab: list[str]) -> list[str]:
    stoi = {c: i for i, c in enumerate(vocab)}
    g = torch.Generator().manual_seed(SAMPLE_SEED)
    out = []
    model.eval()
    for _ in range(N_SAMPLES):
        idx = torch.tensor([[stoi["\n"]]])  # unconditional: start from a line break
        idx = model.generate(idx, SAMPLE_LEN, temperature=TEMPERATURE, generator=g)
        out.append("".join(vocab[i] for i in idx[0, 1:].tolist()))
    return out


def write_training_report(model: GPT, vocab: list[str], ckpt: dict) -> list[str]:
    log = json.loads((RUNS / "log.json").read_text())
    meta = json.loads((DATA / "meta.json").read_text())
    best = log["steps"].index(log["best_step"])
    smp = samples(model, vocab)
    report = {
        "corpus": {
            "name": meta["name"],
            "source_url": meta["source_url"],
            "source_urls": meta["source_urls"],
            "license": meta["license"],
            "chars": meta["chars"],
            "documents": meta["documents"],
            "chars_per_author": meta["chars_per_author"],
            "vocab_size": len(vocab),
        },
        "params": log["params"],
        "config": log["config"],
        "hparams": {k: log["hparams"][k] for k in
                    ("batch_size", "max_iters", "lr", "min_lr", "warmup", "weight_decay", "dropout", "grad_clip")},
        "device": log["device"],
        "train_seconds": round(log["train_seconds"]),
        "steps": log["steps"],
        "train_loss": log["train_loss"],
        "val_loss": log["val_loss"],
        "best_step": log["best_step"],
        "final": {"train_loss": log["train_loss"][best], "val_loss": log["val_loss"][best], "step": log["best_step"]},
        "sampling": {"temperature": TEMPERATURE, "length": SAMPLE_LEN, "prompt": "\n", "seed": SAMPLE_SEED},
        "samples": smp,
    }
    (OUT / "training.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"training.json: best step {log['best_step']}, train {report['final']['train_loss']}, val {report['final']['val_loss']}")
    return smp


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(RUNS / "best.pt", map_location="cpu")
    cfg = GPTConfig(**ckpt["config"])
    vocab = ckpt["vocab"]
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    write_weights(model, vocab)
    exported = reload_from_export(cfg)
    # Sanity: the exported weights reproduce the checkpoint's outputs bit-for-bit.
    x = torch.randint(len(vocab), (1, cfg.block_size), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(model(x)[0], exported(x)[0]), "export round-trip mismatch"
    write_reference(exported, vocab)
    for s in write_training_report(exported, vocab, ckpt):
        print("---\n" + s)


if __name__ == "__main__":
    main()
