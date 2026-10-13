"""Train the character-level GPT on the cleaned corpus (ml/data/corpus.txt).

  python ml/train.py                      # full run (defaults below)
  python ml/train.py --max_iters 200      # quick smoke test

Writes ml/runs/best.pt (best validation loss checkpoint) and ml/runs/log.json
(loss curves, timing, device) which ml/export.py turns into training.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=192)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_iters", type=int, default=4500)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--eval_iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1845)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def load_data():
    text = (DATA / "corpus.txt").read_text()
    meta = json.loads((DATA / "meta.json").read_text())
    vocab = meta["vocab"]
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], vocab, meta


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"device: {device}")

    train_ids, val_ids, vocab, meta = load_data()
    print(f"corpus: {len(train_ids) + len(val_ids):,} chars ({len(train_ids):,} train / {len(val_ids):,} val), vocab {len(vocab)}")
    train_t = torch.from_numpy(train_ids)
    val_t = torch.from_numpy(val_ids)
    gen = torch.Generator().manual_seed(args.seed)

    def batch(split: str):
        data = train_t if split == "train" else val_t
        ix = torch.randint(len(data) - args.block_size - 1, (args.batch_size,), generator=gen)
        x = torch.stack([data[i:i + args.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + args.block_size] for i in ix])
        return x.to(device), y.to(device)

    cfg = GPTConfig(vocab_size=len(vocab), block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    print(f"params: {n_params:,}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.99))

    def lr_at(it: int) -> float:
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        r = (it - args.warmup) / max(1, args.max_iters - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(r, 1.0)))

    @torch.no_grad()
    def estimate():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                x, y = batch(split)
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    RUNS.mkdir(parents=True, exist_ok=True)
    log = {"steps": [], "train_loss": [], "val_loss": [], "lr": [], "elapsed": []}
    best_val = float("inf")
    best_step = -1
    t0 = time.time()

    model.train()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            if device == "mps":
                torch.mps.synchronize()
            ev = estimate()
            elapsed = time.time() - t0
            log["steps"].append(it)
            log["train_loss"].append(round(ev["train"], 4))
            log["val_loss"].append(round(ev["val"], 4))
            log["lr"].append(lr_at(it))
            log["elapsed"].append(round(elapsed, 1))
            mark = ""
            if ev["val"] < best_val:
                best_val, best_step = ev["val"], it
                torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "vocab": vocab,
                            "step": it, "val_loss": ev["val"], "train_loss": ev["train"]}, RUNS / "best.pt")
                mark = "  *best"
            print(f"step {it:5d} | train {ev['train']:.4f} | val {ev['val']:.4f} | lr {lr_at(it):.2e} | {elapsed:6.0f}s{mark}", flush=True)
        if it == args.max_iters:
            break

        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        x, y = batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

    if device == "mps":
        torch.mps.synchronize()
    total = time.time() - t0
    summary = {
        **log,
        "device": device,
        "torch": torch.__version__,
        "train_seconds": round(total, 1),
        "params": n_params,
        "config": cfg.to_dict(),
        "hparams": vars(args),
        "best_step": best_step,
        "best_val_loss": best_val,
        "train_chars": int(len(train_ids)),
        "val_chars": int(len(val_ids)),
    }
    (RUNS / "log.json").write_text(json.dumps(summary, indent=2))
    print(f"done in {total:.0f}s; best val {best_val:.4f} at step {best_step}")


if __name__ == "__main__":
    main()
