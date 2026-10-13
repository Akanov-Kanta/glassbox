# ml/: training the Glassbox model

A 1.8M-parameter character-level GPT, trained from scratch in PyTorch on Kazakh classical literature,
then exported to a flat float16 file that the browser engine (`web/src/engine/`) runs without any ML library.
The file format and architecture contract live in [`../SPEC.md`](../SPEC.md).

## Corpus

Works of **Abai Kunanbayuly** (Абай Құнанбайұлы, 1845–1904): his poems and long poems.
To reach a usable size they are combined with other public-domain Kazakh classics from the
Kazakh section of Wikisource (which lives on the multilingual wikisource.org, not kk.wikisource.org):

| Author | died | chars |
|---|---|---|
| Abai Kunanbayuly | 1904 | 137,322 |
| Akhmet Baitursynuly (fables) | 1937 | 104,964 |
| Ybyrai Altynsarin (stories, poems) | 1889 | 67,882 |
| Shortanbai Kanaiuly | 1881 | 9,491 |
| Magzhan Zhumabayev | 1938 | 8,341 |
| **total** (276 documents) | | **328,545** |

All authors died before 1954, so the texts are in the public domain (Kazakhstan: life + 70 years).
Abai's prose *Book of Words* (Қара сөздер) is not transcribed on Wikisource, so it is not included.

`data.py` pulls the raw wikitext through the MediaWiki API, strips markup (templates, `<ref>`
footnotes, notes sections, tables, links, headers, author bylines), fixes Latin homoglyphs typed
inside Cyrillic words, drops digits and Latin-script junk, skips author index pages and one
vandalised page, and normalises dashes, quotes and whitespace. Characters seen fewer than 5 times
are removed, which leaves a **vocabulary of 86 symbols** (Kazakh Cyrillic, punctuation, space and newline).
Documents are shuffled with a fixed seed, so the 90/10 train/val split mixes authors.

## Model

GPT-2 style, pre-LayerNorm, tied input/output embeddings, tanh GELU, LayerNorm eps 1e-5, biases everywhere
(`model.py`, written by hand). Attention is computed explicitly rather than with the fused kernel,
so the weights can be inspected and exported.

| n_layer | n_head | n_embd | block_size | vocab | params |
|---|---|---|---|---|---|
| 4 | 4 | 192 | 128 | 86 | 1,820,928 |

## Training

AdamW (betas 0.9/0.99, weight decay 0.1 on matrices), peak LR 2e-3 with 200 warmup steps, cosine decay to 1e-4,
batch 64 × 128 characters, dropout 0.3, gradient clipping at 1.0, and 4,500 steps.
The run took **648 s (10.8 min) on an Apple M4 GPU (MPS)**, including evaluation.
Train and val loss are estimated every 100 steps, and the checkpoint with the best val loss is kept.

| | train loss | val loss |
|---|---|---|
| best checkpoint (step 3900) | 1.416 | **1.711** |

A first run (dropout 0.2, 6,000-step schedule) reached val 1.719 at step 2,400 and then overfit while
the learning rate was still high. It was stopped and replaced by the run above, which has stronger dropout
and a shorter schedule. With only ~300k characters the model memorises quickly, so regularisation matters
more than scale here.

## Reproduce

```bash
/opt/homebrew/bin/python3.13 -m venv ml/.venv
ml/.venv/bin/pip install -r ml/requirements.txt
cd ml
.venv/bin/python data.py                       # download + clean -> ml/data/corpus.txt
.venv/bin/python train.py                      # -> ml/runs/best.pt, log.json
.venv/bin/python export.py                     # -> web/public/model/*
.venv/bin/python verify_export.py              # NumPy-only parity check
```

## Exported files (`web/public/model/`)

`export.py` rounds every weight to float16 (`DTYPE`), which halves the download from 7.3 MB to 3.6 MB.
The reference fixture is computed from the rounded weights, so the engine's 1e-4 parity check covers exactly
the model that ships. Rounding does not move the loss over the whole validation split: 1.71698 nats/char for
the float32 checkpoint and 1.71698 for the float16 export. (That full-split number is a little above the
1.711 that training reports, which is an estimate over random batches.)

| file | size | contents |
|---|---|---|
| `model.bin` | 3.6 MB | 1,820,928 little-endian float16, Linear weights stored `[in, out]` |
| `model.json` | 5.6 KB | vocab, config, tensor table (name, shape, offset in floats) |
| `reference.json` | 230 KB | parity fixture: 128 validation characters, last-position logits, last-layer head-0 attention |
| `training.json` | 4.5 KB | corpus info, loss curves, timing, 4 samples at temperature 0.8 |

`verify_export.py` reloads `model.bin` with NumPy alone, re-implements the forward pass from SPEC.md,
and checks it against `reference.json`. The maximum difference is 3.6e-15 on logits and 1.2e-15 on attention,
against a tolerance of 1e-4.

## Sample (temperature 0.8, unconditional)

```
Қалған лақтың баласы деп,
Малшылар таппады сымпылдап.
Жаның алып қоянып,
Қазақтың жайын жастарды көзі.
```

The sample has the shape of Kazakh verse (lines of 7–8 syllables, rhyme-like endings, dialogue with
em-dashes), but at this size most words are plausible inventions rather than real meaning.
