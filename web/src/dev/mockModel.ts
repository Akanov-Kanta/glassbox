/**
 * Development stand-in for the real engine. It implements the frozen `Model` interface with a
 * character n-gram "language model" built from a short embedded text, plus synthetic causal attention
 * patterns (previous character, start of word, same character, recency, anchors). It is only used
 * when the real engine or weights are unavailable, and the UI shows a "demo data" badge when it is.
 */
import type { ForwardResult, GenStep, Model, ModelConfig, SampleOptions } from '../engine/types';

export const MOCK_TEXT = `Қараңғы түнде тау қалғып,
Ұйқыға кетер балбырап.
Даланы жым-жырт дел-сал қып,
Түн басады салбырап.

Әсем ән мен сөз ғана көңілге жақын,
Өлең сөздің патшасы, сөз сарасы.
Ғылым таппай мақтанба, орын таппай баптанба.
Һәм білім, һәм ақыл, һәм жүрек керек.

The quick brown fox jumps over the lazy dog.
To be, or not to be, that is the question.
`;

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hash(a: number, b: number, c: number): number {
  let h = 2166136261 ^ a;
  h = Math.imul(h ^ b, 16777619);
  h = Math.imul(h ^ c, 16777619);
  h ^= h >>> 13;
  return ((h >>> 0) % 10000) / 10000;
}

function softmaxInPlace(a: Float32Array, scale = 1): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] * scale > max) max = a[i] * scale;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    a[i] = Math.exp(a[i] * scale - max);
    sum += a[i];
  }
  for (let i = 0; i < a.length; i++) a[i] /= sum;
  return a;
}

class MockModel implements Model {
  readonly config: ModelConfig;
  readonly paramCount: number;
  private readonly index = new Map<string, number>();
  private readonly uni: Float32Array;
  private readonly bi: Map<number, Float32Array>;
  private readonly tri: Map<string, Float32Array>;

  constructor() {
    const vocab = Array.from(new Set(Array.from(MOCK_TEXT))).sort();
    this.config = { vocab, nLayer: 4, nHead: 4, nEmbd: 192, blockSize: 128 };
    vocab.forEach((c, i) => this.index.set(c, i));
    const V = vocab.length;
    const C = this.config.nEmbd;
    const T = this.config.blockSize;
    const L = this.config.nLayer;
    this.paramCount =
      V * C + T * C + L * (2 * C + C * 3 * C + 3 * C + C * C + C + 2 * C + C * 4 * C + 4 * C + 4 * C * C + C) + 2 * C;

    const ids = this.encode(MOCK_TEXT);
    this.uni = new Float32Array(V).fill(0.05);
    this.bi = new Map();
    this.tri = new Map();
    for (let i = 0; i < ids.length; i++) {
      this.uni[ids[i]] += 1;
      if (i >= 1) {
        const row = this.bi.get(ids[i - 1]) ?? new Float32Array(V);
        row[ids[i]] += 1;
        this.bi.set(ids[i - 1], row);
      }
      if (i >= 2) {
        const key = `${ids[i - 2]},${ids[i - 1]}`;
        const row = this.tri.get(key) ?? new Float32Array(V);
        row[ids[i]] += 1;
        this.tri.set(key, row);
      }
    }
  }

  encode(text: string): number[] {
    const out: number[] = [];
    for (const c of text) {
      const id = this.index.get(c);
      if (id !== undefined) out.push(id);
    }
    return out;
  }

  decode(ids: number[]): string {
    return ids.map((i) => this.config.vocab[i] ?? '').join('');
  }

  /** Next-character distribution (temperature 1) for a context. */
  private nextProbs(ctx: number[]): Float32Array {
    const V = this.config.vocab.length;
    const logits = new Float32Array(V);
    const a = ctx[ctx.length - 2];
    const b = ctx[ctx.length - 1];
    const tri = a !== undefined && b !== undefined ? this.tri.get(`${a},${b}`) : undefined;
    const bi = b !== undefined ? this.bi.get(b) : undefined;
    let total = 0;
    for (let i = 0; i < V; i++) total += this.uni[i];
    for (let i = 0; i < V; i++) {
      const p = 0.03 * (this.uni[i] / total) + 0.27 * (bi ? bi[i] / (sumOf(bi) || 1) : 0) + 0.7 * (tri ? tri[i] / (sumOf(tri) || 1) : 0);
      logits[i] = Math.log(p + 1e-6);
    }
    return softmaxInPlace(logits);
  }

  /** Synthetic attention row for the query at position q (0-based) over positions 0..q. */
  private attentionRow(ids: number[], q: number, layer: number, head: number): Float32Array {
    const n = q + 1;
    const scores = new Float32Array(n);
    const pattern = (layer * this.config.nHead + head) % 6;
    const space = this.index.get(' ');
    const nl = this.index.get('\n');
    let wordStart = 0;
    for (let j = q; j >= 0; j--) {
      if (ids[j] === space || ids[j] === nl) {
        wordStart = j + 1;
        break;
      }
    }
    const sharp = 1 + layer * 0.6;
    for (let j = 0; j < n; j++) {
      const dist = q - j;
      let s = 0;
      switch (pattern) {
        case 0: // previous character
          s = dist === 1 ? 6 : dist === 0 ? 2.5 : -dist * 0.15;
          break;
        case 1: // start of the current word
          s = j === wordStart ? 5 : j > wordStart ? 1.5 : -1 - dist * 0.03;
          break;
        case 2: // same character as the query
          s = ids[j] === ids[q] ? 3.5 : -0.5 - dist * 0.02;
          break;
        case 3: // recency
          s = -dist * 0.35;
          break;
        case 4: // whitespace / newline anchors
          s = ids[j] === space || ids[j] === nl ? 3 - dist * 0.02 : -dist * 0.04;
          break;
        default: // first token as an attention sink
          s = j === 0 ? 4 : -dist * 0.08;
      }
      scores[j] = s * sharp + (hash(layer * 31 + head, j, ids[j]) - 0.5) * 1.2;
    }
    return softmaxInPlace(scores);
  }

  forward(ids: number[]): ForwardResult {
    const ctx = ids.slice(-this.config.blockSize);
    const T = ctx.length;
    const attention: Float32Array[][] = [];
    for (let l = 0; l < this.config.nLayer; l++) {
      const heads: Float32Array[] = [];
      for (let h = 0; h < this.config.nHead; h++) {
        const m = new Float32Array(T * T);
        for (let q = 0; q < T; q++) m.set(this.attentionRow(ctx, q, l, h), q * T);
        heads.push(m);
      }
      attention.push(heads);
    }
    const probs = this.nextProbs(ctx);
    const logits = probs.map((p) => Math.log(p));
    return { logits, probs, attention, t: T };
  }

  *generate(promptIds: number[], opts: SampleOptions): Generator<GenStep, void, void> {
    const random = opts.random ?? mulberry32((Date.now() ^ 0x9e3779b9) >>> 0);
    let ctx = promptIds.slice(-this.config.blockSize);
    if (ctx.length === 0) ctx = [this.index.get('\n') ?? 0];
    for (;;) {
      const base = this.nextProbs(ctx);
      const probs = applySampling(base, opts.temperature, opts.topK);
      let r = random();
      let id = probs.length - 1;
      for (let i = 0; i < probs.length; i++) {
        r -= probs[i];
        if (r <= 0) {
          id = i;
          break;
        }
      }
      const q = ctx.length - 1;
      const attention: Float32Array[][] = [];
      for (let l = 0; l < this.config.nLayer; l++) {
        const heads: Float32Array[] = [];
        for (let h = 0; h < this.config.nHead; h++) heads.push(this.attentionRow(ctx, q, l, h));
        attention.push(heads);
      }
      yield { id, char: this.config.vocab[id], probs, attention };
      ctx.push(id);
      if (ctx.length > this.config.blockSize) ctx = ctx.slice(-this.config.blockSize);
    }
  }
}

function sumOf(a: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
}

function applySampling(base: Float32Array, temperature: number, topK: number): Float32Array {
  const V = base.length;
  const logits = new Float32Array(V);
  for (let i = 0; i < V; i++) logits[i] = Math.log(base[i] + 1e-12) / Math.max(temperature, 1e-3);
  if (topK > 0 && topK < V) {
    const sorted = Array.from(logits).sort((x, y) => y - x);
    const cut = sorted[topK - 1];
    for (let i = 0; i < V; i++) if (logits[i] < cut) logits[i] = -Infinity;
  }
  return softmaxInPlace(logits);
}

export function createMockModel(): Model {
  return new MockModel();
}
