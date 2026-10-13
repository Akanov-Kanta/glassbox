// Test helpers: a seeded PRNG and a builder for random models in the exported file format.

import type { ModelMeta, ParamEntry } from './format';

/** mulberry32: a tiny deterministic uniform [0, 1) generator. */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface RandomModelSpec {
  vocab: string[];
  nLayer: number;
  nHead: number;
  nEmbd: number;
  blockSize: number;
  seed?: number;
}

/**
 * A model.json-like header and model.bin-like buffer with random weights. Tensors are written in reverse
 * order after a few floats of padding, so a parser that ignores `offset` would be caught.
 */
export function randomModel(spec: RandomModelSpec): { meta: ModelMeta; buffer: ArrayBuffer } {
  const { vocab, nLayer, nEmbd: C, blockSize: T } = spec;
  const V = vocab.length;
  const rand = seededRandom(spec.seed ?? 1);
  const uniform = (scale: number) => (rand() * 2 - 1) * scale;

  // [name, shape, initialiser]
  const tensors: [string, number[], () => number][] = [
    ['wte', [V, C], () => uniform(0.5)],
    ['wpe', [T, C], () => uniform(0.3)],
  ];
  const w = 1 / Math.sqrt(C);
  for (let i = 0; i < nLayer; i++) {
    const h = `h.${i}`;
    tensors.push(
      [`${h}.ln_1.weight`, [C], () => 1 + uniform(0.2)],
      [`${h}.ln_1.bias`, [C], () => uniform(0.1)],
      [`${h}.attn.c_attn.weight`, [C, 3 * C], () => uniform(2.5 * w)],
      [`${h}.attn.c_attn.bias`, [3 * C], () => uniform(0.1)],
      [`${h}.attn.c_proj.weight`, [C, C], () => uniform(w)],
      [`${h}.attn.c_proj.bias`, [C], () => uniform(0.1)],
      [`${h}.ln_2.weight`, [C], () => 1 + uniform(0.2)],
      [`${h}.ln_2.bias`, [C], () => uniform(0.1)],
      [`${h}.mlp.c_fc.weight`, [C, 4 * C], () => uniform(w)],
      [`${h}.mlp.c_fc.bias`, [4 * C], () => uniform(0.1)],
      [`${h}.mlp.c_proj.weight`, [4 * C, C], () => uniform(w / 2)],
      [`${h}.mlp.c_proj.bias`, [C], () => uniform(0.1)],
    );
  }
  tensors.push(['ln_f.weight', [C], () => 1 + uniform(0.2)], ['ln_f.bias', [C], () => uniform(0.1)]);

  const padding = 3;
  const total = padding + tensors.reduce((sum, [, shape]) => sum + shape.reduce((a, b) => a * b, 1), 0);
  const data = new Float32Array(total);
  const params: ParamEntry[] = [];
  let offset = padding;
  for (const [name, shape, init] of [...tensors].reverse()) {
    const size = shape.reduce((a, b) => a * b, 1);
    for (let i = 0; i < size; i++) data[offset + i] = init();
    params.push({ name, shape, offset });
    offset += size;
  }

  const meta: ModelMeta = {
    vocab,
    n_layer: nLayer,
    n_head: spec.nHead,
    n_embd: C,
    block_size: T,
    dtype: 'f32',
    params,
  };
  return { meta, buffer: data.buffer };
}

/** `n` distinct printable characters, starting with a newline and a space. */
export function testVocab(n: number): string[] {
  const vocab = ['\n', ' '];
  for (let code = 33; vocab.length < n; code++) vocab.push(String.fromCharCode(code));
  return vocab;
}
