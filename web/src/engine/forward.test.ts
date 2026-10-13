import { describe, expect, it } from 'vitest';
import { createModel } from './index';
import type { ModelMeta } from './index';
import { randomModel, testVocab } from './testUtils';

const SPEC = { vocab: testVocab(11), nLayer: 2, nHead: 2, nEmbd: 16, blockSize: 8 };

// ---------------------------------------------------------------------------------------------------------
// An independent reference implementation, written literally from SPEC.md with plain number[][] matrices
// and float64 arithmetic. Deliberately unoptimised: it shares no code with the engine.
// ---------------------------------------------------------------------------------------------------------

type Matrix = number[][];

function tensor(meta: ModelMeta, data: Float32Array, name: string): number[] {
  const p = meta.params.find((q) => q.name === name)!;
  const size = p.shape.reduce((a, b) => a * b, 1);
  return Array.from(data.subarray(p.offset, p.offset + size));
}

function matrix(flat: number[], rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, (_, i) => flat.slice(i * cols, (i + 1) * cols));
}

function matmul(a: Matrix, b: Matrix): Matrix {
  return a.map((row) => b[0].map((_, j) => row.reduce((s, x, k) => s + x * b[k][j], 0)));
}

function addBias(a: Matrix, bias: number[]): Matrix {
  return a.map((row) => row.map((x, j) => x + bias[j]));
}

function layerNorm(a: Matrix, w: number[], b: number[]): Matrix {
  return a.map((row) => {
    const mean = row.reduce((s, x) => s + x, 0) / row.length;
    const variance = row.reduce((s, x) => s + (x - mean) ** 2, 0) / row.length;
    return row.map((x, j) => ((x - mean) / Math.sqrt(variance + 1e-5)) * w[j] + b[j]);
  });
}

function geluRef(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
}

function softmaxRef(row: number[]): number[] {
  const max = Math.max(...row);
  const exps = row.map((x) => Math.exp(x - max));
  const sum = exps.reduce((s, x) => s + x, 0);
  return exps.map((x) => x / sum);
}

function referenceForward(meta: ModelMeta, buffer: ArrayBuffer, ids: number[]) {
  const data = new Float32Array(buffer);
  const get = (name: string) => tensor(meta, data, name);
  const C = meta.n_embd;
  const V = meta.vocab.length;
  const H = meta.n_head;
  const hd = C / H;
  const T = ids.length;

  const wte = matrix(get('wte'), V, C);
  const wpe = matrix(get('wpe'), meta.block_size, C);
  let x: Matrix = ids.map((id, t) => wte[id].map((e, c) => e + wpe[t][c]));
  const attention: Matrix[][] = [];

  for (let i = 0; i < meta.n_layer; i++) {
    const p = (s: string) => get(`h.${i}.${s}`);
    const a = layerNorm(x, p('ln_1.weight'), p('ln_1.bias'));
    const qkv = addBias(matmul(a, matrix(p('attn.c_attn.weight'), C, 3 * C)), p('attn.c_attn.bias'));
    const y: Matrix = Array.from({ length: T }, () => new Array<number>(C).fill(0));
    attention.push([]);
    for (let h = 0; h < H; h++) {
      const q = qkv.map((row) => row.slice(h * hd, (h + 1) * hd));
      const k = qkv.map((row) => row.slice(C + h * hd, C + (h + 1) * hd));
      const v = qkv.map((row) => row.slice(2 * C + h * hd, 2 * C + (h + 1) * hd));
      const att: Matrix = [];
      for (let t = 0; t < T; t++) {
        const scores = k.map((kj, j) =>
          j <= t ? kj.reduce((s, kd, d) => s + kd * q[t][d], 0) / Math.sqrt(hd) : -Infinity,
        );
        att.push(softmaxRef(scores));
        for (let j = 0; j < T; j++) for (let d = 0; d < hd; d++) y[t][h * hd + d] += att[t][j] * v[j][d];
      }
      attention[i].push(att);
    }
    const attnOut = addBias(matmul(y, matrix(p('attn.c_proj.weight'), C, C)), p('attn.c_proj.bias'));
    x = x.map((row, t) => row.map((e, c) => e + attnOut[t][c]));

    const m = layerNorm(x, p('ln_2.weight'), p('ln_2.bias'));
    const hidden = addBias(matmul(m, matrix(p('mlp.c_fc.weight'), C, 4 * C)), p('mlp.c_fc.bias')).map((row) =>
      row.map(geluRef),
    );
    const mlpOut = addBias(matmul(hidden, matrix(p('mlp.c_proj.weight'), 4 * C, C)), p('mlp.c_proj.bias'));
    x = x.map((row, t) => row.map((e, c) => e + mlpOut[t][c]));
  }

  const final = layerNorm(x, get('ln_f.weight'), get('ln_f.bias'));
  const logits = wte.map((row) => row.reduce((s, e, c) => s + e * final[T - 1][c], 0));
  return { logits, attention };
}

// ---------------------------------------------------------------------------------------------------------

describe('forward', () => {
  const { meta, buffer } = randomModel(SPEC);
  const model = createModel(meta, buffer);
  const V = SPEC.vocab.length;

  it('exposes the config and counts every parameter', () => {
    expect(model.config).toEqual(SPEC);
    const C = SPEC.nEmbd;
    const perLayer = 2 * C + (3 * C * C + 3 * C) + (C * C + C) + 2 * C + (4 * C * C + 4 * C) + (4 * C * C + C);
    expect(model.paramCount).toBe(V * C + SPEC.blockSize * C + SPEC.nLayer * perLayer + 2 * C);
  });

  it('returns correctly shaped, normalised, causal outputs', () => {
    const ids = [3, 1, 4, 1, 5];
    const out = model.forward(ids);
    const T = ids.length;
    expect(out.t).toBe(T);
    expect(out.logits.length).toBe(V);
    expect(out.probs.length).toBe(V);
    expect(out.probs.reduce((s, p) => s + p, 0)).toBeCloseTo(1, 5);
    expect(out.attention.length).toBe(SPEC.nLayer);
    for (const layer of out.attention) {
      expect(layer.length).toBe(SPEC.nHead);
      for (const att of layer) {
        expect(att.length).toBe(T * T);
        for (let i = 0; i < T; i++) {
          const row = att.subarray(i * T, (i + 1) * T);
          expect(row.reduce((s, p) => s + p, 0)).toBeCloseTo(1, 5);
          for (let j = i + 1; j < T; j++) expect(row[j]).toBe(0);
        }
      }
    }
  });

  it('matches the naive reference implementation', () => {
    for (const ids of [[0], [7, 2, 9], [1, 2, 3, 4, 5, 6, 7, 8], [10, 0, 10, 0, 5, 5, 9, 1]]) {
      const ref = referenceForward(meta, buffer, ids);
      const out = model.forward(ids);
      for (let v = 0; v < V; v++) expect(Math.abs(out.logits[v] - ref.logits[v])).toBeLessThan(1e-5);
      const T = ids.length;
      for (let l = 0; l < SPEC.nLayer; l++) {
        for (let h = 0; h < SPEC.nHead; h++) {
          for (let i = 0; i < T; i++) {
            for (let j = 0; j < T; j++) {
              expect(Math.abs(out.attention[l][h][i * T + j] - ref.attention[l][h][i][j])).toBeLessThan(1e-5);
            }
          }
        }
      }
    }
  });

  it('matches the reference for odd sizes too (n_embd not a multiple of 4, 3 heads, 1 layer)', () => {
    const odd = randomModel({ vocab: testVocab(7), nLayer: 1, nHead: 3, nEmbd: 18, blockSize: 5, seed: 99 });
    const oddModel = createModel(odd.meta, odd.buffer);
    const ids = [6, 0, 3, 3, 1];
    const ref = referenceForward(odd.meta, odd.buffer, ids);
    const out = oddModel.forward(ids);
    ref.logits.forEach((x, v) => expect(Math.abs(out.logits[v] - x)).toBeLessThan(1e-5));
  });

  it('attention is not trivially uniform (the random model is a meaningful test)', () => {
    const att = model.forward([1, 2, 3, 4, 5, 6, 7, 8]).attention[0][0];
    const lastRow = Array.from(att.subarray(7 * 8));
    expect(Math.max(...lastRow) - Math.min(...lastRow)).toBeGreaterThan(0.05);
  });

  it('crops the input to the last blockSize tokens', () => {
    const long = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1];
    const out = model.forward(long);
    expect(out.t).toBe(SPEC.blockSize);
    expect(Array.from(out.logits)).toEqual(Array.from(model.forward(long.slice(-SPEC.blockSize)).logits));
  });

  it('rejects empty and out-of-range input', () => {
    expect(() => model.forward([])).toThrow(/at least one token/);
    expect(() => model.forward([V])).toThrow(/out of range/);
    expect(() => model.forward([1.5])).toThrow(/out of range/);
  });
});

describe('loading', () => {
  const build = () => randomModel(SPEC);

  it('rejects a missing tensor', () => {
    const { meta, buffer } = build();
    meta.params = meta.params.filter((p) => p.name !== 'h.1.mlp.c_fc.bias');
    expect(() => createModel(meta, buffer)).toThrow(/missing tensor "h.1.mlp.c_fc.bias"/);
  });

  it('rejects a wrong shape', () => {
    const { meta, buffer } = build();
    meta.params.find((p) => p.name === 'wte')!.shape = [16, 11];
    expect(() => createModel(meta, buffer)).toThrow(/tensor "wte" has shape \[16,11\], expected \[11,16\]/);
  });

  it('rejects a tensor that runs past the end of the buffer', () => {
    const { meta, buffer } = build();
    expect(() => createModel(meta, buffer.slice(0, buffer.byteLength - 4))).toThrow(/does not fit in model.bin/);
  });

  it('rejects unexpected tensors, bad dtypes and inconsistent dimensions', () => {
    const a = build();
    a.meta.params.push({ name: 'lm_head.weight', shape: [1], offset: 0 });
    expect(() => createModel(a.meta, a.buffer)).toThrow(/unexpected tensors: lm_head.weight/);
    const b = build();
    b.meta.dtype = 'f16';
    expect(() => createModel(b.meta, b.buffer)).toThrow(/dtype/);
    const c = build();
    c.meta.n_head = 3;
    expect(() => createModel(c.meta, c.buffer)).toThrow(/divisible/);
  });
});

describe('tokenizer', () => {
  const { meta, buffer } = randomModel({ ...SPEC, vocab: ['\n', ' ', '.', "'", 'a', 'b', 'e', 'E', 'h', 'l', 'o'] });
  const model = createModel(meta, buffer);

  it('round-trips known characters', () => {
    const text = "hello.\nbe 'a'";
    expect(model.decode(model.encode(text))).toBe(text);
  });

  it('maps unknown characters to the closest known text, else a space', () => {
    const decode = (s: string) => model.decode(model.encode(s));
    expect(decode('héllo')).toBe('hello'); // accent stripped
    expect(decode('’a’')).toBe("'a'"); // curly quotes
    expect(decode('…')).toBe('...'); // ellipsis decomposes
    expect(decode('H')).toBe('h'); // other case
    expect(decode('b\r\nb')).toBe('b\nb'); // CR dropped
    expect(decode('a\u{1F600}b')).toBe('a b'); // emoji: one code point, one space
    expect(decode('zq')).toBe('  ');
  });

  it('rejects out-of-range ids in decode', () => {
    expect(() => model.decode([99])).toThrow(RangeError);
  });
});
