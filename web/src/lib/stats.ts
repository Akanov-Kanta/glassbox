/** Map a probability to 0..1 on a log scale: p <= 1% -> 0, p = 10% -> 0.5, p = 100% -> 1. */
export function confidence(p: number): number {
  if (!(p > 0)) return 0;
  return Math.min(1, Math.max(0, (Math.log10(p) + 2) / 2));
}

export function entropyBits(probs: Float32Array): number {
  let h = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (p > 0) h -= p * Math.log2(p);
  }
  return h;
}

/** Indices of the k largest values, descending. */
export function topK(values: Float32Array, k: number): number[] {
  const idx = Array.from(values.keys());
  idx.sort((a, b) => values[b] - values[a]);
  return idx.slice(0, k);
}

export function rankOf(values: Float32Array, i: number): number {
  let r = 1;
  const v = values[i];
  for (let j = 0; j < values.length; j++) if (values[j] > v) r++;
  return r;
}

export function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

export function maxOf(values: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < values.length; i++) if (values[i] > m) m = values[i];
  return m;
}

/** Mean over heads of one layer's attention rows. */
export function meanRows(rows: Float32Array[]): Float32Array {
  const n = rows[0]?.length ?? 0;
  const out = new Float32Array(n);
  for (const r of rows) for (let i = 0; i < n; i++) out[i] += r[i] / rows.length;
  return out;
}
