// Numerical kernels. Matrices are row-major Float32Arrays; callers pass the dimensions explicitly and own
// the output buffers, so the hot path allocates nothing. Sums are accumulated in JS numbers (float64)
// where a reduction would otherwise lose precision.

/**
 * out[n, N] = x[n, K] @ w[K, N] + b[N]
 *
 * Loop order i-k-j: the innermost loop walks rows of `w` and one row of `out` contiguously (no strided
 * reads), so for n = 1 (one token) it is a single sequential pass over `w`. The k loop is unrolled by 4
 * so each out[j] is loaded and stored once per four rows of `w` instead of once per row; in V8 that
 * roughly doubles throughput.
 */
export function linear(
  out: Float32Array,
  x: Float32Array,
  w: Float32Array,
  b: Float32Array,
  n: number,
  K: number,
  N: number,
): void {
  for (let i = 0; i < n; i++) {
    const o = i * N;
    const xi = i * K;
    out.set(b, o);
    let k = 0;
    for (; k + 3 < K; k += 4) {
      const a0 = x[xi + k];
      const a1 = x[xi + k + 1];
      const a2 = x[xi + k + 2];
      const a3 = x[xi + k + 3];
      const w0 = k * N;
      const w1 = w0 + N;
      const w2 = w1 + N;
      const w3 = w2 + N;
      for (let j = 0; j < N; j++) {
        out[o + j] += a0 * w[w0 + j] + a1 * w[w1 + j] + a2 * w[w2 + j] + a3 * w[w3 + j];
      }
    }
    for (; k < K; k++) {
      const a = x[xi + k];
      const wk = k * N;
      for (let j = 0; j < N; j++) out[o + j] += a * w[wk + j];
    }
  }
}

/** Per row: out = (x - mean) / sqrt(var + 1e-5) * weight + bias, with the biased (1/C) variance. */
export function layerNorm(
  out: Float32Array,
  x: Float32Array,
  weight: Float32Array,
  bias: Float32Array,
  n: number,
  C: number,
): void {
  for (let i = 0; i < n; i++) {
    const o = i * C;
    let mean = 0;
    for (let c = 0; c < C; c++) mean += x[o + c];
    mean /= C;
    let variance = 0;
    for (let c = 0; c < C; c++) {
      const d = x[o + c] - mean;
      variance += d * d;
    }
    const inv = 1 / Math.sqrt(variance / C + 1e-5);
    for (let c = 0; c < C; c++) out[o + c] = (x[o + c] - mean) * inv * weight[c] + bias[c];
  }
}

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);

/** GELU, tanh approximation (as in GPT-2), in place over the first `len` entries. */
export function gelu(x: Float32Array, len: number): void {
  for (let i = 0; i < len; i++) {
    const v = x[i];
    x[i] = 0.5 * v * (1 + Math.tanh(SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)));
  }
}

/**
 * In-place softmax over x[start .. start + len). Subtracting the max first keeps exp() from overflowing;
 * it cancels out in the normalisation. Entries equal to -Infinity come out as exactly 0.
 */
export function softmax(x: Float32Array, start: number, len: number): void {
  const end = start + len;
  let max = -Infinity;
  for (let i = start; i < end; i++) if (x[i] > max) max = x[i];
  let sum = 0;
  for (let i = start; i < end; i++) {
    const e = Math.exp(x[i] - max);
    x[i] = e;
    sum += e;
  }
  const inv = 1 / sum;
  for (let i = start; i < end; i++) x[i] *= inv;
}

/** a += b over the first `len` entries (the residual connection). */
export function addInPlace(a: Float32Array, b: Float32Array, len: number): void {
  for (let i = 0; i < len; i++) a[i] += b[i];
}
