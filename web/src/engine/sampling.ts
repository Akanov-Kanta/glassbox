// Turning next-token logits into a choice.

import { softmax } from './kernels';

/**
 * The distribution generation samples from: softmax(logits / temperature), restricted to the `topK` most
 * likely tokens (0 = all of them) and renormalised. Returns a fresh array of length V.
 *
 * Temperature < 1 sharpens the distribution, > 1 flattens it. Top-k is applied after temperature, which
 * gives the same set either way since dividing by a positive number preserves the ranking.
 */
export function samplingDistribution(logits: Float32Array, temperature: number, topK: number): Float32Array {
  const V = logits.length;
  const probs = new Float32Array(V);
  for (let v = 0; v < V; v++) probs[v] = logits[v] / temperature;

  if (topK > 0 && topK < V) {
    // V is a few dozen characters, so a full sort per token costs nothing measurable.
    const ranked = Array.from({ length: V }, (_, v) => v).sort((a, b) => probs[b] - probs[a]);
    for (let r = topK; r < V; r++) probs[ranked[r]] = -Infinity;
  }

  softmax(probs, 0, V);
  return probs;
}

/** Inverse-CDF sampling: the first index whose cumulative probability exceeds `u` in [0, 1). */
export function sampleIndex(probs: Float32Array, u: number): number {
  let cumulative = 0;
  let lastNonZero = 0;
  for (let v = 0; v < probs.length; v++) {
    if (probs[v] === 0) continue;
    cumulative += probs[v];
    lastNonZero = v;
    if (u < cumulative) return v;
  }
  // Only reachable through rounding (the probabilities summed to slightly less than u).
  return lastNonZero;
}
