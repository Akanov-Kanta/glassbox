import { describe, expect, it } from 'vitest';
import { createModel } from './index';
import type { GenStep, Model, SampleOptions } from './index';
import { randomModel, seededRandom, testVocab } from './testUtils';

const SPEC = { vocab: testVocab(11), nLayer: 2, nHead: 2, nEmbd: 16, blockSize: 8 };
const { meta, buffer } = randomModel(SPEC);
const model = createModel(meta, buffer);
const V = SPEC.vocab.length;

function take(model: Model, prompt: number[], opts: SampleOptions, n: number): GenStep[] {
  const steps: GenStep[] = [];
  for (const step of model.generate(prompt, opts)) {
    steps.push(step);
    if (steps.length === n) break;
  }
  return steps;
}

function softmaxOf(logits: ArrayLike<number>, temperature: number, keep: (v: number) => boolean): number[] {
  const scaled = Array.from(logits, (x, v) => (keep(v) ? x / temperature : -Infinity));
  const max = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - max));
  const sum = exps.reduce((s, x) => s + x, 0);
  return exps.map((x) => x / sum);
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  expect(a.length).toBe(b.length);
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  return max;
}

describe('generate: KV cache matches a fresh forward pass', () => {
  /**
   * The context of each step is always a suffix of prompt + generated tokens, and its length is the length
   * of the step's attention row, so every step can be replayed with forward().
   */
  function checkAgainstForward(prompt: number[], opts: SampleOptions, n: number) {
    const steps = take(model, prompt, opts, n);
    const sequence = prompt.length > 0 ? [...prompt] : [model.encode('\n')[0]];
    const lengths: number[] = [];
    for (const step of steps) {
      const L = step.attention[0][0].length;
      lengths.push(L);
      expect(L).toBeLessThanOrEqual(SPEC.blockSize);
      const context = sequence.slice(sequence.length - L);
      const full = model.forward(context);

      const k = opts.topK > 0 ? opts.topK : V;
      const ranked = Array.from(full.logits.keys()).sort((a, b) => full.logits[b] - full.logits[a]);
      const kept = new Set(ranked.slice(0, k));
      const expected = softmaxOf(full.logits, opts.temperature, (v) => kept.has(v));
      expect(maxAbsDiff(step.probs, expected)).toBeLessThan(1e-4);

      for (let l = 0; l < SPEC.nLayer; l++) {
        for (let h = 0; h < SPEC.nHead; h++) {
          const lastRow = full.attention[l][h].subarray((L - 1) * L, L * L);
          expect(maxAbsDiff(step.attention[l][h], lastRow)).toBeLessThan(1e-4);
        }
      }
      expect(step.char).toBe(SPEC.vocab[step.id]);
      sequence.push(step.id);
    }
    return lengths;
  }

  it('at temperature 1 without top-k, including across window slides', () => {
    const lengths = checkAgainstForward([4, 7, 2], { temperature: 1, topK: 0, random: seededRandom(42) }, 30);
    // 3, 4, ... 8, then the window slides to blockSize / 2 = 4 tokens and grows again.
    expect(lengths.slice(0, 10)).toEqual([3, 4, 5, 6, 7, 8, 4, 5, 6, 7]);
    expect(lengths.filter((L) => L === 4).length).toBeGreaterThan(3);
  });

  it('with temperature and top-k', () => {
    checkAgainstForward([1, 2, 3, 4, 5], { temperature: 0.7, topK: 3, random: seededRandom(7) }, 20);
  });

  it('from an empty prompt (seeded with a newline)', () => {
    const lengths = checkAgainstForward([], { temperature: 1.3, topK: 0, random: seededRandom(3) }, 12);
    expect(lengths[0]).toBe(1);
  });

  it('from a prompt longer than the block size (cropped)', () => {
    const prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2];
    const steps = take(model, prompt, { temperature: 1, topK: 0, random: seededRandom(1) }, 1);
    expect(steps[0].attention[0][0].length).toBe(SPEC.blockSize);
    expect(maxAbsDiff(steps[0].probs, model.forward(prompt).probs)).toBeLessThan(1e-5);
  });

  it('is unaffected by forward() calls between steps', () => {
    const opts = () => ({ temperature: 1, topK: 0, random: seededRandom(9) });
    const plain = take(model, [3], opts(), 15).map((s) => s.id);
    const interleaved: number[] = [];
    for (const step of model.generate([3], opts())) {
      interleaved.push(step.id);
      model.forward([9, 9, 9, 9, 9]);
      if (interleaved.length === 15) break;
    }
    expect(interleaved).toEqual(plain);
  });
});

describe('generate: sampling', () => {
  it('top-k = 1 is greedy', () => {
    const steps = take(model, [5, 6], { temperature: 1, topK: 1, random: seededRandom(5) }, 12);
    const sequence = [5, 6];
    for (const step of steps) {
      const { logits } = model.forward(sequence.slice(sequence.length - step.attention[0][0].length));
      const argmax = Array.from(logits).indexOf(Math.max(...logits));
      expect(step.id).toBe(argmax);
      expect(step.probs[argmax]).toBe(1);
      expect(step.probs.reduce((s, p) => s + p, 0)).toBe(1);
      sequence.push(step.id);
    }
  });

  it('top-k keeps exactly k tokens', () => {
    const [step] = take(model, [2], { temperature: 1, topK: 4, random: seededRandom(5) }, 1);
    expect(Array.from(step.probs).filter((p) => p > 0).length).toBe(4);
    expect(step.probs.reduce((s, p) => s + p, 0)).toBeCloseTo(1, 6);
  });

  it('temperature sharpens (< 1) or flattens (> 1) the distribution', () => {
    const entropy = (p: Float32Array) => -p.reduce((s, x) => (x > 0 ? s + x * Math.log(x) : s), 0);
    const first = (temperature: number) => take(model, [8, 1], { temperature, topK: 0 }, 1)[0].probs;
    const [cold, base, hot] = [first(0.5), first(1), first(2)];
    expect(entropy(cold)).toBeLessThan(entropy(base));
    expect(entropy(base)).toBeLessThan(entropy(hot));
    const argmax = Array.from(base).indexOf(Math.max(...base));
    expect(cold[argmax]).toBeGreaterThan(base[argmax]);
    expect(first(1e-4)[argmax]).toBeCloseTo(1, 6); // T -> 0 approaches greedy
  });

  it('is deterministic for a given random source, and uses it', () => {
    const run = (seed: number) =>
      take(model, [1], { temperature: 1.5, topK: 0, random: seededRandom(seed) }, 25).map((s) => s.id);
    expect(run(11)).toEqual(run(11));
    expect(run(11)).not.toEqual(run(12));
  });

  it('samples by inverse CDF from the returned distribution', () => {
    // u = 0 picks the first token with non-zero probability; u -> 1 picks the last.
    const [low] = take(model, [3], { temperature: 1, topK: 0, random: () => 0 }, 1);
    expect(low.id).toBe(Array.from(low.probs).findIndex((p) => p > 0));
    const [high] = take(model, [3], { temperature: 1, topK: 3, random: () => 0.999999 }, 1);
    const kept = Array.from(high.probs.keys()).filter((v) => high.probs[v] > 0);
    expect(high.id).toBe(kept[kept.length - 1]);
  });

  it('validates options eagerly', () => {
    expect(() => model.generate([1], { temperature: 0, topK: 0 })).toThrow(/temperature/);
    expect(() => model.generate([1], { temperature: 1, topK: -1 })).toThrow(/topK/);
    expect(() => model.generate([V], { temperature: 1, topK: 0 })).toThrow(/out of range/);
  });
});
