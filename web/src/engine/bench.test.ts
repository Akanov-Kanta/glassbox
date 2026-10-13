// Micro-benchmark at the production model size (SPEC.md defaults) with random weights. It only logs
// timings; the thresholds are deliberately loose so it never flakes.

import { describe, expect, it } from 'vitest';
import { createModel } from './index';
import { randomModel, seededRandom, testVocab } from './testUtils';

describe('benchmark', () => {
  it('logs forward and per-token generation time (n_layer 4, n_head 4, n_embd 192, block 128, V 90)', () => {
    const { meta, buffer } = randomModel({ vocab: testVocab(90), nLayer: 4, nHead: 4, nEmbd: 192, blockSize: 128 });
    const model = createModel(meta, buffer);
    const prompt = model.encode('To be, or not to be, that is the question:\n');

    // Warm up the JIT.
    model.forward(prompt);
    let i = 0;
    for (const step of model.generate(prompt, { temperature: 1, topK: 0, random: seededRandom(1) })) {
      void step;
      if (++i === 50) break;
    }

    const fullContext = Array.from({ length: 128 }, (_, k) => k % 90);
    let t0 = performance.now();
    const runs = 5;
    for (let r = 0; r < runs; r++) model.forward(fullContext);
    const forwardMs = (performance.now() - t0) / runs;

    // 400 tokens from a 44-token prompt: includes several window slides (each a 64-token re-prefill).
    const tokens = 400;
    i = 0;
    t0 = performance.now();
    for (const step of model.generate(prompt, { temperature: 1, topK: 0, random: seededRandom(2) })) {
      void step;
      if (++i === tokens) break;
    }
    const perToken = (performance.now() - t0) / tokens;

    console.log(
      `benchmark: ${model.paramCount.toLocaleString('en-US')} params | forward(128 tokens) ${forwardMs.toFixed(1)} ms | ` +
        `generate ${perToken.toFixed(2)} ms/token (${Math.round(1000 / perToken)} tokens/s, incl. window slides)`,
    );
    expect(perToken).toBeGreaterThan(0);
    expect(perToken).toBeLessThan(1000);
  }, 60_000);
});
