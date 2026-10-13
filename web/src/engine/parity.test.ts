// Parity with the PyTorch model: web/public/model/reference.json is produced by the training code from the
// same exported weights (SPEC.md, "Parity fixture").

import { describe, expect, it } from 'vitest';
import { createModel } from './index';
import type { ModelMeta } from './index';

interface Reference {
  input_ids: number[];
  logits_last: number[];
  attn_last_layer_head0: number[];
}

// The app's tsconfig has no Node types, so node:fs is imported untyped and given the shape we use.
interface Fs {
  existsSync(path: URL): boolean;
  readFileSync(path: URL): Uint8Array;
  readFileSync(path: URL, encoding: 'utf8'): string;
}
const fs = (await import('node:fs' as string)) as Fs;

const dir = new URL('../../public/model/', import.meta.url);
const files = ['model.json', 'model.bin', 'reference.json'].map((name) => new URL(name, dir));
const available = files.every((f) => fs.existsSync(f));

describe('parity with PyTorch', () => {
  if (!available) {
    it.skip('web/public/model/{model.json,model.bin,reference.json} not exported yet; skipping parity', () => {});
    return;
  }

  it('matches logits_last and attn_last_layer_head0 within 1e-4', () => {
    const [metaFile, binFile, refFile] = files;
    const meta = JSON.parse(fs.readFileSync(metaFile, 'utf8')) as ModelMeta;
    const bytes = fs.readFileSync(binFile);
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    const ref = JSON.parse(fs.readFileSync(refFile, 'utf8')) as Reference;

    const model = createModel(meta, buffer);
    const out = model.forward(ref.input_ids);
    expect(out.t).toBe(ref.input_ids.length);

    let maxLogitErr = 0;
    expect(out.logits.length).toBe(ref.logits_last.length);
    ref.logits_last.forEach((x, i) => (maxLogitErr = Math.max(maxLogitErr, Math.abs(out.logits[i] - x))));

    let maxAttnErr = 0;
    const attn = out.attention[meta.n_layer - 1][0];
    expect(attn.length).toBe(ref.attn_last_layer_head0.length);
    ref.attn_last_layer_head0.forEach((x, i) => (maxAttnErr = Math.max(maxAttnErr, Math.abs(attn[i] - x))));

    console.log(`parity: max |logit err| = ${maxLogitErr.toExponential(2)}, max |attn err| = ${maxAttnErr.toExponential(2)}`);
    expect(maxLogitErr).toBeLessThan(1e-4);
    expect(maxAttnErr).toBeLessThan(1e-4);
  });
});
