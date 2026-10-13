import { describe, expect, it } from 'vitest';
import { createModel } from './index';
import { halfToFloat } from './format';
import { randomModel, testVocab } from './testUtils';

const SPEC = { vocab: testVocab(11), nLayer: 2, nHead: 2, nEmbd: 16, blockSize: 8 };

/** Nearest float16 bit pattern to x. Positive finite halves (bits 0..0x7bff) increase monotonically, so binary search. */
function toHalfBits(x: number): number {
  const a = Math.abs(x);
  let lo = 0, hi = 0x7bff;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (halfToFloat(mid) < a) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && a - halfToFloat(lo - 1) < halfToFloat(lo) - a) lo -= 1;
  return (x < 0 ? 0x8000 : 0) | lo;
}

describe('float16 weights', () => {
  it('decodes binary16 bit patterns exactly', () => {
    expect(halfToFloat(0x0000)).toBe(0);
    expect(Object.is(halfToFloat(0x8000), -0)).toBe(true);
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x3555)).toBe(0.333251953125);
    expect(halfToFloat(0x7bff)).toBe(65504); // largest finite
    expect(halfToFloat(0x0400)).toBe(2 ** -14); // smallest normal
    expect(halfToFloat(0x0001)).toBe(2 ** -24); // smallest subnormal
    expect(halfToFloat(0x7c00)).toBe(Infinity);
    expect(halfToFloat(0xfc00)).toBe(-Infinity);
    expect(halfToFloat(0x7e00)).toBeNaN();
  });

  it('an f16 file gives exactly the outputs of the same weights stored as f32', () => {
    const { meta, buffer } = randomModel({ ...SPEC, seed: 7 });
    const src = new Float32Array(buffer);
    const bits = new Uint16Array(src.length);
    const widened = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) {
      bits[i] = toHalfBits(src[i]);
      widened[i] = halfToFloat(bits[i]);
    }
    const f16 = createModel({ ...meta, dtype: 'f16' }, bits.buffer);
    const f32 = createModel({ ...meta, dtype: 'f32' }, widened.buffer);
    const ids = [0, 3, 5, 1, 9, 2, 4, 7];
    const a = f16.forward(ids), b = f32.forward(ids);
    expect(Array.from(a.logits)).toEqual(Array.from(b.logits));
    expect(a.attention.map((l) => l.map((h) => Array.from(h)))).toEqual(b.attention.map((l) => l.map((h) => Array.from(h))));
    expect(f16.paramCount).toBe(f32.paramCount);
  });

  it('rejects an f16 file with an odd byte length', () => {
    const { meta } = randomModel({ ...SPEC, seed: 7 });
    expect(() => createModel({ ...meta, dtype: 'f16' }, new ArrayBuffer(7))).toThrow(/float16/);
  });
});
