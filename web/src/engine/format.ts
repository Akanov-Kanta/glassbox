// Parsing and validation of the exported weights (SPEC.md, "Weight file format").
//
// model.json describes the tensors; model.bin is one flat little-endian float32 array. Every tensor is
// handed to the model as a zero-copy Float32Array view into that array.

import type { ModelConfig } from './types';

/** The JSON header that accompanies model.bin. */
export interface ModelMeta {
  vocab: string[];
  n_layer: number;
  n_head: number;
  n_embd: number;
  block_size: number;
  dtype: string;
  params: ParamEntry[];
}

export interface ParamEntry {
  name: string;
  shape: number[];
  /** in floats (not bytes) from the start of model.bin */
  offset: number;
}

/** One transformer block. Linear weights are [in, out], row-major, so y = x @ W + b. */
export interface LayerWeights {
  ln1W: Float32Array; // [C]
  ln1B: Float32Array; // [C]
  attnW: Float32Array; // [C, 3C], output columns are q | k | v
  attnB: Float32Array; // [3C]
  attnProjW: Float32Array; // [C, C]
  attnProjB: Float32Array; // [C]
  ln2W: Float32Array; // [C]
  ln2B: Float32Array; // [C]
  fcW: Float32Array; // [C, 4C]
  fcB: Float32Array; // [4C]
  mlpProjW: Float32Array; // [4C, C]
  mlpProjB: Float32Array; // [C]
}

export interface Weights {
  config: ModelConfig;
  wte: Float32Array; // [V, C], also the (tied) output projection
  wpe: Float32Array; // [T, C]
  layers: LayerWeights[];
  lnfW: Float32Array; // [C]
  lnfB: Float32Array; // [C]
  /** total number of floats across all tensors */
  paramCount: number;
}

function fail(message: string): never {
  throw new Error(`Invalid model: ${message}`);
}

function positiveInt(meta: ModelMeta, key: 'n_layer' | 'n_head' | 'n_embd' | 'block_size'): number {
  const value = meta[key];
  if (!Number.isInteger(value) || value <= 0) fail(`"${key}" must be a positive integer, got ${JSON.stringify(value)}`);
  return value;
}

function parseConfig(meta: ModelMeta): ModelConfig {
  const { vocab } = meta;
  if (!Array.isArray(vocab) || vocab.length === 0 || !vocab.every((c) => typeof c === 'string')) {
    fail('"vocab" must be a non-empty array of strings');
  }
  if (new Set(vocab).size !== vocab.length) fail('"vocab" contains duplicate entries');

  const config: ModelConfig = {
    vocab,
    nLayer: positiveInt(meta, 'n_layer'),
    nHead: positiveInt(meta, 'n_head'),
    nEmbd: positiveInt(meta, 'n_embd'),
    blockSize: positiveInt(meta, 'block_size'),
  };
  if (config.nEmbd % config.nHead !== 0) {
    fail(`n_embd (${config.nEmbd}) must be divisible by n_head (${config.nHead})`);
  }
  return config;
}

/** model.bin as float32. Typed arrays use the host byte order, so convert on (rare) big-endian hosts. */
function littleEndianFloats(buffer: ArrayBuffer): Float32Array {
  if (buffer.byteLength % 4 !== 0) fail(`model.bin is ${buffer.byteLength} bytes, not a whole number of float32s`);
  const hostIsLittleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
  if (hostIsLittleEndian) return new Float32Array(buffer);
  const view = new DataView(buffer);
  const out = new Float32Array(buffer.byteLength / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

/** Validates model.json against SPEC.md and slices model.bin into named tensors. Throws on any mismatch. */
export function parseWeights(meta: ModelMeta, buffer: ArrayBuffer): Weights {
  if (meta === null || typeof meta !== 'object') fail('model.json is not an object');
  const config = parseConfig(meta);
  if (meta.dtype !== 'f32') fail(`dtype must be "f32", got ${JSON.stringify(meta.dtype)}`);
  if (!Array.isArray(meta.params)) fail('"params" must be an array');

  const data = littleEndianFloats(buffer);
  const entries = new Map<string, ParamEntry>();
  for (const p of meta.params) {
    if (entries.has(p.name)) fail(`duplicate tensor "${p.name}"`);
    entries.set(p.name, p);
  }

  let paramCount = 0;
  /** Removes the named tensor from `entries`, checks its shape and bounds, and returns a view onto it. */
  const take = (name: string, ...shape: number[]): Float32Array => {
    const p = entries.get(name);
    if (!p) fail(`missing tensor "${name}"`);
    entries.delete(name);
    const sameShape = Array.isArray(p.shape) && p.shape.length === shape.length && p.shape.every((d, i) => d === shape[i]);
    if (!sameShape) fail(`tensor "${name}" has shape [${p.shape}], expected [${shape}]`);
    const size = shape.reduce((a, b) => a * b, 1);
    if (!Number.isInteger(p.offset) || p.offset < 0 || p.offset + size > data.length) {
      fail(`tensor "${name}" (offset ${p.offset}, ${size} floats) does not fit in model.bin (${data.length} floats)`);
    }
    paramCount += size;
    return data.subarray(p.offset, p.offset + size);
  };

  const { nLayer, nEmbd: C, blockSize: T } = config;
  const V = config.vocab.length;
  const wte = take('wte', V, C);
  const wpe = take('wpe', T, C);
  const layers: LayerWeights[] = [];
  for (let i = 0; i < nLayer; i++) {
    const h = `h.${i}`;
    layers.push({
      ln1W: take(`${h}.ln_1.weight`, C),
      ln1B: take(`${h}.ln_1.bias`, C),
      attnW: take(`${h}.attn.c_attn.weight`, C, 3 * C),
      attnB: take(`${h}.attn.c_attn.bias`, 3 * C),
      attnProjW: take(`${h}.attn.c_proj.weight`, C, C),
      attnProjB: take(`${h}.attn.c_proj.bias`, C),
      ln2W: take(`${h}.ln_2.weight`, C),
      ln2B: take(`${h}.ln_2.bias`, C),
      fcW: take(`${h}.mlp.c_fc.weight`, C, 4 * C),
      fcB: take(`${h}.mlp.c_fc.bias`, 4 * C),
      mlpProjW: take(`${h}.mlp.c_proj.weight`, 4 * C, C),
      mlpProjB: take(`${h}.mlp.c_proj.bias`, C),
    });
  }
  const lnfW = take('ln_f.weight', C);
  const lnfB = take('ln_f.bias', C);
  if (entries.size > 0) fail(`unexpected tensors: ${[...entries.keys()].join(', ')}`);

  return { config, wte, wpe, layers, lnfW, lnfB, paramCount };
}
