// Glassbox inference engine: a character-level GPT in plain TypeScript, no ML or math libraries.
//
//   format.ts     model.json + model.bin -> validated, named Float32Array views
//   kernels.ts    linear, layerNorm, gelu, softmax
//   gpt.ts        the transformer: forward pass, KV-cached generation
//   sampling.ts   temperature, top-k, inverse-CDF sampling
//   tokenizer.ts  characters <-> ids

import { parseWeights } from './format';
import type { ModelMeta } from './format';
import { GPT } from './gpt';
import type { LoadModel, Model } from './types';

export type * from './types';
export type { ModelMeta, ParamEntry } from './format';

/** Builds a model from an already-loaded model.json object and model.bin buffer. Throws if they disagree. */
export function createModel(meta: ModelMeta, buffer: ArrayBuffer): Model {
  return new GPT(parseWeights(meta, buffer));
}

async function fetchOk(url: string): Promise<Response> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: HTTP ${res.status}`);
  return res;
}

/** Fetches model.json and model.bin in parallel and builds the model. */
export const loadModel: LoadModel = async (jsonUrl, binUrl) => {
  const [metaRes, binRes] = await Promise.all([fetchOk(jsonUrl), fetchOk(binUrl)]);
  const [meta, buffer] = await Promise.all([metaRes.json() as Promise<ModelMeta>, binRes.arrayBuffer()]);
  return createModel(meta, buffer);
};
