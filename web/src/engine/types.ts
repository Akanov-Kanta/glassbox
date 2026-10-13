// FROZEN interface between the engine (web/src/engine/) and the UI. See SPEC.md.

export interface ModelConfig {
  vocab: string[];
  nLayer: number;
  nHead: number;
  nEmbd: number;
  blockSize: number;
}

/** One full forward pass over a context of T tokens (T <= blockSize). */
export interface ForwardResult {
  /** logits for the next token after the last position, length V */
  logits: Float32Array;
  /** softmax(logits / temperature=1), length V */
  probs: Float32Array;
  /** attention[layer][head]: T*T row-major, causal (zeros above the diagonal) */
  attention: Float32Array[][];
  /** number of positions T actually used (the input is cropped to its last blockSize tokens) */
  t: number;
}

export interface SampleOptions {
  /** > 0; 1 = the model's distribution */
  temperature: number;
  /** keep the k most likely tokens; 0 = no top-k */
  topK: number;
  /** uniform [0,1) source, injectable for tests */
  random?: () => number;
}

/** One generated token, with everything the UI shows about the decision. */
export interface GenStep {
  id: number;
  char: string;
  /** the distribution it was sampled from (after temperature / top-k), length V */
  probs: Float32Array;
  /** attention of this token's query over the context, per [layer][head]: length = context length at that step */
  attention: Float32Array[][];
}

export interface Model {
  readonly config: ModelConfig;
  readonly paramCount: number;
  encode(text: string): number[];
  decode(ids: number[]): string;
  forward(ids: number[]): ForwardResult;
  /**
   * Autoregressive generation with a KV cache. Yields one step per token; the caller decides when to
   * stop (and can await between steps to keep the UI responsive). The prompt is cropped to blockSize;
   * when the context fills up, the window slides (the cache is rebuilt).
   */
  generate(promptIds: number[], opts: SampleOptions): Generator<GenStep, void, void>;
}

/** Implemented in web/src/engine/index.ts */
export type LoadModel = (jsonUrl: string, binUrl: string) => Promise<Model>;
