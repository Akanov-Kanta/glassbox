// The GPT itself: a pre-LayerNorm transformer decoder (SPEC.md, "Architecture").
//
// There is a single code path, `advance`, which pushes a chunk of new tokens through the network and
// appends their keys and values to a KV cache. A full forward pass is `advance` on an empty cache with the
// whole context; a generation step is `advance` with a single token. Every token therefore attends to
// exactly the same keys and values whichever way it was computed.

import type { Weights } from './format';
import { addInPlace, gelu, layerNorm, linear, softmax } from './kernels';
import { sampleIndex, samplingDistribution } from './sampling';
import { CharTokenizer } from './tokenizer';
import type { ForwardResult, GenStep, Model, ModelConfig, SampleOptions } from './types';

/** Keys and values of every position processed so far, one [blockSize, C] row-major buffer per layer. */
class KVCache {
  readonly k: Float32Array[];
  readonly v: Float32Array[];
  /** number of positions filled */
  length = 0;

  constructor(nLayer: number, blockSize: number, C: number) {
    this.k = Array.from({ length: nLayer }, () => new Float32Array(blockSize * C));
    this.v = Array.from({ length: nLayer }, () => new Float32Array(blockSize * C));
  }
}

export class GPT implements Model {
  readonly config: ModelConfig;
  readonly paramCount: number;

  private readonly w: Weights;
  private readonly tokenizer: CharTokenizer;
  private readonly forwardCache: KVCache;

  // Scratch buffers sized for a full context (blockSize rows), reused by every call.
  private readonly x: Float32Array; // residual stream [T, C]
  private readonly normed: Float32Array; // LayerNorm output [T, C]
  private readonly qkv: Float32Array; // [T, 3C]
  private readonly heads: Float32Array; // concatenated attention outputs [T, C]
  private readonly proj: Float32Array; // output of an attention / MLP projection [T, C]
  private readonly hidden: Float32Array; // MLP hidden layer [T, 4C]
  private readonly scores: Float32Array; // one attention row, when the caller does not want it [T]
  private readonly logits: Float32Array; // next-token logits after the last position [V]

  constructor(weights: Weights) {
    this.w = weights;
    this.config = weights.config;
    this.paramCount = weights.paramCount;
    this.tokenizer = new CharTokenizer(weights.config.vocab);

    const { nLayer, nEmbd: C, blockSize: T, vocab } = this.config;
    this.forwardCache = new KVCache(nLayer, T, C);
    this.x = new Float32Array(T * C);
    this.normed = new Float32Array(T * C);
    this.qkv = new Float32Array(T * 3 * C);
    this.heads = new Float32Array(T * C);
    this.proj = new Float32Array(T * C);
    this.hidden = new Float32Array(T * 4 * C);
    this.scores = new Float32Array(T);
    this.logits = new Float32Array(vocab.length);
  }

  encode(text: string): number[] {
    return this.tokenizer.encode(text);
  }

  decode(ids: number[]): string {
    return this.tokenizer.decode(ids);
  }

  forward(ids: number[]): ForwardResult {
    this.checkIds(ids);
    if (ids.length === 0) throw new Error('forward() needs at least one token');
    const context = ids.slice(-this.config.blockSize);
    const t = context.length;

    const attention = this.newAttention(t * t);
    this.forwardCache.length = 0;
    this.advance(this.forwardCache, context, attention);

    const logits = this.logits.slice();
    const probs = logits.slice();
    softmax(probs, 0, probs.length);
    return { logits, probs, attention, t };
  }

  /**
   * Validates eagerly (a generator body would only run on the first next()), then hands off to `run`.
   *
   * Empty prompt: generation is seeded with a newline (the natural "start of a passage" in a plain-text
   * corpus), or with id 0 if the vocabulary has no newline. The seed is context only; it is not yielded.
   */
  generate(promptIds: number[], opts: SampleOptions): Generator<GenStep, void, void> {
    this.checkIds(promptIds);
    const { temperature, topK } = opts;
    if (!(temperature > 0) || !Number.isFinite(temperature)) {
      throw new RangeError(`temperature must be a positive number, got ${temperature}`);
    }
    if (!Number.isInteger(topK) || topK < 0) throw new RangeError(`topK must be a non-negative integer, got ${topK}`);

    const seed = this.tokenizer.idOf('\n') ?? 0;
    const prompt = promptIds.length > 0 ? promptIds.slice(-this.config.blockSize) : [seed];
    return this.run(prompt, temperature, topK, opts.random ?? Math.random);
  }

  /**
   * The generation loop. Each step samples from the logits left by the previous `advance`, yields the
   * token together with the attention row of the query that produced those logits, then feeds the token
   * back in.
   *
   * Window policy: positions use learned absolute embeddings, so a cached key/value belongs to its position
   * and the cache cannot be shifted by one. When the context is full (blockSize tokens) and another token
   * arrives, the context is cut to its most recent blockSize/2 tokens (including the new one) and the cache
   * is rebuilt from them. Re-encoding the full window minus one on every step would keep more context but
   * cost a whole forward pass per token; halving keeps the amortised cost close to a plain cached step
   * while never leaving the model with less than half a window of context.
   */
  private *run(
    prompt: number[],
    temperature: number,
    topK: number,
    random: () => number,
  ): Generator<GenStep, void, void> {
    const { blockSize, vocab } = this.config;
    const keep = Math.max(1, Math.floor(blockSize / 2));
    const cache = new KVCache(this.config.nLayer, blockSize, this.config.nEmbd);

    let context = prompt;
    let attention = this.prefill(cache, context);
    for (;;) {
      // Read the logits before yielding: the caller may run forward() on this model in between steps.
      const probs = samplingDistribution(this.logits, temperature, topK);
      const id = sampleIndex(probs, random());
      yield { id, char: vocab[id], probs, attention };

      if (context.length < blockSize) {
        context.push(id);
        attention = this.step(cache, id);
      } else {
        context = [...context.slice(context.length - keep + 1), id];
        attention = this.prefill(cache, context);
      }
    }
  }

  /** Rebuilds `cache` from `ids` and returns the attention row of the last token. */
  private prefill(cache: KVCache, ids: number[]): Float32Array[][] {
    cache.length = 0;
    // Everything but the last token in one batched pass (no attention needed), then the last token as a
    // regular step so that we get its attention row.
    if (ids.length > 1) this.advance(cache, ids.slice(0, -1), null);
    return this.step(cache, ids[ids.length - 1]);
  }

  /** Appends one token; returns its attention over the whole context, per [layer][head]. */
  private step(cache: KVCache, id: number): Float32Array[][] {
    const attention = this.newAttention(cache.length + 1);
    this.advance(cache, [id], attention);
    return attention;
  }

  /**
   * Runs `ids` through the network at positions cache.length .. cache.length + n - 1, appends their keys
   * and values to `cache`, and leaves the next-token logits after the last of them in `this.logits`.
   *
   * If `attn` is given, attn[layer][head] receives this chunk's attention weights as an
   * n x (cache.length + n) row-major matrix. Entries above the diagonal are not written (left at zero).
   */
  private advance(cache: KVCache, ids: number[], attn: Float32Array[][] | null): void {
    const { nLayer, nHead, nEmbd: C, blockSize } = this.config;
    const { wte, wpe, layers } = this.w;
    const { x, normed, qkv, heads, proj, hidden } = this;
    const n = ids.length;
    const start = cache.length;
    const end = start + n; // context length once this chunk is in
    if (end > blockSize) throw new Error(`context of ${end} tokens exceeds block size ${blockSize}`);

    const headDim = C / nHead;
    const scale = 1 / Math.sqrt(headDim);
    const C3 = 3 * C;

    // Token embedding + position embedding.
    for (let r = 0; r < n; r++) {
      const te = ids[r] * C;
      const pe = (start + r) * C;
      const o = r * C;
      for (let c = 0; c < C; c++) x[o + c] = wte[te + c] + wpe[pe + c];
    }

    for (let l = 0; l < nLayer; l++) {
      const L = layers[l];
      const K = cache.k[l];
      const V = cache.v[l];

      // ---- Attention: x += c_proj(attention(LN(x))) ----
      layerNorm(normed, x, L.ln1W, L.ln1B, n, C);
      linear(qkv, normed, L.attnW, L.attnB, n, C, C3);
      // Each qkv row is [q | k | v]; the new keys and values go into the cache at their positions.
      for (let r = 0; r < n; r++) {
        const row = r * C3;
        K.set(qkv.subarray(row + C, row + 2 * C), (start + r) * C);
        V.set(qkv.subarray(row + 2 * C, row + C3), (start + r) * C);
      }

      for (let h = 0; h < nHead; h++) {
        const hOff = h * headDim; // head h owns columns [hOff, hOff + headDim) of q, k, v and the output
        for (let r = 0; r < n; r++) {
          const pos = start + r; // the query's position; it may attend to positions 0..pos (causal)
          const weights = attn ? attn[l][h] : this.scores;
          const w0 = attn ? r * end : 0;
          const q = r * C3 + hOff;

          // scores_j = q . k_j / sqrt(headDim)
          for (let j = 0; j <= pos; j++) {
            const k = j * C + hOff;
            let dot = 0;
            for (let d = 0; d < headDim; d++) dot += qkv[q + d] * K[k + d];
            weights[w0 + j] = dot * scale;
          }
          softmax(weights, w0, pos + 1);

          // output = sum_j weights_j * v_j
          const out = r * C + hOff;
          heads.fill(0, out, out + headDim);
          for (let j = 0; j <= pos; j++) {
            const p = weights[w0 + j];
            const v = j * C + hOff;
            for (let d = 0; d < headDim; d++) heads[out + d] += p * V[v + d];
          }
        }
      }
      linear(proj, heads, L.attnProjW, L.attnProjB, n, C, C);
      addInPlace(x, proj, n * C);

      // ---- MLP: x += c_proj(GELU(c_fc(LN(x)))) ----
      layerNorm(normed, x, L.ln2W, L.ln2B, n, C);
      linear(hidden, normed, L.fcW, L.fcB, n, C, 4 * C);
      gelu(hidden, n * 4 * C);
      linear(proj, hidden, L.mlpProjW, L.mlpProjB, n, 4 * C, C);
      addInPlace(x, proj, n * C);
    }
    cache.length = end;

    // Final LayerNorm and the tied output head (logits = x @ wte^T), for the last position only.
    layerNorm(normed, x.subarray((n - 1) * C), this.w.lnfW, this.w.lnfB, 1, C);
    const { logits } = this;
    for (let v = 0; v < logits.length; v++) {
      const e = v * C;
      let dot = 0;
      for (let c = 0; c < C; c++) dot += normed[c] * wte[e + c];
      logits[v] = dot;
    }
  }

  /** A fresh attention[layer][head] array of zeroed buffers. */
  private newAttention(size: number): Float32Array[][] {
    const { nLayer, nHead } = this.config;
    return Array.from({ length: nLayer }, () => Array.from({ length: nHead }, () => new Float32Array(size)));
  }

  private checkIds(ids: number[]): void {
    const V = this.config.vocab.length;
    for (const id of ids) {
      if (!Number.isInteger(id) || id < 0 || id >= V) {
        throw new RangeError(`token id ${id} is out of range for a vocabulary of ${V}`);
      }
    }
  }
}
