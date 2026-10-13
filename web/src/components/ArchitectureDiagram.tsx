import type { ModelConfig } from '../engine/types';

interface Props {
  config: ModelConfig;
}

/** GPT-2 style, pre-LayerNorm stack, drawn as styled HTML so it reflows on small screens. */
export function ArchitectureDiagram({ config }: Props) {
  const { nLayer, nHead, nEmbd: C, blockSize: T, vocab } = config;
  const V = vocab.length;
  const hd = C / nHead;
  return (
    <figure className="arch" aria-labelledby="arch-cap">
      <figcaption id="arch-cap" className="visually-hidden">
        Model architecture: token and position embeddings, {nLayer} pre-LayerNorm transformer blocks with
        {` ${nHead}`}-head causal self-attention and a GELU MLP, a final LayerNorm, and logits computed with the
        tied token embedding.
      </figcaption>
      <ol className="arch__flow mono">
        <li className="arch__node arch__node--io">
          <span>{T} character ids</span>
          <span className="arch__dim">context window</span>
        </li>
        <li className="arch__node">
          <span>token emb + position emb</span>
          <span className="arch__dim">
            wte [{V}×{C}] + wpe [{T}×{C}]
          </span>
        </li>
        <li className="arch__block">
          <span className="arch__repeat">× {nLayer}</span>
          <ol>
            <li className="arch__node arch__node--sub">
              <span>LayerNorm</span>
            </li>
            <li className="arch__node arch__node--accent">
              <span>causal self-attention</span>
              <span className="arch__dim">
                {nHead} heads × {hd} dims, KV cached
              </span>
            </li>
            <li className="arch__node arch__node--res">
              <span>+ residual</span>
            </li>
            <li className="arch__node arch__node--sub">
              <span>LayerNorm</span>
            </li>
            <li className="arch__node">
              <span>MLP, GELU</span>
              <span className="arch__dim">
                {C} → {4 * C} → {C}
              </span>
            </li>
            <li className="arch__node arch__node--res">
              <span>+ residual</span>
            </li>
          </ol>
        </li>
        <li className="arch__node arch__node--sub">
          <span>LayerNorm</span>
        </li>
        <li className="arch__node">
          <span>logits = x · wteᵀ</span>
          <span className="arch__dim">tied weights, {V} outputs</span>
        </li>
        <li className="arch__node arch__node--io">
          <span>softmax, sample</span>
          <span className="arch__dim">temperature, top-k</span>
        </li>
      </ol>
    </figure>
  );
}
