import type { ModelConfig } from '../engine/types';
import type { Run } from '../hooks/useGeneration';
import { formatPct, spokenChar, visibleChar } from '../lib/format';
import { entropyBits, meanRows, rankOf } from '../lib/stats';
import { AttentionContext } from './AttentionContext';
import { Candidates } from './Candidates';
import { HeadGrid, type HeadChoice } from './HeadGrid';

interface Props {
  run: Run | null;
  index: number | null;
  pinned: boolean;
  config: ModelConfig;
  layer: number;
  head: HeadChoice;
  onLayerHead: (layer: number, head: HeadChoice) => void;
}

/**
 * The characters a step's attention row refers to. The row belongs to the query at the last position of
 * the context the model saw, so it covers the `len` tokens immediately before the chosen character.
 */
function contextChars(run: Run, index: number, len: number, vocab: string[]): string[] {
  const ids = run.promptIds.concat(run.steps.slice(0, index + 1).map((s) => s.id));
  const p = run.promptIds.length + index;
  const start = Math.max(0, p - len);
  return ids.slice(start, start + len).map((id) => vocab[id] ?? '');
}

export function Inspector({ run, index, pinned, config, layer, head, onLayerHead }: Props) {
  const step = run && index !== null ? run.steps[index] : undefined;

  if (!run || !step || index === null) {
    return (
      <aside className="inspector inspector--empty" aria-label="Inspector">
        <h3 className="inspector__title">Inspector</h3>
        <p className="inspector__hint">
          Generate some text, then hover or click any character to see the candidates the model weighed and
          which earlier characters it attended to.
        </p>
      </aside>
    );
  }

  const p = step.probs[step.id];
  const rank = rankOf(step.probs, step.id);
  const len = step.attention[0]?.[0]?.length ?? 0;
  const chars = contextChars(run, index, len, config.vocab);
  const layerRows = step.attention[layer] ?? step.attention[0];
  const weights = head === 'avg' ? meanRows(layerRows) : layerRows[head];
  const { temperature, topK } = run.settings;

  return (
    <aside className="inspector" aria-label="Inspector">
      <div className="inspector__head">
        <div className="glyph mono" aria-hidden="true">
          {visibleChar(step.char)}
        </div>
        <div className="inspector__meta">
          <h3 className="inspector__title">
            Character {index + 1}
            <span className="visually-hidden">: {spokenChar(step.char)}</span>
            <span className="inspector__mode label">{pinned ? 'pinned' : 'live'}</span>
          </h3>
          <dl className="stats mono">
            <div>
              <dt>p</dt>
              <dd>{formatPct(p)}</dd>
            </div>
            <div>
              <dt>rank</dt>
              <dd>
                {rank} / {config.vocab.length}
              </dd>
            </div>
            <div>
              <dt>surprisal</dt>
              <dd>{(-Math.log2(Math.max(p, 1e-12))).toFixed(2)} bits</dd>
            </div>
            <div>
              <dt>entropy</dt>
              <dd>{entropyBits(step.probs).toFixed(2)} bits</dd>
            </div>
          </dl>
        </div>
      </div>

      <section className="panel" aria-labelledby="cands-title">
        <h4 id="cands-title" className="panel__title">
          Next-character candidates
          <span className="panel__note mono">
            T={temperature.toFixed(2)}, top-k={topK || 'off'}
          </span>
        </h4>
        <Candidates probs={step.probs} chosen={step.id} config={config} />
      </section>

      <section className="panel" aria-labelledby="attn-title">
        <div className="panel__bar">
          <h4 id="attn-title" className="panel__title">
            Attention
          </h4>
          <div className="seg-group">
            <div className="seg" role="group" aria-label="Layer">
              {Array.from({ length: config.nLayer }, (_, l) => (
                <button
                  key={l}
                  type="button"
                  aria-pressed={layer === l}
                  className="seg__btn mono"
                  onClick={() => onLayerHead(l, head)}
                >
                  L{l}
                </button>
              ))}
            </div>
            <div className="seg" role="group" aria-label="Head">
              {Array.from({ length: config.nHead }, (_, h) => (
                <button
                  key={h}
                  type="button"
                  aria-pressed={head === h}
                  className="seg__btn mono"
                  onClick={() => onLayerHead(layer, h)}
                >
                  H{h}
                </button>
              ))}
              <button
                type="button"
                aria-pressed={head === 'avg'}
                className="seg__btn mono"
                onClick={() => onLayerHead(layer, 'avg')}
              >
                avg
              </button>
            </div>
          </div>
        </div>
        <AttentionContext chars={chars} weights={weights} chosen={step.char} />
      </section>

      <section className="panel" aria-labelledby="heads-title">
        <h4 id="heads-title" className="panel__title">
          Every head
          <span className="panel__note mono">peak = where it looks hardest</span>
        </h4>
        <HeadGrid attention={step.attention} layer={layer} head={head} onSelect={onLayerHead} />
      </section>
    </aside>
  );
}
