import type { ModelConfig } from '../engine/types';
import { formatPct, spokenChar, visibleChar } from '../lib/format';
import { rankOf, topK } from '../lib/stats';

interface Props {
  probs: Float32Array;
  chosen: number;
  config: ModelConfig;
}

export function Candidates({ probs, chosen, config }: Props) {
  const top = topK(probs, 10);
  const max = probs[top[0]] || 1;
  const rows = top.includes(chosen) ? top : [...top, chosen];
  return (
    <ol className="cands" aria-label="Top 10 candidate next characters">
      {rows.map((id, i) => {
        const p = probs[id];
        const isChosen = id === chosen;
        const ch = config.vocab[id];
        const rank = i < top.length ? i + 1 : rankOf(probs, id);
        return (
          <li key={id} className={`cand${isChosen ? ' cand--chosen' : ''}`}>
            <span className="cand__rank mono" aria-hidden="true">
              {rank}
            </span>
            <span className="cand__char mono" aria-hidden="true">
              {visibleChar(ch)}
            </span>
            <span className="cand__bar" aria-hidden="true">
              <span style={{ width: `${Math.max(0.5, (p / max) * 100)}%` }} />
            </span>
            <span className="cand__pct mono">{formatPct(p)}</span>
            <span className="visually-hidden">
              {`Rank ${rank}: ${spokenChar(ch)}${isChosen ? ' (chosen)' : ''}`}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
