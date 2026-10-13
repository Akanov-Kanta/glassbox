import { useState, type CSSProperties } from 'react';
import { formatPct, spokenChar, visibleChar } from '../lib/format';
import { maxOf } from '../lib/stats';

interface Props {
  chars: string[];
  weights: Float32Array;
  chosen: string;
}

export function AttentionContext({ chars, weights, chosen }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const n = Math.min(chars.length, weights.length);
  const max = maxOf(weights) || 1;
  const order = Array.from({ length: n }, (_, j) => j).sort((a, b) => weights[b] - weights[a]);
  const strongest = order.slice(0, 3);
  const back = (j: number) => n - 1 - j;
  const describe = (j: number) =>
    `${visibleChar(chars[j])} ${back(j) === 0 ? 'current' : `${back(j)} back`} ${formatPct(weights[j])}`;

  return (
    <div className="attn">
      <p className="attn__summary mono">
        {hover !== null ? (
          <>
            <span className="attn__key">hover</span> {describe(hover)}
          </>
        ) : (
          <>
            <span className="attn__key">strongest</span>{' '}
            {strongest.map((j, k) => (
              <span key={j}>
                {k > 0 && <span className="attn__sep">/</span>}
                {describe(j)}
              </span>
            ))}
          </>
        )}
      </p>
      <div
        className="attn__text mono"
        role="img"
        aria-label={`Attention over the last ${n} characters. Strongest: ${strongest
          .map((j) => `${spokenChar(chars[j])}, ${back(j)} back, ${formatPct(weights[j])}`)
          .join('; ')}.`}
        onMouseLeave={() => setHover(null)}
      >
        {chars.slice(0, n).map((c, j) => {
          const a = Math.pow(weights[j] / max, 0.75);
          return (
            <span
              key={j}
              className={`attn__ch${j === n - 1 ? ' attn__ch--query' : ''}${a > 0.55 ? ' attn__ch--hot' : ''}`}
              style={{ '--a': a.toFixed(3) } as CSSProperties}
              onMouseEnter={() => setHover(j)}
            >
              {c === '\n' ? '↵\n' : c === ' ' ? ' ' : c}
            </span>
          );
        })}
        <span className="attn__next" aria-hidden="true">
          {visibleChar(chosen)}
        </span>
      </div>
    </div>
  );
}
