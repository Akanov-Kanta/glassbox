import { memo } from 'react';
import { argmax, maxOf, meanRows } from '../lib/stats';

export type HeadChoice = number | 'avg';

interface Props {
  attention: Float32Array[][];
  layer: number;
  head: HeadChoice;
  onSelect: (layer: number, head: HeadChoice) => void;
}

const W = 120;
const H = 30;

/** Bars over context positions; the right edge is the current character. */
const Spark = memo(function Spark({ row }: { row: Float32Array }) {
  const n = row.length;
  const max = maxOf(row) || 1;
  const step = W / Math.max(n, 1);
  let d = '';
  for (let j = 0; j < n; j++) {
    const h = (row[j] / max) * (H - 2);
    if (h < 0.4) continue;
    const x = (j + 0.5) * step;
    d += `M${x.toFixed(2)} ${H}V${(H - h).toFixed(2)}`;
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true" className="spark">
      <line x1="0" x2={W} y1={H - 0.5} y2={H - 0.5} className="spark__base" />
      <path d={d} className="spark__bars" strokeWidth={Math.max(0.8, Math.min(step * 0.75, 6))} />
    </svg>
  );
});

function peakLabel(row: Float32Array): { short: string; long: string } {
  const back = row.length - 1 - argmax(row);
  if (back === 0) return { short: 'self', long: 'strongest on the current character' };
  return { short: `−${back}`, long: `strongest ${back} character${back === 1 ? '' : 's'} back` };
}

export function HeadGrid({ attention, layer, head, onSelect }: Props) {
  const nHead = attention[0]?.length ?? 0;
  return (
    <div className="heads" role="group" aria-label="Attention of this character for every layer and head">
      <div className="heads__grid" style={{ gridTemplateColumns: `auto repeat(${nHead + 1}, minmax(0, 1fr))` }}>
        <span />
        {Array.from({ length: nHead }, (_, h) => (
          <span key={h} className="heads__col label">
            H{h}
          </span>
        ))}
        <span className="heads__col label">avg</span>
        {attention.map((heads, l) => {
          const cells: { key: HeadChoice; row: Float32Array }[] = [
            ...heads.map((row, h) => ({ key: h as HeadChoice, row })),
            { key: 'avg', row: meanRows(heads) },
          ];
          return (
            <div className="heads__row" key={l} style={{ display: 'contents' }}>
              <span className="heads__rowlabel label">L{l}</span>
              {cells.map(({ key, row }) => {
                const selected = layer === l && head === key;
                const peak = peakLabel(row);
                return (
                  <button
                    type="button"
                    key={String(key)}
                    className={`head${selected ? ' head--on' : ''}${key === 'avg' ? ' head--avg' : ''}`}
                    aria-pressed={selected}
                    aria-label={`Layer ${l}, ${key === 'avg' ? 'average of heads' : `head ${key}`}: ${peak.long}`}
                    onClick={() => onSelect(l, key)}
                  >
                    <Spark row={row} />
                    <span className="head__peak mono">{peak.short}</span>
                  </button>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}
