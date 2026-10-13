import { useEffect, useRef, useState, type PointerEvent } from 'react';
import { formatInt } from '../lib/format';

interface Props {
  steps: number[];
  train: number[];
  val: number[];
}

function niceTicks(min: number, max: number, count: number): number[] {
  const span = max - min || 1;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) out.push(+v.toFixed(10));
  return out;
}

const HEIGHT = 280;
const M = { top: 16, right: 56, bottom: 40, left: 52 };

export function LossChart({ steps, train, val }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(720);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(280, Math.round(entry.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const n = Math.min(steps.length, train.length, val.length);
  if (n < 2) return <p className="label">No training curve in training.json.</p>;

  const right = width < 480 ? 40 : M.right;
  const iw = width - M.left - right;
  const ih = HEIGHT - M.top - M.bottom;
  const x0 = steps[0];
  const x1 = steps[n - 1];
  const all = [...train.slice(0, n), ...val.slice(0, n)].filter(Number.isFinite);
  const yMin = Math.floor(Math.min(...all) * 10) / 10;
  const yMax = Math.ceil(Math.max(...all) * 10) / 10;
  const sx = (v: number) => M.left + ((v - x0) / (x1 - x0 || 1)) * iw;
  const sy = (v: number) => M.top + (1 - (v - yMin) / (yMax - yMin || 1)) * ih;
  const path = (ys: number[]) =>
    ys
      .slice(0, n)
      .map((y, i) => `${i ? 'L' : 'M'}${sx(steps[i]).toFixed(1)} ${sy(y).toFixed(1)}`)
      .join('');
  const xt = niceTicks(x0, x1, width < 480 ? 3 : 6);
  const yt = niceTicks(yMin, yMax, 5);

  // Direct labels at the line ends, nudged apart if they would collide.
  let ty = sy(train[n - 1]);
  let vy = sy(val[n - 1]);
  if (Math.abs(ty - vy) < 14) {
    const mid = (ty + vy) / 2;
    const up = val[n - 1] >= train[n - 1];
    vy = mid + (up ? -7 : 7);
    ty = mid + (up ? 7 : -7);
  }

  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width;
    const target = x0 + ((px - M.left) / iw) * (x1 - x0);
    let best = 0;
    for (let i = 1; i < n; i++) if (Math.abs(steps[i] - target) < Math.abs(steps[best] - target)) best = i;
    setHover(best);
  };

  const hx = hover !== null ? sx(steps[hover]) : 0;
  const tipLeft = hover !== null && hx > width - 170;

  return (
    <figure className="chart">
      <div className="chart__legend mono" aria-hidden="true">
        <span className="chart__key chart__key--train">train</span>
        <span className="chart__key chart__key--val">validation</span>
        <span className="chart__unit">cross-entropy, nats per character</span>
      </div>
      <div ref={ref} className="chart__plot">
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`Training and validation loss over ${formatInt(x1)} steps. Final train loss ${train[n - 1].toFixed(3)}, final validation loss ${val[n - 1].toFixed(3)}.`}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
        >
          {yt.map((v) => (
            <g key={`y${v}`}>
              <line className="chart__grid" x1={M.left} x2={M.left + iw} y1={sy(v)} y2={sy(v)} />
              <text className="chart__tick" x={M.left - 8} y={sy(v)} dy="0.32em" textAnchor="end">
                {v.toFixed(1)}
              </text>
            </g>
          ))}
          {xt.map((v) => (
            <text key={`x${v}`} className="chart__tick" x={sx(v)} y={M.top + ih + 18} textAnchor="middle">
              {v >= 1000 ? `${v / 1000}k` : v}
            </text>
          ))}
          <line className="chart__axis" x1={M.left} x2={M.left + iw} y1={M.top + ih} y2={M.top + ih} />
          <text className="chart__title" x={M.left + iw / 2} y={HEIGHT - 4} textAnchor="middle">
            training step
          </text>
          <text
            className="chart__title"
            transform={`translate(12 ${M.top + ih / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            loss
          </text>
          <path className="chart__line chart__line--train" d={path(train)} />
          <path className="chart__line chart__line--val" d={path(val)} />
          <text className="chart__end chart__end--train" x={M.left + iw + 6} y={ty} dy="0.32em">
            train
          </text>
          <text className="chart__end chart__end--val" x={M.left + iw + 6} y={vy} dy="0.32em">
            val
          </text>
          {hover !== null && (
            <g className="chart__hover">
              <line x1={hx} x2={hx} y1={M.top} y2={M.top + ih} />
              <circle className="chart__dot--train" cx={hx} cy={sy(train[hover])} r={4} />
              <circle className="chart__dot--val" cx={hx} cy={sy(val[hover])} r={4} />
            </g>
          )}
        </svg>
        {hover !== null && (
          <div
            className="chart__tip mono"
            style={{ left: tipLeft ? hx - 12 : hx + 12, transform: tipLeft ? 'translateX(-100%)' : undefined }}
          >
            <div className="chart__tip-step">step {formatInt(steps[hover])}</div>
            <div>
              <span className="chart__key chart__key--train">train</span> {train[hover].toFixed(3)}
            </div>
            <div>
              <span className="chart__key chart__key--val">val</span> {val[hover].toFixed(3)}
            </div>
          </div>
        )}
      </div>
      <details className="chart__table">
        <summary className="label">Show the numbers</summary>
        <div className="chart__table-scroll">
          <table className="mono">
            <thead>
              <tr>
                <th scope="col">step</th>
                <th scope="col">train</th>
                <th scope="col">val</th>
              </tr>
            </thead>
            <tbody>
              {steps.slice(0, n).map((s, i) => (
                <tr key={s}>
                  <td>{formatInt(s)}</td>
                  <td>{train[i].toFixed(4)}</td>
                  <td>{val[i].toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
