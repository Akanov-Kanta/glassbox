import { formatBytes } from '../lib/format';
import type { LoadProgress } from '../lib/types';

export function LoadingPanel({ progress }: { progress: LoadProgress | null }) {
  const pct = progress && progress.total > 0 ? Math.min(100, (progress.loaded / progress.total) * 100) : null;
  const text =
    !progress || progress.phase === 'meta'
      ? 'Fetching the model description'
      : progress.phase === 'weights'
        ? `Downloading weights ${formatBytes(progress.loaded)}${progress.total ? ` / ${formatBytes(progress.total)}` : ''}`
        : 'Building the model in memory';
  return (
    <div className="loading" aria-busy="true">
      <div className="loading__row">
        <p className="mono loading__text" aria-live="polite">
          {text}
        </p>
        {pct !== null && <span className="mono loading__pct">{Math.round(pct)}%</span>}
      </div>
      <div
        className="progress"
        role="progressbar"
        aria-label="Model download"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct !== null ? Math.round(pct) : undefined}
      >
        <span className={pct === null ? 'progress__indeterminate' : ''} style={pct !== null ? { width: `${pct}%` } : undefined} />
      </div>
      <div className="loading__skeleton" aria-hidden="true">
        <div className="skeleton skeleton--block" />
        <div className="skeleton skeleton--block skeleton--tall" />
      </div>
    </div>
  );
}

export function LoadError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="load-error" role="alert">
      <h3 className="load-error__title">The model could not be loaded.</h3>
      <p className="mono load-error__msg">{message}</p>
      <p className="load-error__help">
        Check your connection and try again.
      </p>
      <button type="button" className="btn btn-primary" onClick={onRetry}>
        Try again
      </button>
    </div>
  );
}
