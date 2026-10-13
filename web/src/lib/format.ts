/** Visible glyph for a character, so whitespace and control characters can be seen. */
export function visibleChar(c: string): string {
  if (c === ' ') return '␣';
  if (c === '\n') return '↵';
  if (c === '\t') return '⇥';
  if (c === '\r') return '␍';
  if (c === '') return '∅';
  return c;
}

/** Human description of a character for screen readers. */
export function spokenChar(c: string): string {
  if (c === ' ') return 'space';
  if (c === '\n') return 'newline';
  if (c === '\t') return 'tab';
  return c;
}

export function formatParams(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 1 : 2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

export function formatInt(n: number): string {
  return n.toLocaleString('en-US');
}

export function formatBytes(n: number): string {
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return 'n/a';
  if (seconds < 90) return `${Math.round(seconds)} s`;
  const m = Math.round(seconds / 60);
  if (m < 90) return `${m} min`;
  const h = Math.floor(m / 60);
  return `${h} h ${m % 60} min`;
}

export function formatPct(p: number): string {
  const v = p * 100;
  if (v >= 99.95) return '100%';
  if (v >= 10) return `${v.toFixed(1)}%`;
  if (v >= 0.1) return `${v.toFixed(2)}%`;
  if (v === 0) return '0%';
  return '<0.1%';
}

export function deviceLabel(device: string): string {
  const d = device.toLowerCase();
  if (d === 'mps') return 'Apple MPS';
  if (d.startsWith('cuda')) return 'CUDA GPU';
  if (d === 'cpu') return 'CPU';
  return device;
}
