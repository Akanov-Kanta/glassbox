import { memo, useEffect, useRef, type CSSProperties, type KeyboardEvent, type MouseEvent } from 'react';
import type { GenStatus, Run } from '../hooks/useGeneration';
import { formatPct, spokenChar } from '../lib/format';
import { confidence } from '../lib/stats';

interface Props {
  run: Run | null;
  status: GenStatus;
  active: number | null;
  pinned: number | null;
  onHover: (i: number | null) => void;
  onPin: (i: number) => void;
}

const optionId = (i: number) => `gen-char-${i}`;

const Char = memo(function Char({
  i,
  char,
  p,
  active,
  pinned,
}: {
  i: number;
  char: string;
  p: number;
  active: boolean;
  pinned: boolean;
}) {
  const c = confidence(p);
  const cls = ['ch', p < 0.05 ? 'ch--rare' : '', active ? 'ch--active' : '', pinned ? 'ch--pinned' : '']
    .filter(Boolean)
    .join(' ');
  return (
    <span
      id={optionId(i)}
      role="option"
      aria-selected={pinned}
      aria-label={`${spokenChar(char)}, ${formatPct(p)}`}
      data-i={i}
      className={cls}
      style={{ '--c': c } as CSSProperties}
    >
      {char === '\n' ? (
        <>
          <span className="ch__nl" aria-hidden="true">
            ↵
          </span>
          {'\n'}
        </>
      ) : (
        char
      )}
    </span>
  );
});

function indexFromEvent(e: MouseEvent): number | null {
  const el = (e.target as HTMLElement).closest<HTMLElement>('[data-i]');
  return el ? Number(el.dataset.i) : null;
}

export function OutputStream({ run, status, active, pinned, onHover, onPin }: Props) {
  const boxRef = useRef<HTMLDivElement>(null);
  const steps = run?.steps ?? [];
  const count = steps.length;

  // Keep the newest characters in view while writing, unless the reader scrolled up.
  useEffect(() => {
    const box = boxRef.current;
    if (!box || status !== 'running') return;
    const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
    if (nearBottom) box.scrollTop = box.scrollHeight;
  }, [count, status]);

  // Keyboard selection scrolls the selected character into view.
  useEffect(() => {
    if (pinned === null) return;
    document.getElementById(optionId(pinned))?.scrollIntoView({ block: 'nearest' });
  }, [pinned]);

  const onKeyDown = (e: KeyboardEvent) => {
    if (!count) return;
    const cur = pinned ?? active ?? count - 1;
    let next: number | null = null;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = Math.min(count - 1, cur + 1);
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = Math.max(0, cur - 1);
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = count - 1;
    else if (e.key === 'PageDown') next = Math.min(count - 1, cur + 20);
    else if (e.key === 'PageUp') next = Math.max(0, cur - 20);
    if (next !== null) {
      e.preventDefault();
      onPin(next);
    }
  };

  return (
    <div
      ref={boxRef}
      className="output mono"
      role="listbox"
      tabIndex={count ? 0 : -1}
      aria-label="Generated characters. Use the arrow keys to inspect each one."
      aria-activedescendant={pinned !== null && pinned < count ? optionId(pinned) : undefined}
      aria-busy={status === 'running'}
      onKeyDown={onKeyDown}
      onMouseOver={(e) => onHover(indexFromEvent(e))}
      onMouseLeave={() => onHover(null)}
      onClick={(e) => {
        const i = indexFromEvent(e);
        if (i !== null) {
          onPin(i);
          boxRef.current?.focus({ preventScroll: true });
        }
      }}
    >
      {run ? (
        <>
          <span className="output__prompt" aria-hidden="true">
            {run.promptText}
          </span>
          {steps.map((s, i) => (
            <Char key={i} i={i} char={s.char} p={s.probs[s.id]} active={active === i} pinned={pinned === i} />
          ))}
          {status === 'running' && <span className="caret" aria-hidden="true" />}
        </>
      ) : (
        <span className="output__empty">
          Output appears here, one character at a time. Each character is tinted by how likely the model
          thought it was.
        </span>
      )}
    </div>
  );
}
