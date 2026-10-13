import type { CSSProperties } from 'react';
import type { GenSettings } from '../hooks/useGeneration';
import { visibleChar } from '../lib/format';

interface Props {
  prompt: string;
  onPrompt: (v: string) => void;
  examples: string[];
  settings: GenSettings;
  onSettings: (s: GenSettings) => void;
  running: boolean;
  disabled: boolean;
  onGenerate: () => void;
  onStop: () => void;
}

function Slider({
  id,
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="slider">
      <div className="slider__top">
        <label htmlFor={id} className="label">
          {label}
        </label>
        <output htmlFor={id} className="slider__val mono">
          {display}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ '--fill': `${((value - min) / (max - min)) * 100}%` } as CSSProperties}
      />
    </div>
  );
}

/** Example prompt shown on a chip: newlines become a visible glyph. */
function chipText(s: string): string {
  return s.replace(/\n/g, ` ${visibleChar('\n')} `).trim();
}

export function Controls({
  prompt,
  onPrompt,
  examples,
  settings,
  onSettings,
  running,
  disabled,
  onGenerate,
  onStop,
}: Props) {
  return (
    <form
      className="controls"
      onSubmit={(e) => {
        e.preventDefault();
        if (!running) onGenerate();
      }}
    >
      <div className="prompt">
        <label htmlFor="prompt" className="label">
          prompt
        </label>
        <textarea
          id="prompt"
          className="prompt__input mono"
          value={prompt}
          rows={2}
          spellCheck={false}
          aria-describedby="prompt-help"
          onChange={(e) => onPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              if (!running && !disabled) onGenerate();
            }
          }}
        />
        <p id="prompt-help" className="prompt__help">
          The model continues this text. Characters it never saw in training are replaced or skipped.{' '}
          <span className="kbd">Ctrl</span>+<span className="kbd">Enter</span> to generate.
        </p>
        {examples.length > 0 && (
          <div className="examples" role="group" aria-label="Example prompts">
            <span className="label">try</span>
            {examples.map((ex) => (
              <button type="button" key={ex} className="chip mono" onClick={() => onPrompt(ex)} disabled={running}>
                {chipText(ex)}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="sliders">
        <Slider
          id="temperature"
          label="temperature"
          value={settings.temperature}
          min={0.1}
          max={2}
          step={0.05}
          display={settings.temperature.toFixed(2)}
          onChange={(v) => onSettings({ ...settings, temperature: v })}
        />
        <Slider
          id="topk"
          label="top-k"
          value={settings.topK}
          min={0}
          max={40}
          step={1}
          display={settings.topK === 0 ? 'off' : String(settings.topK)}
          onChange={(v) => onSettings({ ...settings, topK: v })}
        />
        <Slider
          id="length"
          label="length"
          value={settings.length}
          min={50}
          max={500}
          step={10}
          display={`${settings.length} chars`}
          onChange={(v) => onSettings({ ...settings, length: v })}
        />
      </div>

      <div className="actions">
        <button type="submit" className="btn btn-primary" disabled={disabled || running}>
          Generate
        </button>
        <button type="button" className="btn" disabled={!running} onClick={onStop}>
          Stop
        </button>
      </div>
    </form>
  );
}
