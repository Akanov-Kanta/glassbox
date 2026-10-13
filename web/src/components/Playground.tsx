import { useCallback, useMemo, useState } from 'react';
import type { Model } from '../engine/types';
import { useGeneration, type GenSettings } from '../hooks/useGeneration';
import type { LoaderState } from '../hooks/useModelLoader';
import { useReducedMotion } from '../hooks/useReducedMotion';
import type { TrainingReport } from '../lib/types';
import { Controls } from './Controls';
import type { HeadChoice } from './HeadGrid';
import { Inspector } from './Inspector';
import { LoadError, LoadingPanel } from './LoadState';
import { OutputStream } from './OutputStream';
import { examplePrompts } from '../lib/examples';

interface ReadyProps {
  model: Model;
  training: TrainingReport;
}

function PlaygroundReady({ model, training }: ReadyProps) {
  const reducedMotion = useReducedMotion();
  const { run, status, msPerChar, start, stop } = useGeneration(model, reducedMotion);
  const examples = useMemo(() => examplePrompts(training.samples ?? [], model), [training, model]);
  const [prompt, setPrompt] = useState(() => examples[0] ?? '');
  const [settings, setSettings] = useState<GenSettings>({ temperature: 0.8, topK: 20, length: 240 });
  const [pinned, setPinned] = useState<number | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [layer, setLayer] = useState(model.config.nLayer - 1);
  const [head, setHead] = useState<HeadChoice>('avg');

  const count = run?.steps.length ?? 0;
  const active = hover ?? pinned ?? (count ? count - 1 : null);

  const generate = useCallback(() => {
    setPinned(null);
    setHover(null);
    void start(prompt, settings);
  }, [prompt, settings, start]);

  const onLayerHead = useCallback((l: number, h: HeadChoice) => {
    setLayer(l);
    setHead(h);
  }, []);

  return (
    <div className="pg">
      <div className="pg__main">
        <Controls
          prompt={prompt}
          onPrompt={setPrompt}
          examples={examples}
          settings={settings}
          onSettings={setSettings}
          running={status === 'running'}
          disabled={false}
          onGenerate={generate}
          onStop={stop}
        />
        <div className="out">
          <div className="out__bar">
            <p className="out__status mono">
              {status === 'running'
                ? `writing ${count} / ${run?.settings.length ?? 0}`
                : status === 'done'
                  ? `${count} characters`
                  : 'ready'}
              {msPerChar !== null && (
                <span className="out__perf"> engine {msPerChar.toFixed(1)} ms/char</span>
              )}
            </p>
            <div className="legend mono" aria-hidden="true">
              <span>surprising</span>
              <span className="legend__ramp" />
              <span>confident</span>
            </div>
          </div>
          <p className="visually-hidden" aria-live="polite">
            {status === 'running' ? 'Generating.' : status === 'done' ? `Done: ${count} characters generated.` : ''}
          </p>
          <OutputStream
            run={run}
            status={status}
            active={active}
            pinned={pinned}
            onHover={setHover}
            onPin={setPinned}
          />
          {run?.adjusted && (
            <p className="out__note">
              Some prompt characters are not in the model's vocabulary, so they were replaced or skipped. The
              grey text above is what the model actually read.
            </p>
          )}
          <p className="out__hint">
            Hover or click a character to inspect it. With the output focused, <span className="kbd">←</span>{' '}
            <span className="kbd">→</span> step through characters.
          </p>
        </div>
      </div>
      <Inspector
        run={run}
        index={active}
        pinned={pinned !== null && hover === null}
        config={model.config}
        layer={layer}
        head={head}
        onLayerHead={onLayerHead}
      />
    </div>
  );
}

interface Props {
  state: LoaderState;
  onRetry: () => void;
}

export function Playground({ state, onRetry }: Props) {
  return (
    <section className="section wrap" id="playground" aria-labelledby="pg-title">
      <h2 id="pg-title" className="section__title">
        Watch it write
      </h2>
      <p className="section__lede">
        Every character below is sampled live in your browser. Pick one to see the candidates the model
        weighed and where each attention head was looking when it chose.
      </p>
      {state.status === 'ready' ? (
        <PlaygroundReady model={state.app.model} training={state.app.training} />
      ) : state.status === 'error' ? (
        <LoadError message={state.error} onRetry={onRetry} />
      ) : (
        <LoadingPanel progress={state.progress} />
      )}
    </section>
  );
}
