import { useCallback, useEffect, useRef, useState } from 'react';
import type { GenStep, Model } from '../engine/types';

export interface GenSettings {
  temperature: number;
  topK: number;
  length: number;
}

export interface Run {
  /** prompt as the model sees it: decode(encode(prompt)), so positions line up with token ids */
  promptText: string;
  promptIds: number[];
  /** true when encode() had to replace or skip characters that are not in the vocabulary */
  adjusted: boolean;
  settings: GenSettings;
  steps: GenStep[];
}

export type GenStatus = 'idle' | 'running' | 'done';

/** Delay between characters when animating; generation itself takes a few ms per character. */
const CHAR_DELAY_MS = 28;

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export function useGeneration(model: Model | null, reducedMotion: boolean) {
  const [run, setRun] = useState<Run | null>(null);
  const [status, setStatus] = useState<GenStatus>('idle');
  const [msPerChar, setMsPerChar] = useState<number | null>(null);
  const runId = useRef(0);

  useEffect(
    () => () => {
      runId.current++;
    },
    [],
  );

  const stop = useCallback(() => {
    runId.current++;
    setStatus((s) => (s === 'running' ? 'done' : s));
  }, []);

  const start = useCallback(
    async (prompt: string, settings: GenSettings) => {
      if (!model) return;
      const id = ++runId.current;
      let promptIds = model.encode(prompt);
      const adjusted = model.decode(promptIds) !== prompt;
      if (promptIds.length === 0) {
        // An empty context has nothing to attend to; start from a newline (or the first token).
        const nl = model.encode('\n');
        promptIds = nl.length ? nl : [0];
      }
      const promptText = model.decode(promptIds);
      const steps: GenStep[] = [];
      setRun({ promptText, promptIds, adjusted, settings, steps: [] });
      setStatus('running');
      setMsPerChar(null);

      const gen = model.generate(promptIds, { temperature: settings.temperature, topK: settings.topK });
      let computeMs = 0;
      try {
        for (let i = 0; i < settings.length; i++) {
          const t0 = performance.now();
          const r = gen.next();
          computeMs += performance.now() - t0;
          if (r.done) break;
          steps.push(r.value);
          const flush = !reducedMotion || i % 24 === 23 || i === settings.length - 1;
          if (flush) {
            setRun((cur) => (cur && runId.current === id ? { ...cur, steps: steps.slice() } : cur));
            setMsPerChar(computeMs / steps.length);
          }
          await sleep(reducedMotion ? 0 : CHAR_DELAY_MS);
          if (runId.current !== id) break;
        }
      } catch (err) {
        console.error('[glassbox] generation failed', err);
      } finally {
        gen.return();
      }
      // Commit whatever was generated, including when stopped early.
      setRun((cur) => (cur && cur.promptIds === promptIds ? { ...cur, steps: steps.slice() } : cur));
      if (steps.length) setMsPerChar(computeMs / steps.length);
      if (runId.current === id) setStatus('done');
    },
    [model, reducedMotion],
  );

  return { run, status, msPerChar, start, stop };
}
