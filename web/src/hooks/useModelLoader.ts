import { useCallback, useEffect, useState } from 'react';
import { loadApp, type Preview } from '../lib/loader';
import type { LoadedApp, LoadProgress } from '../lib/types';

export type LoaderState =
  | { status: 'loading'; progress: LoadProgress | null; preview: Preview | null }
  | { status: 'ready'; app: LoadedApp; preview: Preview | null }
  | { status: 'error'; error: string; preview: Preview | null };

export function useModelLoader(): { state: LoaderState; retry: () => void } {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<LoaderState>({ status: 'loading', progress: null, preview: null });

  useEffect(() => {
    const controller = new AbortController();
    let preview: Preview | null = null;
    loadApp({
      signal: controller.signal,
      onPreview: (p) => {
        preview = p;
        if (!controller.signal.aborted) setState((s) => ({ ...s, preview: p }));
      },
      onProgress: (progress) => {
        if (!controller.signal.aborted) {
          setState((s) => (s.status === 'loading' ? { ...s, progress } : s));
        }
      },
    })
      .then((app) => {
        if (!controller.signal.aborted) setState({ status: 'ready', app, preview });
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        const message = err instanceof Error ? err.message : String(err);
        console.error('[glassbox] failed to load the model', err);
        setState({ status: 'error', error: message, preview });
      });
    return () => controller.abort();
  }, [attempt]);

  const retry = useCallback(() => {
    setState((s) => ({ status: 'loading', progress: null, preview: s.preview }));
    setAttempt((a) => a + 1);
  }, []);

  return { state, retry };
}
