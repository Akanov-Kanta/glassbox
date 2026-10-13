import type { LoadModel, Model, ModelConfig } from '../engine/types';
import type { DataSource, LoadedApp, LoadProgress, ModelMeta, TrainingReport } from './types';

/**
 * The engine lives in web/src/engine/index.ts and is written in parallel with the UI. `import.meta.glob`
 * resolves to an empty object when the file does not exist yet (instead of failing the build), so the
 * app compiles either way and picks the real engine up automatically once it exists.
 */
interface EngineModule {
  loadModel?: LoadModel;
  createModel?: (meta: ModelMeta, buffer: ArrayBuffer) => Model | Promise<Model>;
}
const engineImports = import.meta.glob<EngineModule>('../engine/index.ts');
const loadEngine = engineImports['../engine/index.ts'] as (() => Promise<EngineModule>) | undefined;

const MODEL_DIR = `${import.meta.env.BASE_URL}model/`;

export interface Preview {
  config: ModelConfig;
  paramCount: number;
  training: TrainingReport;
  modelSource: DataSource;
  trainingSource: DataSource;
}

export interface LoadCallbacks {
  onPreview?: (p: Preview) => void;
  onProgress?: (p: LoadProgress) => void;
  signal?: AbortSignal;
}

/** Fetches JSON; resolves to null when the file is missing (404, or the dev server's HTML fallback). */
async function fetchJsonOrNull<T>(url: string, signal?: AbortSignal): Promise<T | null> {
  const res = await fetch(url, { signal });
  if (!res.ok) return null;
  const type = res.headers.get('content-type') ?? '';
  if (type.includes('text/html')) return null;
  try {
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

function metaToConfig(meta: ModelMeta): ModelConfig {
  return {
    vocab: meta.vocab,
    nLayer: meta.n_layer,
    nHead: meta.n_head,
    nEmbd: meta.n_embd,
    blockSize: meta.block_size,
  };
}

/** Bytes per stored value for the weight file's dtype (SPEC.md: "f32" or "f16"). */
export const bytesPerValue = (meta: ModelMeta): number => (meta.dtype === 'f16' ? 2 : 4);

/** Number of values the weight file must contain, from the tensor table. */
export function expectedFloats(meta: ModelMeta): number {
  let end = 0;
  for (const p of meta.params) {
    const size = p.shape.reduce((a, b) => a * b, 1);
    end = Math.max(end, p.offset + size);
  }
  return end;
}

function paramsFromMeta(meta: ModelMeta): number {
  return meta.params.reduce((sum, p) => sum + p.shape.reduce((a, b) => a * b, 1), 0);
}

/** Downloads a binary file, reporting progress. `expectedBytes` is used when Content-Length is absent. */
async function fetchWithProgress(
  url: string,
  expectedBytes: number,
  onProgress: (loaded: number, total: number) => void,
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`Could not download the weights (${url}: HTTP ${res.status}).`);
  const lengthHeader = Number(res.headers.get('content-length'));
  const encoded = (res.headers.get('content-encoding') ?? 'identity') !== 'identity';
  const total = expectedBytes > 0 ? expectedBytes : !encoded && lengthHeader > 0 ? lengthHeader : 0;
  if (!res.body) {
    const buf = await res.arrayBuffer();
    onProgress(buf.byteLength, buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  let out = new Uint8Array(total > 0 ? total : 1 << 20);
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.byteLength > out.byteLength) {
      const grown = new Uint8Array(Math.max(out.byteLength * 2, loaded + value.byteLength));
      grown.set(out.subarray(0, loaded));
      out = grown;
    }
    out.set(value, loaded);
    loaded += value.byteLength;
    onProgress(loaded, Math.max(total, loaded));
  }
  return out.buffer.slice(0, loaded);
}

async function loadMock(training: TrainingReport | null, cb: LoadCallbacks): Promise<LoadedApp> {
  const [{ createMockModel, MOCK_TEXT }, { createMockTraining }] = await Promise.all([
    import('../dev/mockModel'),
    import('../dev/mockTraining'),
  ]);
  const model = createMockModel();
  const report = training ?? createMockTraining(model.paramCount, MOCK_TEXT.length);
  const trainingSource: DataSource = training ? 'real' : 'mock';
  cb.onPreview?.({ config: model.config, paramCount: model.paramCount, training: report, modelSource: 'mock', trainingSource });
  return { model, training: report, modelSource: 'mock', trainingSource, weightBytes: model.paramCount * 4 };
}

/**
 * Loads the real model when both the engine module and /model/model.json exist; otherwise falls back to
 * the development mock (and the UI shows a "demo data" badge). Once the real files are present, any
 * failure is a real error and is surfaced, not hidden behind the mock.
 */
export async function loadApp(cb: LoadCallbacks = {}): Promise<LoadedApp> {
  const { signal } = cb;
  cb.onProgress?.({ phase: 'meta', loaded: 0, total: 0 });
  const [meta, training] = await Promise.all([
    fetchJsonOrNull<ModelMeta>(`${MODEL_DIR}model.json`, signal),
    fetchJsonOrNull<TrainingReport>(`${MODEL_DIR}training.json`, signal),
  ]);

  if (!meta || !loadEngine) {
    if (import.meta.env.DEV) {
      console.info(`[glassbox] using demo model (${!loadEngine ? 'engine module missing' : 'model.json missing'})`);
    }
    return loadMock(training, cb);
  }

  const config = metaToConfig(meta);
  const bytes = expectedFloats(meta) * bytesPerValue(meta);
  const trainingSource: DataSource = training ? 'real' : 'mock';
  let report = training;
  if (!report) {
    const { createMockTraining } = await import('../dev/mockTraining');
    report = createMockTraining(paramsFromMeta(meta), 0);
  }
  cb.onPreview?.({ config, paramCount: paramsFromMeta(meta), training: report, modelSource: 'real', trainingSource });
  cb.onProgress?.({ phase: 'weights', loaded: 0, total: bytes });

  const [engine, buffer] = await Promise.all([
    loadEngine(),
    fetchWithProgress(
      `${MODEL_DIR}model.bin`,
      bytes,
      (loaded, total) => cb.onProgress?.({ phase: 'weights', loaded, total }),
      signal,
    ),
  ]);
  if (buffer.byteLength < bytes) {
    throw new Error(`The weight file is truncated: got ${buffer.byteLength} bytes, expected ${bytes}.`);
  }
  cb.onProgress?.({ phase: 'init', loaded: buffer.byteLength, total: buffer.byteLength });

  let model: Model;
  if (engine.createModel) {
    model = await engine.createModel(meta, buffer);
  } else if (engine.loadModel) {
    // The browser cache normally serves this second request.
    model = await engine.loadModel(`${MODEL_DIR}model.json`, `${MODEL_DIR}model.bin`);
  } else {
    throw new Error('The engine module exports neither createModel nor loadModel.');
  }
  return { model, training: report, modelSource: 'real', trainingSource, weightBytes: buffer.byteLength };
}
