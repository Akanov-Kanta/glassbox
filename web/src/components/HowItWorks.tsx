import type { ReactNode } from 'react';
import type { Preview } from '../lib/loader';
import { deviceLabel, formatBytes, formatDuration, formatInt, formatParams } from '../lib/format';
import { REPO_URL } from '../lib/links';
import { ArchitectureDiagram } from './ArchitectureDiagram';
import { LossChart } from './LossChart';

interface Props {
  preview: Preview | null;
  weightBytes: number | null;
}

export function HowItWorks({ preview, weightBytes }: Props) {
  if (!preview) {
    return (
      <section className="section wrap" id="how" aria-labelledby="how-title" aria-busy="true">
        <h2 id="how-title" className="section__title">
          How it works
        </h2>
        <div className="skeleton skeleton--block" />
      </section>
    );
  }
  const { config, training, paramCount } = preview;
  const { corpus } = training;
  const lastStep = training.steps[training.steps.length - 1];
  const bytes = weightBytes ?? paramCount * 4;
  const optimizer = training.optimizer ?? 'AdamW';

  const pipeline: { title: string; body: ReactNode }[] = [
    {
      title: 'Corpus',
      body: (
        <>
          {corpus.source_url ? (
            <a href={corpus.source_url} target="_blank" rel="noreferrer">
              {corpus.name}
            </a>
          ) : (
            corpus.name
          )}
          {corpus.chars ? `, ${formatInt(corpus.chars)} characters` : ''}
          {corpus.license ? ` (${corpus.license})` : ''}. Split into train and validation text.
        </>
      ),
    },
    {
      title: 'Character tokenizer',
      body: (
        <>
          Every distinct character is one token: a vocabulary of {config.vocab.length}. No BPE, so each step of
          generation is exactly one visible character.
        </>
      ),
    },
    {
      title: `${config.nLayer}-layer transformer`,
      body: (
        <>
          GPT-2 style, pre-LayerNorm: {config.nLayer} blocks, {config.nHead} attention heads, width {config.nEmbd},
          a {config.blockSize}-character context. {formatParams(paramCount)} parameters, input and output embeddings
          tied.
        </>
      ),
    },
    {
      title: `Trained with ${optimizer}`,
      body: (
        <>
          PyTorch on {deviceLabel(training.device)}
          {lastStep ? ` for ${formatInt(lastStep)} steps` : ''}, {formatDuration(training.train_seconds)} wall
          clock. Final loss {training.final.train_loss.toFixed(3)} train, {training.final.val_loss.toFixed(3)}{' '}
          validation.
        </>
      ),
    },
    {
      title: 'Exported to one flat file',
      body: (
        <>
          {training.export?.dtype === 'f16' ? (
            <>
              Every weight rounded to float16 and concatenated: {formatBytes(bytes)}, half the size of float32,
              widened back to float32 in your browser. Full validation loss{' '}
              {training.export.val_loss_f32.toFixed(4) === training.export.val_loss_exported.toFixed(4)
                ? `is unchanged by the rounding (${training.export.val_loss_f32.toFixed(4)})`
                : `goes from ${training.export.val_loss_f32.toFixed(4)} to ${training.export.val_loss_exported.toFixed(4)}`}
              . A small JSON index holds the names, shapes and offsets.
            </>
          ) : (
            <>
              Every tensor concatenated as little-endian float32: {formatBytes(bytes)}, described by a small JSON
              index of names, shapes and offsets.
            </>
          )}
        </>
      ),
    },
    {
      title: 'Hand-written TypeScript engine',
      body: (
        <>
          Matmuls, LayerNorm, GELU and causal softmax attention written from scratch, checked against PyTorch
          to 1e-4. A KV cache means each new character costs one position, not a pass over the whole context.
          It all runs in this tab.
        </>
      ),
    },
  ];

  return (
    <section className="section wrap" id="how" aria-labelledby="how-title">
      <h2 id="how-title" className="section__title">
        How it works
      </h2>
      <p className="section__lede">
        From raw text to a model you can poke at, with no ML framework at inference time.{' '}
        <a href={REPO_URL} target="_blank" rel="noreferrer">
          The full source is on GitHub.
        </a>
      </p>

      <div className="how">
        <ol className="pipeline">
          {pipeline.map((s) => (
            <li key={s.title} className="pipeline__step">
              <h3 className="pipeline__title">{s.title}</h3>
              <p className="pipeline__body">{s.body}</p>
            </li>
          ))}
        </ol>
        <ArchitectureDiagram config={config} />
      </div>

      <div className="loss">
        <h3 className="loss__title">Training curve</h3>
        <LossChart steps={training.steps} train={training.train_loss} val={training.val_loss} />
      </div>
    </section>
  );
}
