import type { Preview } from '../lib/loader';
import { deviceLabel, formatDuration, formatInt, formatParams } from '../lib/format';
import { REPO_URL } from '../lib/links';

interface Props {
  preview: Preview | null;
}

interface Fact {
  label: string;
  value: string;
  detail?: string;
}

function facts(p: Preview): Fact[] {
  const { config, training, paramCount } = p;
  const lastStep = training.steps[training.steps.length - 1];
  const val = training.final.val_loss;
  return [
    { label: 'parameters', value: formatParams(paramCount), detail: formatInt(paramCount) },
    {
      label: 'layers × heads × width',
      value: `${config.nLayer} × ${config.nHead} × ${config.nEmbd}`,
      detail: `${config.nEmbd / config.nHead} dims per head`,
    },
    { label: 'context', value: `${config.blockSize} chars`, detail: `${config.vocab.length}-symbol vocabulary` },
    {
      label: 'corpus',
      value: training.corpus.name,
      detail: training.corpus.chars ? `${formatInt(training.corpus.chars)} characters` : undefined,
    },
    {
      label: 'training',
      value: formatDuration(training.train_seconds),
      detail: `${deviceLabel(training.device)}${lastStep ? `, ${formatInt(lastStep)} steps` : ''}`,
    },
    {
      label: 'final val loss',
      value: val.toFixed(3),
      detail: `nats/char (${(val / Math.LN2).toFixed(2)} bits)`,
    },
  ];
}

export function Hero({ preview }: Props) {
  const list = preview ? facts(preview) : null;
  return (
    <section className="hero wrap" id="top" aria-labelledby="hero-title">
      <div className="hero__copy">
        <h1 id="hero-title" className="hero__title">
          Glassbox
        </h1>
        <p className="hero__pitch">A GPT you can see through.</p>
        <p className="hero__lede">
          A character-level transformer trained from scratch, running in this tab on an inference engine
          hand-written in TypeScript. Zero ML libraries, no server.
        </p>
        <div className="hero__ctas">
          <a className="btn btn-primary" href="#playground">
            Make it write
          </a>
          <a className="btn" href={REPO_URL} target="_blank" rel="noreferrer">
            Read the source
          </a>
        </div>
      </div>

      <dl className="facts" aria-busy={!list}>
        {list
          ? list.map((f) => (
              <div className="fact" key={f.label}>
                <dt className="label">{f.label}</dt>
                <dd>
                  <span className="fact__value">{f.value}</span>
                  {f.detail && <span className="fact__detail mono">{f.detail}</span>}
                </dd>
              </div>
            ))
          : Array.from({ length: 6 }, (_, i) => (
              <div className="fact fact--skeleton" key={i}>
                <dt className="skeleton skeleton--label" />
                <dd className="skeleton skeleton--value" />
              </div>
            ))}
      </dl>
    </section>
  );
}
