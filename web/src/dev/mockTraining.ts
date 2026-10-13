/**
 * Placeholder training report with the same shape as web/public/model/training.json. Only used when the
 * real report is unavailable; every number here is fake and the UI labels it as demo data.
 */
import type { TrainingReport } from '../lib/types';

export function createMockTraining(paramCount: number, corpusChars: number): TrainingReport {
  const steps: number[] = [];
  const train: number[] = [];
  const val: number[] = [];
  let seed = 7;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
  for (let s = 0; s <= 5000; s += 125) {
    steps.push(s);
    const t = 1.32 + 2.9 * Math.exp(-s / 520) + 0.25 * Math.exp(-s / 2600);
    train.push(+(t + (rand() - 0.5) * 0.04).toFixed(4));
    val.push(+(t + 0.11 * (1 - Math.exp(-s / 1500)) + (rand() - 0.5) * 0.03).toFixed(4));
  }
  return {
    corpus: { name: 'Demo corpus (placeholder)', source_url: '', license: 'n/a', chars: corpusChars },
    params: paramCount,
    device: 'mps',
    train_seconds: 540,
    steps,
    train_loss: train,
    val_loss: val,
    final: { train_loss: train[train.length - 1], val_loss: val[val.length - 1] },
    samples: [
      'Қараңғы түнде тау қалғып,\nҰйқыға кетер балбырап.',
      'Әсем ән мен сөз ғана көңілге жақын,',
      'To be, or not to be, that is the question.',
    ],
  };
}
