import type { Model } from '../engine/types';

/** Up to three short openings taken from the model's own unconditional samples. */
export function examplePrompts(samples: string[], model: Model): string[] {
  const out: string[] = [];
  for (const raw of samples) {
    const s = raw.replace(/^\s+/, '');
    if (!s) continue;
    let cut = s.slice(0, 34);
    const nl = cut.indexOf('\n', 8);
    if (nl > 0) cut = cut.slice(0, nl + 1);
    else if (s.length > 34) {
      const sp = cut.lastIndexOf(' ');
      if (sp > 8) cut = cut.slice(0, sp + 1);
    }
    // Keep only characters the model knows, so the chip never silently changes.
    cut = model.decode(model.encode(cut));
    if (cut.trim() && !out.includes(cut)) out.push(cut);
    if (out.length === 3) break;
  }
  return out;
}
