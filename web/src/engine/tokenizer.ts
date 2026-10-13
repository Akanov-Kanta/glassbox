// Character-level tokenizer: a token id is an index into the vocabulary.

/** Common typographic characters a plain-text corpus usually spells with ASCII. */
const ASCII_EQUIVALENTS: Record<string, string> = {
  '‘': "'", // left single quote
  '’': "'", // right single quote / apostrophe
  '“': '"', // left double quote
  '”': '"', // right double quote
  '–': '-', // en dash
  '—': '-', // em dash
  ' ': ' ', // no-break space
  '\t': ' ',
  '\r': '', // \r\n -> \n
};

export class CharTokenizer {
  readonly vocab: readonly string[];
  private readonly ids: Map<string, number>;

  constructor(vocab: readonly string[]) {
    this.vocab = vocab;
    this.ids = new Map(vocab.map((ch, id) => [ch, id]));
  }

  /** The id of `ch` if it is in the vocabulary. */
  idOf(ch: string): number | undefined {
    return this.ids.get(ch);
  }

  /**
   * Text to ids, one code point at a time. Characters outside the vocabulary never throw; they map to the
   * closest thing the model knows (see `fallback`).
   */
  encode(text: string): number[] {
    const out: number[] = [];
    for (const ch of text) {
      const id = this.ids.get(ch);
      if (id !== undefined) out.push(id);
      else out.push(...this.fallback(ch));
    }
    return out;
  }

  decode(ids: number[]): string {
    let text = '';
    for (const id of ids) {
      const ch = this.vocab[id];
      if (ch === undefined) throw new RangeError(`token id ${id} is out of range for a vocabulary of ${this.vocab.length}`);
      text += ch;
    }
    return text;
  }

  /**
   * Unknown-character policy, first match wins:
   *  1. a typographic ASCII equivalent (curly quotes, dashes, no-break space, tab; '\r' is dropped);
   *  2. Unicode compatibility decomposition without accents ('é' -> 'e', '…' -> '...', 'ﬁ' -> 'fi'),
   *     if every resulting character is in the vocabulary (a lone combining accent therefore vanishes);
   *  3. the same letter in the other case;
   *  4. a space, if the vocabulary has one, otherwise id 0.
   */
  private fallback(ch: string): number[] {
    const candidates = [
      ASCII_EQUIVALENTS[ch],
      ch.normalize('NFKD').replace(/\p{M}/gu, ''),
      ch.toLowerCase(),
      ch.toUpperCase(),
    ];
    for (const candidate of candidates) {
      if (candidate === undefined || candidate === ch) continue;
      const chars = [...candidate];
      if (chars.every((c) => this.ids.has(c))) return chars.map((c) => this.ids.get(c)!);
    }
    return [this.ids.get(' ') ?? 0];
  }
}
