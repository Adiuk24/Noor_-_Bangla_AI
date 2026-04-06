# Borno Tokenizer — Design Spec

**বর্ণ (Borno)** — "letter/alphabet" in Bangla. Custom 64K BPE tokenizer for the Noor model family.

**Goal**: Bangla-native tokenization with English+code coverage. Bangla text at 1.5-2.5 tokens/word (not 10-15x like standard tokenizers). 10x faster than tiktoken at inference.

---

## 1. Vocab Layout

| ID Range | Count | Purpose |
|----------|-------|---------|
| 0-255 | 256 | Raw byte fallback (every UTF-8 byte encodable) |
| 256-275 | 20 | Special tokens (ADE-aware) |
| 276-2999 | 2724 | Reserved (future ADE/control tokens) |
| 3000-63999 | 61000 | BPE merges (all languages, frequency-ordered) |

**Total: 64,000 tokens**

### Special Tokens (IDs 256-275)

| ID | Token | Purpose |
|----|-------|---------|
| 256 | `<bos>` | Beginning of sequence |
| 257 | `<eos>` | End of sequence |
| 258 | `<pad>` | Padding |
| 259 | `<unk>` | Unknown (should never fire with byte fallback) |
| 260 | `<user>` | User turn delimiter |
| 261 | `<assistant>` | Assistant turn delimiter |
| 262 | `<system>` | System prompt delimiter |
| 263 | `<tool_call>` | ADE P2 tool invocation start |
| 264 | `<tool_result>` | ADE P2 tool result start |
| 265 | `<think>` | Reasoning start |
| 266 | `</think>` | Reasoning end |
| 267 | `<memory>` | ADE P3 memory start |
| 268 | `</memory>` | ADE P3 memory end |
| 269 | `<code>` | Code block start |
| 270 | `</code>` | Code block end |
| 271-275 | reserved | Future ADE programs |

## 2. Bangla-Native Pre-tokenization

Three layers ensure Bangla is treated as a first-class language:

### Layer 1: Unicode NFC Normalization

All input text is NFC-normalized before tokenization. Bangla has multiple compositions for the same character (e.g., "ো" can be two codepoints ে + া or a single precomposed form). NFC canonicalizes these so identical text always produces identical tokens.

### Layer 2: Grapheme Cluster Segmentation

Before BPE processes Bangla text, it is segmented into grapheme clusters (aksharas). BPE can merge whole clusters into larger units but never splits a cluster.

**Rule**: `[consonant] ([hasanta][consonant])* [vowel_sign|anusvara|visarga|chandrabindu]?`

Examples:
- `ক্ষ` (ksha) → one pre-token, never split
- `স্ত্র` (stra) → one pre-token (3 consonants joined by hasanta)
- `কি` (ki) → one pre-token (consonant + vowel sign)
- `বাংলা` → `বা` + `ং` + `লা` (3 grapheme clusters)

### Layer 3: Script-Aware Routing

Input text is split into spans by script:
- **Bangla spans** (Bengali Unicode block U+0980-U+09FF): NFC → grapheme segmentation → BPE
- **Latin/code spans**: GPT-4-style regex pre-tokenization → BPE
- **Shared**: numbers, punctuation, whitespace handled uniformly

Both paths feed into the same BPE merge table.

## 3. Training Strategy

### Corpus

| Source | Language | Target Size | Method |
|--------|----------|-------------|--------|
| FineWeb sample | English | ~400MB | HuggingFace datasets download |
| CC-100 Bangla | Bangla | ~350MB | Direct download |
| The Stack sample | Code (Python/JS/Rust) | ~250MB | HuggingFace sample |

**Total: ~1GB**. Bangla deliberately oversampled to ~35% (vs ~0.5% of natural web data).

### Training Tool

HuggingFace `tokenizers` Rust crate. Configured with:
- BPE model type
- Byte-level fallback (bytes 0-255 as base vocab)
- Custom pre-tokenizer chain: NFC → script detection → Bangla grapheme / Latin regex
- 61,000 merge operations (filling IDs 3000-63999)
- Special tokens added post-training at fixed IDs

Training time: <30 seconds on 1GB corpus.

### Validation Metric

**Fertility** = tokens per word on a held-out test set.

| Language | Target Fertility | Fail Threshold |
|----------|-----------------|----------------|
| English | 1.2-1.5 | > 2.0 |
| Bangla | 1.5-2.5 | > 3.0 |
| Python code | 2.0-3.0 | > 4.0 |

If any language exceeds fail threshold, adjust corpus ratios and retrain.

## 4. Inference Encoder

The `bpe` crate (rs-bpe) implements backtracking BPE encoding:
- Loads the trained merge table
- Produces optimal BPE segmentation (not greedy — backtracking finds the true optimal merge sequence)
- 10x faster than tiktoken, ~8MB RAM for vocab data
- Encoding speed target: 2-5M tokens/sec single-threaded

## 5. Crate Structure

```
crates/borno/
├── Cargo.toml
├── src/
│   ├── lib.rs              — pub struct Borno { encode(), decode(), vocab_size(), train() }
│   ├── trainer.rs           — BPE training via HF tokenizers crate
│   ├── encoder.rs           — rs-bpe backtracking fast encoder
│   ├── bangla.rs            — NFC normalization + grapheme cluster segmentation
│   ├── pretokenize.rs       — Script detection + routing (Bangla vs Latin/code)
│   ├── vocab.rs             — Merge table I/O, special token registry, byte fallback
│   └── bin/
│       └── borno_train.rs   — CLI: corpus download, train, export, validate fertility
└── tests/
    ├── bangla_tests.rs      — Grapheme preservation, fertility, NFC consistency
    └── roundtrip_tests.rs   — Encode/decode roundtrip for all scripts + edge cases
```

### Dependencies

| Crate | Purpose |
|-------|---------|
| `bpe` | rs-bpe backtracking encoder |
| `tokenizers` | HF BPE trainer |
| `unicode-normalization` | NFC normalization |
| `unicode-segmentation` | Grapheme cluster boundaries (UAX #29) |
| `reqwest` | Corpus download |
| `serde` / `serde_json` | Vocab serialization |
| `clap` | CLI for borno_train |
| `rayon` | Parallel encoding (optional) |

## 6. Integration with Noor

1. `noor-core` adds `borno` as workspace dependency
2. `NoorTokenizer` in `tokenizer.rs` wraps `Borno` — same API, new backend
3. All config files updated: `vocab_size = 64000`
4. Data shards re-preprocessed with Borno encoding
5. Proxy model retrained from scratch with 64K vocab

### Config Changes

- `proxy.toml`: `vocab_size = 32000` → `64000`
- `edge.toml`: already `64000` (no change)
- `pro.toml`: already `64000` (no change)
- `max.toml`: already `64000` (no change)

## 7. Proxy Training After Tokenizer

Once Borno is built and validated:
1. Re-encode Shakespeare + downloaded corpus into binary shards using Borno
2. Update proxy config: vocab_size=64000, keep other hyperparams
3. Train 0.5B proxy for 10K steps
4. Validate: loss convergence similar to previous run, Bangla input produces sensible tokens

## 8. File Outputs

After training, Borno produces:
- `borno_vocab.json` — token-to-ID mapping (64K entries)
- `borno_merges.txt` — ordered merge rules (61K lines)
- `borno_config.json` — metadata (vocab size, special tokens, version)

These files are loaded at inference by the rs-bpe encoder. They also serve as the canonical vocab for GGUF model exports.

## 9. Self-Training Pipeline Support

Borno's special tokens are designed to enable Noor's K2+ self-training system (separate spec, built after Borno):

- `<think>`/`</think>` — reasoning traces for actor-critic self-improvement
- `<tool_call>`/`<tool_result>` — synthetic tool environment practice (ADE P2)
- `<memory>`/`</memory>` — cross-cycle memory for competence-aware self-training (ADE P3)
- `<user>`/`<assistant>`/`<system>` — conversation structure for self-generated training data

The reserved block (IDs 271-2999) provides room for future self-training control tokens (e.g., `<critique>`, `<confidence>`, `<competence_gap>`) without retraining the tokenizer.

**Key differentiator vs K2**: Noor's ADE P5 Competence Boundary will identify weak areas, and the self-training loop will target those specifically — competence-aware self-training. The tokenizer vocabulary is permanent; the self-training system will be designed in a dedicated spec.
