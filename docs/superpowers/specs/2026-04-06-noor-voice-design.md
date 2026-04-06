# NoorVoice — Native Voice-to-Voice Design Spec

**Audio in, audio out. One model. No ASR. No TTS. No pipeline.**

Noor is the first AI model with voice built into the foundation — not bolted on. Speech tokens live in the same vocabulary as text tokens. The transformer doesn't know the difference. One forward pass handles both modalities.

---

## 1. Why This Matters

Every voice AI today:
```
Audio → [ASR] → Text → [LLM] → Text → [TTS] → Audio
  3 models, 3 latencies, emotion lost, robotic output
```

NoorVoice:
```
Audio → [Codec Encoder] → Tokens → [Noor] → Tokens → [Codec Decoder] → Audio
                    ↑                              ↑
            Same vocabulary              Same model weights
            Same embedding table         One forward pass
```

The codec encoder/decoder is NOT a separate AI model — it's a small signal processor (~10M params) that converts between waveforms and discrete tokens. It's Borno for audio.

## 2. NoorCodec — Speech Tokenizer (Rust + Zig)

### Architecture

EnCodec-style residual vector quantization:

```
Encoder: PCM audio → 5 Conv1d layers → RVQ (8 codebooks × 1024) → discrete tokens
Decoder: discrete tokens → RVQ lookup → 5 TransposedConv1d layers → PCM audio
```

~10M parameters. Conv1d + codebook lookup — no attention, no MoE. Straightforward in Rust/Zig.

### Why Build Our Own (Not Port EnCodec)

- EnCodec is trained on English/European audio — Bangla tonal patterns sound terrible
- Training our own codec on Bangla + English speech gives native quality for both
- We own the stack — no dependency on Meta's weights or format
- Can optimize codebook size for Noor's deployment targets (phone, RPi, browser)

### Training Data (MIT-licensed)

| Source | Language | Hours | License |
|--------|----------|-------|---------|
| Common Voice Bangla | Bangla | ~200 | CC-0 |
| Common Voice English | English | ~2000 | CC-0 |
| LibriSpeech | English | 960 | CC-BY-4.0 |

Target: ~1000 hours balanced Bangla+English.

### Crate Structure

```
crates/noor-codec/
├── Cargo.toml
├── src/
│   ├── lib.rs          — pub struct NoorCodec { encode(), decode() }
│   ├── encoder.rs      — Conv1d stack → RVQ
│   ├── decoder.rs      — RVQ → TransposedConv1d stack → PCM
│   ├── rvq.rs          — Residual Vector Quantizer (8 codebooks × 1024)
│   ├── conv1d.rs       — 1D convolution layers (forward + backward)
│   └── audio_io.rs     — PCM read/write, resampling, VAD
└── tests/
```

## 3. Expanded Borno Vocabulary

### Layout

```
┌─────────────┬────────┬───────────────────────────────┐
│ ID Range    │ Count  │ Purpose                        │
├─────────────┼────────┼───────────────────────────────┤
│ 0-255       │ 256    │ Byte fallback (text)           │
│ 256-275     │ 20     │ Special tokens (ADE)           │
│ 276-2999    │ 2724   │ Reserved (future text/control) │
│ 3000-63999  │ 61000  │ BPE text merges                │
│ 64000-64001 │ 2      │ <audio_start>, <audio_end>     │
│ 64002-64009 │ 8      │ Codebook layer markers         │
│ 64010-72201 │ 8192   │ Speech codec tokens (8×1024)   │
│ 72202-72255 │ 54     │ Reserved audio control          │
└─────────────┴────────┴───────────────────────────────┘
Total: 72,256
```

### Embedding Cost

| Model | Text-only (64K) | With Voice (72K) | Delta |
|-------|-----------------|------------------|-------|
| Edge (d=1024) | 128MB | 144MB | +16MB |
| Pro (d=2048) | 256MB | 295MB | +39MB |

Minimal cost. The transformer layers don't change at all.

### Special Audio Tokens

| ID | Token | Purpose |
|----|-------|---------|
| 64000 | `<audio_start>` | Begin audio sequence |
| 64001 | `<audio_end>` | End audio sequence |
| 64002-64009 | `<cb0>`-`<cb7>` | Codebook layer markers (interleave pattern) |

### Interleaving Pattern

Speech tokens are interleaved by codebook layer within each frame:
```
<audio_start> <cb0> [tok] [tok] [tok] <cb1> [tok] [tok] [tok] ... <cb7> [tok] [tok] [tok] <audio_end>
```

This lets the transformer learn temporal patterns across codebook layers.

## 4. Staged Training

### Phase A: Text Base Model (current)
Train Noor on text only with 64K Borno vocab. This is Phases 0-3 of the existing plan.
No voice involvement. Focus on language + agent capabilities.

### Phase B: Train NoorCodec
Build and train the speech codec in NoorTorch:
- Implement Conv1d + RVQ in Rust/Zig
- Train on ~1000 hours Bangla+English
- Validate: codec quality (PESQ > 3.5, STOI > 0.9)
- Export to GGUF
- ~10M params, trains in hours on RTX 3060

### Phase C: Vocabulary Expansion + Alignment
- Expand Borno from 64K → 72K
- Initialize speech embedding vectors from noise (small scale)
- Freeze all text weights
- Train on interleaved text+speech data:
  ```
  <user><audio_start>[codec tokens]<audio_end>
  <assistant>Text response here.
  
  <user>Text question here.
  <assistant><audio_start>[codec tokens]<audio_end>
  ```
- Model learns mapping between speech tokens and text tokens
- ~500M tokens, ~1 week on RTX 3060

### Phase D: End-to-End Voice Fine-tuning
- Unfreeze all weights
- Train on conversational speech data (paired audio dialogues)
- Model learns to respond in speech when given speech input
- Fine-tune on Bangla conversational data specifically
- ~200M tokens, ~3 days on RTX 3060

### Phase E: Voice + Agent
- Combine voice with NSTE self-training
- Model practices tool-calling via voice ("আমার bKash ব্যালেন্স কত?")
- Voice-native agent: hear question → think → call tool → speak answer

## 5. Inference — One Binary

```rust
// The entire NoorVoice pipeline
fn voice_loop(mic: &Mic, speaker: &Speaker, noor: &NoorModel, codec: &NoorCodec) {
    loop {
        let audio = mic.read_chunk();            // 20ms PCM @ 16kHz
        let tokens = codec.encode(&audio);        // audio → codec tokens
        let prompt = wrap_audio_tokens(&tokens);  // add <audio_start>/<audio_end>
        let response = noor.generate(&prompt);    // one forward pass
        
        if is_audio_response(&response) {
            let audio_out = codec.decode(&response);  // tokens → audio
            speaker.play(&audio_out);
        } else {
            // Text response — use separate TTS or display text
            display_text(&response);
        }
    }
}
```

One Rust binary. No Python. No ASR. No TTS. Audio in, tokens, tokens out, audio.

### Streaming

For real-time conversation, Noor generates tokens autoregressively. As soon as enough codec tokens are generated for one audio frame (~20ms), the codec decoder runs and audio plays. This gives ~50ms mouth-to-ear latency on-device.

## 6. What Makes This Different

| | GPT-4o | Google Gemini | Noor |
|---|---|---|---|
| Architecture | Text LLM + audio heads | Multimodal from start | Speech tokens in vocab |
| Codec | Proprietary | Proprietary | NoorCodec (open, Bangla-native) |
| Size | 200B+ cloud | 200B+ cloud | 1-12B on-device |
| Latency | ~300ms (network) | ~250ms (network) | ~50ms (local) |
| Privacy | Audio → cloud | Audio → cloud | Audio never leaves device |
| Bangla | Afterthought | Afterthought | First-class, trained on Bangla speech |
| Stack | Python/C++ | Python/JAX | Single Rust binary |
| License | Closed | Closed | MIT |

## 7. Updated Phase Plan

```
Phase 0: Proxy (0.5B text)           ✅ in progress
Phase 1: Zig kernels + Accelerate    ✅ done  
Phase 2: Noor-Edge (2.8B text)       → base language model
Phase 3: Noor-Pro (12B MoE text)     → full language model
Phase 3.3: NSTE self-training        → agent capabilities
Phase 4: ADE integration             → deployment pipeline
Phase 5: NoorCodec                   → speech tokenizer (Rust/Zig)
Phase 6: NoorVoice training          → expand vocab, multimodal training
Phase 7: Voice fine-tuning           → conversational voice
Phase 8: Voice + Agent               → voice-native tool calling
```

## 8. Hardware Requirements

### NoorCodec Training (~10M params)
- RTX 3060 12GB: fits easily, trains in hours
- M4 24GB: also works

### NoorVoice Training (expand existing model)
- Vocabulary expansion: minimal memory cost (+39MB for Pro)
- Phase C (alignment): same hardware as text training
- Phase D (fine-tuning): same hardware as text training
- Speech data preprocessing: codec encoding is fast (~100x real-time)

### Inference
- Codec encoder: ~10M params, ~40MB quantized, runs in <5ms
- Noor model: same as text inference
- Codec decoder: ~10M params, ~40MB quantized, runs in <5ms
- Total overhead for voice: ~80MB + ~10ms latency

On a phone with Noor-Edge TQ3 (1.2GB) + NoorCodec (80MB) = 1.28GB total for full voice AI.
