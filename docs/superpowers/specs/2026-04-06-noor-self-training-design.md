# Noor Self-Training Engine (NSTE) — Design Spec

**Competence-aware self-training with memory persistence across cycles.**

No other open-source model carries a competence map between training runs. K2 self-trains blindly each cycle. Noor remembers its weaknesses and targets them — compounding improvement.

**Goal**: Build an agent-focused self-training system that uses ADE (P2+P3+P4+P5) to create a training loop that gets exponentially better at weak spots across multiple cycles.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  NSTE — 3 Stages                     │
│                                                       │
│  Stage 1: Synthetic Data Generation                   │
│  ├── Rephrasing pipeline (model rewrites own data)   │
│  └── Tool environment generator (70+ scenarios)      │
│                                                       │
│  Stage 2: Practice Loop                               │
│  ├── Agent executes tasks in synthetic environments  │
│  ├── P2 Instruments: tool calling                    │
│  ├── P3 Memory: persists learnings across cycles     │
│  └── P4 Experience: accumulates patterns             │
│                                                       │
│  Stage 3: Competence-Aware Actor-Critic              │
│  ├── Competence gate (min threshold required)        │
│  ├── Actor: generates response                       │
│  ├── P5 Confidence: rates own certainty BEFORE       │
│  ├── Critic: scores response (verify + rubric)       │
│  ├── Gap detector: confident+wrong = priority target │
│  ├── Replay buffer: 70% targeted + 30% original     │
│  └── P3 Memory: competence map persists to next cycle│
│                                                       │
│  Cycle N output → feeds Cycle N+1 input              │
│  Weaknesses compound-fix across cycles               │
└─────────────────────────────────────────────────────┘
```

### Key Differentiator vs K2

K2 runs each self-training cycle independently. Noor's P3 Memory carries a "competence map" between cycles — a structured record of what it's good at and what it struggles with. Each cycle targets the weakest areas. This is **competence-aware self-training**.

### ADE Programs Used

| Program | Role in Self-Training |
|---------|----------------------|
| P2 Instruments | Tool calling — the model practices using tools in synthetic environments |
| P3 Memory | Persists competence map across cycles — remembers weaknesses |
| P4 Experience | Accumulates patterns from practice — learns from interaction history |
| P5 Competence Boundary | Rates own confidence before answering — enables gap detection |

P1 (State) and P6 (Confidence calibration) are runtime concerns, not active during self-training.

---

## 2. Self-Training Data Pipeline

### Stage 1a: Synthetic Rephrasing

The model takes existing training data and rephrases it in multiple styles:
- Same fact → different tone (formal, casual, Bangla colloquial)
- Same instruction → step-by-step explanation
- Same code → with comments → without → different language
- English → Bangla translation pairs (bidirectional)

This multiplies training data quality without new data collection. K2 showed this gives more signal per token than raw web data.

### Stage 1b: Tool Environment Generator

TOML-based tool definition format. Each environment specifies:

```toml
[tool]
name = "bkash_send_money"
description = "Send money via bKash mobile banking"
parameters = [
    { name = "recipient", type = "phone", required = true },
    { name = "amount", type = "float", required = true },
    { name = "pin", type = "string", required = true },
]
returns = { type = "object", fields = ["transaction_id", "status", "balance"] }
errors = ["insufficient_balance", "invalid_pin", "daily_limit_exceeded"]

[[scenarios]]
task = "Send 500 taka to 01712345678"
expected_calls = ["bkash_send_money(recipient='01712345678', amount=500, pin='****')"]
success_condition = "status == 'success'"
```

**Scope**: 50 generic tools (file ops, web search, calculator, calendar, weather) + 20 Bangladesh-specific tools (bKash, Nagad, NID lookup, Bangla OCR). Expandable to thousands via the same TOML format.

**Layered approach**: Generic tools teach HOW to be an agent. Bangladesh tools teach WHERE to apply those skills.

---

## 3. Competence Map — The Core Innovation

A persistent data structure tracking what the model knows and doesn't know, driving the data sampler for the next training cycle.

### Format

```json
{
  "version": 2,
  "cycle": 3,
  "timestamp": "2026-04-15T10:00:00Z",
  "categories": {
    "reasoning": {
      "math_arithmetic": { "score": 0.82, "samples": 500, "trend": "up" },
      "math_word_problems": { "score": 0.41, "samples": 300, "trend": "flat" },
      "logic_deduction": { "score": 0.67, "samples": 200, "trend": "up" }
    },
    "language": {
      "english_fluency": { "score": 0.88, "samples": 1000, "trend": "up" },
      "bangla_fluency": { "score": 0.55, "samples": 800, "trend": "up" },
      "bangla_formal": { "score": 0.38, "samples": 150, "trend": "flat" },
      "code_python": { "score": 0.71, "samples": 400, "trend": "up" },
      "code_rust": { "score": 0.29, "samples": 100, "trend": "flat" }
    },
    "agent": {
      "single_tool_call": { "score": 0.76, "samples": 600, "trend": "up" },
      "multi_step_tool_chain": { "score": 0.23, "samples": 200, "trend": "flat" },
      "error_recovery": { "score": 0.15, "samples": 80, "trend": "down" },
      "bangla_tool_calling": { "score": 0.31, "samples": 150, "trend": "up" }
    }
  }
}
```

### How It Drives Training

The data sampler inverts scores to create sampling weights:

```
weight = (1.0 - score)^2
```

`error_recovery` at 0.15 → weight 0.72. `english_fluency` at 0.88 → weight 0.014. The model trains 50x more on error recovery than English fluency.

### How It Persists

Saved as JSON after each cycle. P3 Memory loads it at the start of the next cycle. This is the "memory between training runs" that K2 doesn't have.

### How Scores Update

After each evaluation on held-out data:

```
new_score = 0.7 * eval_accuracy + 0.3 * old_score
```

Exponential moving average prevents single bad evals from crashing a score.

---

## 4. Actor-Critic Loop with Competence Gate

### Competence Gate

Self-critique is only useful when the model is competent enough to judge itself. Below the gate, use verifiable rewards only.

| Threshold | Metric |
|-----------|--------|
| MMLU | > 60% |
| GSM8K | > 40% |
| Bangla fluency | > 0.5 score |

### Below Gate (Early Training)

- Math problems → binary right/wrong
- Code tasks → passes tests or doesn't
- Tool calls → correct arguments or not
- No self-critique (model too weak to judge itself)

### Above Gate (Competent Model)

1. **Actor generates** response to a task
2. **P5 rates confidence** before seeing result (0.0-1.0)
3. **Verifiable check** if applicable (math/code/tool)
4. **Critic scores** open-ended responses using rubrics:
   - Clarity (1-5)
   - Accuracy (1-5)
   - Helpfulness (1-5)
   - Bangla naturalness (1-5, for Bangla tasks)
5. **Gap detector** compares confidence vs actual score:
   - Confident + wrong → **competence gap** → high priority for next cycle
   - Uncertain + right → **hidden strength** → boost confidence calibration
   - Confident + right → working as intended
   - Uncertain + wrong → expected, normal training signal
6. **Competence map update** — gap detector feeds category scores

### Anti-Forgetting (Replay Buffer)

Every training batch:
- 70% targeted from competence map (weak areas)
- 30% replay from original pretraining data (PTX-style)

### Token Budget

Each response capped at 2048 tokens during self-training. Prevents verbosity-based reward hacking (K2's lesson).

---

## 5. Phasing & Integration

### When Self-Training Activates

| Phase | Model | Self-Training |
|-------|-------|---------------|
| Phase 0 | Proxy 0.5B | None — too weak |
| Phase 2 | Edge 2.8B | Stage 1 only (rephrasing) — generates better training data |
| Phase 3 | Pro 12B | Full NSTE — all 3 stages, competence gate likely passes |
| Phase 3.3 | Pro post-train | Multiple NSTE cycles (3-5 cycles, each improves on last) |

### One NSTE Cycle

1. Load competence map from previous cycle (or initialize fresh)
2. Generate synthetic data:
   - Rephrase existing data (weighted by competence map)
   - Generate tool scenarios from environment definitions
3. Practice loop (1000+ tasks):
   - Model attempts each task
   - Actor-critic scores it
   - Gap detector flags competence gaps
4. Train on results:
   - 70% weak-area targeted data
   - 30% replay buffer
   - ~500M tokens per cycle
5. Evaluate on held-out benchmarks
6. Update competence map
7. Save competence map + checkpoint

### Cycle Duration Estimate (RTX 3060)

- Data generation: ~2 hours (model inference for rephrasing)
- Practice loop: ~4 hours (1000 tasks, multi-step each)
- Training: ~8 hours (500M tokens)
- Eval + save: ~30 min
- **Total: ~15 hours per cycle**
- 5 cycles = ~3 days continuous GPU training

---

## 6. Crate Structure

```
crates/noor-self-train/
├── Cargo.toml
├── src/
│   ├── lib.rs              — NSTE orchestrator (cycle runner)
│   ├── rephraser.rs        — synthetic data rephrasing pipeline
│   ├── environments.rs     — TOML tool environment loader + executor
│   ├── actor_critic.rs     — response generation + scoring
│   ├── competence_map.rs   — persistent competence tracking (JSON I/O)
│   ├── gap_detector.rs     — confidence vs result analysis
│   ├── data_sampler.rs     — weighted sampling driven by competence map
│   └── replay_buffer.rs    — anti-forgetting mix of old + new data
├── environments/
│   ├── generic/            — 50 generic tool definitions (TOML)
│   └── bangladesh/         — 20 BD-specific tool definitions (TOML)
└── tests/
    ├── competence_map_tests.rs
    ├── gap_detector_tests.rs
    └── sampler_tests.rs
```

### Dependencies on Existing Crates

- `noor-core` — model forward pass, tokenizer (Borno)
- `noor-train` — training loop, optimizer (Muon + SMEBU), data pipeline
- `borno` — tokenization of synthetic data

---

## 7. Borno Tokenizer Support

Borno's special tokens directly enable NSTE:

| Token | NSTE Use |
|-------|----------|
| `<think>`/`</think>` | Actor's reasoning traces during self-training |
| `<tool_call>`/`<tool_result>` | Tool environment practice (P2 Instruments) |
| `<memory>`/`</memory>` | P3 Memory persistence markers |
| `<user>`/`<assistant>` | Conversation structure for synthetic data |
| Reserved (271-2999) | Future tokens: `<confidence>`, `<competence_gap>`, `<critique>` |

---

## 8. Success Criteria

After 5 NSTE cycles on Noor-Pro:

| Metric | Before NSTE | After NSTE (target) |
|--------|-------------|---------------------|
| Tool-calling accuracy | ~70% | > 90% |
| Multi-step tool chains | ~20% | > 60% |
| Bangla tool-calling | ~30% | > 70% |
| Confidence calibration | uncalibrated | ECE < 0.1 |
| Competence map coverage | 0 categories | 20+ categories tracked |
| GSM8K (math) | baseline | +10 points |
| Bangla fluency | baseline | +15 points |

The competence map should show monotonic improvement in targeted weak areas across cycles, with no regression in strong areas (anti-forgetting verified).
