# Data-Efficient LLM Training: Research Report (2024-2026)

**Prepared for:** Noor model family (Edge 2.8B / Pro 12B / Max 28B)
**Hardware context:** RTX 3060 12GB + M4 24GB, Kaggle T4
**Training data:** ~290K verified lines + 16GB master JSONL (English + Bangla + reasoning)
**Stack:** NoorTorch (Mojo + Zig + Rust), no Python/PyTorch

---

## 1. Curriculum Learning for LLMs

**What:** Order training data from easy to hard rather than random shuffling.

### Key Papers
| Paper | ID | Year | Finding |
|-------|-----|------|---------|
| Beyond Random Sampling | arXiv:2506.11300 | 2025 | Compression ratio, lexical diversity (MTLD), and Flesch readability are the best difficulty signals. Sustained 1.2-3.5% accuracy improvement from reordering alone |
| Dataset Decomposition (Apple) | NeurIPS 2024 | 2024 | Variable sequence length curriculum: train short sequences first, then long. Reduces compute by training most tokens with cheaper short-context attention |
| LR Decay Wastes Best Data | arXiv:2511.18903 | 2025 | Curriculum gains are real but **diminished by cosine LR decay** -- the best data arrives when LR is lowest. Constant LR or model averaging restores the benefit |
| Influence-driven Curriculum | arXiv:2508.15475 | 2025 | Sort by influence function scores instead of heuristics. Outperforms random ordering even on limited data |

### Measured Speedup
- **18-45% fewer training steps** to reach baseline perplexity
- **1.2-3.5% accuracy improvement** at matched compute
- Effect is **stronger for smaller models** (< 160M), shrinks at larger scales
- Interacts poorly with cosine LR decay (fix: use constant LR phase or weight averaging)

### Implementation Complexity: **Easy**
- Precompute difficulty score per document (perplexity from a small proxy, or compression ratio)
- Sort shards by difficulty before training
- No changes to the training loop itself

### Recommendation for Noor
**HIGH PRIORITY.** Since you already have a Borno tokenizer and shard pipeline, add a difficulty scoring pass. Use compression ratio (bytes / Borno tokens) as the signal -- free to compute, language-agnostic, works for Bangla. Train easy-first for first 70% of tokens, then mix in hard. Pair with constant LR + cosine decay only in final 20%.

---

## 2. Influence Functions / Data Attribution

**What:** Score which training examples contribute most to model quality; train only on high-value data.

### Key Papers
| Paper | ID | Year | Finding |
|-------|-----|------|---------|
| Studying LLM Generalization with Influence Functions (Anthropic/Google) | arXiv:2308.03296 | 2023 | EK-FAC approximation works at LLM scale. Shows cross-lingual generalization patterns -- relevant for bilingual |
| DataInf | arXiv:2310.00902 (ICLR 2024) | 2024 | Closed-form influence approx for LoRA-tuned models. Orders of magnitude faster than full influence functions |
| TRAK | arXiv:2303.14186 | 2023 | Randomly-projected kernel for data attribution at scale. Works on BERT, mT5, CLIP |
| In2Core | EMNLP 2024 | 2024 | Uses influence functions for coreset selection in instruction fine-tuning |

### Measured Impact
- DataInf: 1000x faster than retraining-based influence, accurately identifies mislabeled/harmful data
- TRAK: Scales to ImageNet-scale datasets, practical for selecting most-valuable 20-30% of data
- **Practical ceiling:** These methods work well for fine-tuning data curation but are **expensive for pretraining** (must run inference on entire dataset with a trained model)

### Implementation Complexity: **Hard**
- Requires a trained reference model to compute influence scores
- EK-FAC needs second-order gradient information
- DataInf is simpler (closed-form for LoRA) but still needs per-example gradient computation
- Custom Rust implementation of EK-FAC/TRAK would be significant engineering

### Recommendation for Noor
**LOW PRIORITY for pretraining, MEDIUM for fine-tuning.** The compute cost of scoring every pretraining example exceeds the savings. Instead, use simpler proxy signals (perplexity, compression ratio) for pretraining data selection. Reserve influence functions for curating the ~500M token fine-tuning dataset (tool-calling, ADE, Bangla reasoning) where each example matters more.

---

## 3. Online Data Filtering During Training

**What:** The model itself decides which examples are worth learning from mid-training.

### Key Papers
| Paper | ID | Year | Finding |
|-------|-----|------|---------|
| FineWeb / FineWeb-Edu | arXiv:2406.17557 | 2024 | Model-based quality classifier (trained on LLM annotations). FineWeb-Edu: training on **10% of filtered data matches 350B tokens of unfiltered** |
| Ultra-FineWeb | arXiv:2505.05427 | 2025 | Lightweight fastText classifier achieves near-LLM filtering quality. 1T English + 120B Chinese tokens |
| FinerWeb-10BT | arXiv:2501.07314 | 2025 | Line-level filtering (not document-level). LLM scores each line, DeBERTa distills the classifier |
| PreSelect | arXiv:2503.00808 | 2025 | Data that **predicts** downstream performance is the data that teaches. fastText-based, lightweight |
| DATAMASK | arXiv:2512.24265 | 2025 | Joint quality+diversity optimization via policy gradient mask learning |
| Online Data Mixing (ODM) | arXiv:2312.02406 | 2023 | Multi-armed bandit adjusts domain weights during training based on validation perplexity |

### Measured Impact
- **FineWeb-Edu: 10x data reduction** with matched performance (the single most impactful technique in this report)
- Ultra-FineWeb fastText classifier: near-zero overhead, applicable to any language
- ODM: 2-3% improvement over static mixing with minimal compute overhead
- PreSelect: fastText-based selection outperforms random by 5-8% on downstream tasks

### Implementation Complexity: **Medium**
- Pre-filtering (FineWeb-style): Train a fastText or small classifier once, score all documents, threshold
- Online filtering: Requires modifying the data loader to skip low-scoring examples mid-training
- Both are feasible in a Rust data pipeline

### Recommendation for Noor
**HIGHEST PRIORITY.** This is the single biggest lever. Steps:
1. Train a fastText quality classifier on your verified datasets (290K lines = positive examples) vs raw web data (negative)
2. Score the 16GB master JSONL and the CC-100 Bangla data
3. Keep only top 30-50% by quality score
4. This alone could halve your training time with no quality loss

For Bangla specifically: Use Ultra-FineWeb's pipeline adapted to Bangla. Machine-translate a small set of FineWeb-Edu quality annotations to Bangla to bootstrap the classifier.

---

## 4. Deduplication Impact on Training

**What:** Remove duplicate and near-duplicate data before training.

### Key Papers
| Paper | ID | Year | Finding |
|-------|-----|------|---------|
| SemDeDup (Meta) | arXiv:2303.09540 | 2023 | Semantic dedup via embeddings. **Remove 50% of data with minimal performance loss.** Improves OOD generalization |
| Deduplicating Training Data Makes LMs Better | arXiv:2107.06499 | 2021 | Foundation paper. Exact + near-dedup reduces memorization, improves perplexity |
| SoftDedup | arXiv:2407.06654 | 2024 | **Don't remove, reweight.** Score "commonness" via n-gram model, downweight common examples. Better than hard dedup |
| SlimPajama-DC | arXiv:2309.10818 | 2023 | Comprehensive dedup study. Shows optimal dedup level depends on data diversity |
| LSHBloom | arXiv:2411.04257 | 2024 | MinHash + Bloom filters for memory-efficient dedup at extreme scale |
| Mix, MinHash, and Match | arXiv:2512.18834 | 2025 | Cross-source agreement for multilingual dedup. Relevant for bilingual pipelines |

### Measured Impact
- **SemDeDup: 50% data removal, minimal quality loss** (strongest single result)
- SemDeDup overhead: < 1% of training cost (embedding + clustering)
- Exact dedup (MinHash): typically removes 10-30% of web data
- SoftDedup: 1-2% accuracy improvement over hard dedup at matched token count
- Combined exact + semantic dedup: **60-70% data can be removed** from raw web crawls

### Implementation Complexity: **Medium**
- Exact dedup (MinHash): straightforward, many Rust crates exist (e.g., `gaoya`)
- Semantic dedup: requires embedding model (can use a small sentence transformer)
- SoftDedup: requires n-gram model for scoring, then modify data loader weights
- For 16GB of data, all methods are computationally trivial

### Recommendation for Noor
**HIGH PRIORITY.** Your 16GB master JSONL likely has significant duplication.
1. Run MinHash dedup first (exact + near-exact). Expect 10-30% removal
2. Then run SoftDedup: score remaining data by n-gram commonness, downweight (not remove) the most common patterns
3. Skip full SemDeDup unless you have a good Bangla embedding model -- the embedding quality matters
4. For Bangla specifically: be careful not to over-dedup, since Bangla data is already scarce

---

## 5. Data Mixing Ratios

**What:** Optimal proportions of different data types (code, math, language, knowledge).

### Key Papers
| Paper | ID | Year | Finding |
|-------|-----|------|---------|
| DoReMi (Google) | arXiv:2305.10429 (NeurIPS 2023) | 2023 | 280M proxy finds optimal weights for 8B model. **6.5% accuracy improvement, 2.6x fewer training steps** vs default Pile weights |
| SampleMix | arXiv:2503.01506 | 2025 | Sample-level (not domain-level) mixing. Surpasses DoReMi with fewer training steps |
| Olmix | arXiv:2602.12237 | 2026 | Framework for recomputing mixtures when domain sets evolve. Practical for iterative dataset building |
| OptiMer | arXiv:2603.28858 | 2026 | Post-hoc Bayesian optimization of distribution vectors. Decouple mixing from training |
| Online Data Mixing (ODM) | arXiv:2312.02406 | 2023 | Multi-armed bandit adjusts domain weights online. 5-shot MMLU improvement |
| R&B | arXiv:2505.00358 | 2025 | Semantic regrouping + gradient-based balancing. 26 upvotes on HF, strong results |
| Multilingual Mixing (EPFL) | arXiv:2502.10361 | 2025 | Model-based selection for multilingual pretraining. English pivot + target language works best |
| Revisiting Multilingual Mixtures | arXiv:2510.25947 | 2025 | Optimal ratios change with scale. English as high-proportion pivot is robust |

### Measured Impact
- DoReMi: **2.6x speedup** to reach baseline accuracy
- Key finding for bilingual: English as pivot language with sufficient target language tokens works best
- No inherent penalty for bilingual training if token budgets are sufficient
- Optimal ratios are **scale-dependent** -- what works at 1B may not work at 10B

### Bilingual English+Bangla Guidance
Based on multilingual mixing research:
- **English: 60-70%** (pivot language, rich in knowledge/code/math)
- **Bangla: 15-25%** (ensure sufficient representation, minimum ~2B tokens)
- **Code: 10-15%** (improves reasoning even for non-code tasks)
- **Math/reasoning: 5-10%** (GSM8K, ARC, etc.)
- Use Unimax or temperature-based sampling to prevent Bangla being drowned out

### Implementation Complexity: **Easy-Medium**
- Static mixing: just set shard proportions (easy)
- DoReMi: train a small proxy with DRO, use its weights (medium, needs proxy model)
- Online mixing: modify data loader with bandit algorithm (medium)

### Recommendation for Noor
**HIGH PRIORITY.** Your current dataset is ~10% Bangla, which is probably too low.
1. Start with static ratios: 65% English / 20% Bangla / 10% code / 5% math-reasoning
2. After training the Noor-Edge proxy (288M), use DoReMi-style DRO to find better weights
3. Supplement Bangla data aggressively -- CC-100, Bangla Wikipedia, synthetic translation of English knowledge
4. Machine-translate high-quality English data (FineWeb-Edu filtered) to Bangla using a good MT model

---

## 6. Synthetic Data Augmentation

**What:** Use a teacher model to generate or rephrase training data.

### Key Papers
| Paper | ID | Year | Finding |
|-------|-----|------|---------|
| WRAP | arXiv:2401.16380 | 2024 | Rephrase web data into Wikipedia/QA style. **Matches 3x more real data** in perplexity and accuracy |
| BeyondWeb (Datology AI) | arXiv:2508.10975 | 2025 | Best-in-class synthetic pretraining data. **+7.1pp over RedPajama, 7.7x training speedup** on 8B models |
| FineInstructions | arXiv:2601.22146 | 2026 | Pre-train from scratch on synthetic instruction-response pairs. Outperforms traditional pretraining on response quality benchmarks |
| Cosmopedia (HuggingFace) | Blog post | 2024 | 39M synthetic textbooks/stories from Mixtral-8x7B. Powers SmolLM models |
| Synthetic Continued Pretraining (EntiGraph) | arXiv:2409.07431 (ICLR 2025) | 2024 | Entity-graph-based augmentation for domain-specific knowledge. Works with small corpora |
| REWIRE | arXiv:2506.04689 | 2025 | Rewrite low-quality web data to high-quality. Surpasses 2x filtered web data |
| LLM2LLM | arXiv:2403.15042 | 2024 | Iterative: teacher generates data where student fails. Up to 24% improvement in low-data regimes |

### Measured Impact
- WRAP: **3x data efficiency** (100B rephrased tokens = 300B raw tokens)
- BeyondWeb: **7.7x training speedup** over raw web data
- Cosmopedia: Powers competitive sub-2B models entirely on synthetic data
- K2's rephrasing pipeline: Core ingredient in achieving SOTA with 15.5T tokens (already in your notes)

### Implementation Complexity: **Medium**
- Use Gemma E2B (already planned) to rephrase existing data
- Style templates: Wikipedia-style, QA-style, step-by-step explanation
- For Bangla: rephrase English knowledge into Bangla (dual benefit: augmentation + translation)
- Pipeline: raw text -> prompt template -> teacher model -> filtered output -> training shard

### Recommendation for Noor
**HIGH PRIORITY -- already planned.** Your training strategy already includes "Generate synthetic data with Gemma E2B on M4 (2B tokens, free)." Key refinements:
1. Use WRAP-style rephrasing: take each document and generate 2-3 style variants (Wikipedia, QA, step-by-step)
2. Prioritize Bangla augmentation: rephrase English knowledge into Bangla to expand the 10% ratio
3. Use EntiGraph for domain-specific data (Bangladesh history, culture, science)
4. Quality filter the synthetic output -- not all generations are helpful
5. BeyondWeb shows that **which** teacher model matters: larger teachers produce diminishing returns vs medium teachers. Gemma E2B is a good choice

---

## 7. Knowledge Distillation as Pretraining

**What:** Train a small model using logits from a larger model instead of (or in addition to) raw text.

### Key Papers
| Paper | ID | Year | Finding |
|-------|-----|------|---------|
| Pre-training Distillation Design Space | arXiv:2410.16215 (ACL 2025) | 2024 | GLM-4-9B teacher -> 1.9B student. **+1.6% average improvement** over standard LM loss. Explores offline vs online logits |
| MiniLLM | arXiv:2306.08543 | 2023 | Reverse KLD (not forward KLD) is better for generative LLM distillation |
| Sparse Logit Sampling | arXiv:2503.16870 | 2025 | Random sample of teacher logits (not full vocabulary). Near-full-distillation quality at fraction of I/O cost |
| Cross-Tokenizer Distillation (ULD) | arXiv:2402.12030 | 2024 | Optimal transport loss enables distillation across different tokenizers |
| Co-training/Co-distillation (Meta) | arXiv:2311.02849 | 2023 | Teacher and student train simultaneously, mutual knowledge transfer |

### Measured Impact
- Pre-training distillation: **+1.6% accuracy** on English+Chinese benchmarks (GLM-4-9B -> 1.9B)
- Larger students benefit more from distillation than smaller ones
- Reverse KLD > forward KLD for generation quality
- Sparse logit sampling: 10x reduction in teacher logit storage with minimal quality loss
- **Key insight from Spurious Rewards paper (already in your notes):** Distillation has same ceiling as RLVR -- pretraining quality is the real ceiling

### Implementation Complexity: **Hard**
- Requires running teacher model to generate logits for entire training corpus
- Storage: full vocab logits per token = massive (64K vocab * 2 bytes * billions of tokens)
- Sparse logit sampling reduces this dramatically (sample top-K logits)
- Cross-tokenizer distillation adds another layer of complexity if teacher uses different tokenizer
- Custom Rust implementation of KD loss + teacher inference pipeline = significant work

### Recommendation for Noor
**MEDIUM PRIORITY -- already planned for Phase 2-3.** Your strategy already includes distilling from Gemma E2B and Gemma 26B-A4B. Key refinements:
1. Use **sparse logit sampling**: save only top-128 teacher logits per token (not full 64K vocab). This reduces storage by 500x
2. Use **reverse KLD** (MiniLLM) instead of forward KLD for the distillation loss
3. Pre-compute teacher logits offline and store as compressed binary shards alongside token IDs
4. Cross-tokenizer distillation (ULD) may be needed if Gemma's tokenizer differs from Borno -- investigate

---

## 8. Packed Sequences / Variable Length Batching

**What:** Eliminate padding waste by packing multiple short sequences into single training sequences.

### Key Papers
| Paper | ID | Year | Finding |
|-------|-----|------|---------|
| Efficient Sequence Packing (Graphcore) | arXiv:2107.02027 | 2021 | SPFHP and NNLSHP histogram-based packing algorithms. Eliminates padding entirely |
| Packing with Flash Attention | arXiv:2407.09105 | 2024 | Proper attention masking with packing. Critical to prevent cross-contamination |
| Best-fit Packing (Amazon) | arXiv:2404.10830 | 2024 | Length-aware combinatorial optimization. Reduces hallucination from cross-document attention |
| Prepacking | arXiv:2404.09529 | 2024 | Bin-packing for prefilling, 2x throughput improvement |
| Dataset Decomposition (Apple) | arXiv:2405.13226 | 2024 | Variable sequence length training as implicit curriculum. Train short first, long later |

### Measured Impact
- **Up to 50% of tokens can be padding** in typical NLP datasets
- Packing eliminates this entirely: **2x throughput** in extreme cases
- With Unsloth: 60% memory reduction, 2x speedup on instruction-tuning
- Best-fit packing reduces cross-contamination hallucination vs naive concatenation
- For pretraining with uniform-length shards: padding waste is already low (~5-10%)

### Implementation Complexity: **Easy-Medium**
- For pretraining: if you pre-shard to fixed length (e.g., 4096 tokens), padding waste is already minimal
- For fine-tuning (variable length): implement SPFHP (sort by length, greedily pack shortest-first)
- Attention masking: need block-diagonal attention mask to prevent cross-document attention
- In Rust: bin-packing is a simple greedy algorithm, attention masking is a flag per position

### Recommendation for Noor
**MEDIUM PRIORITY.** Your current shard pipeline likely already packs to fixed length for pretraining, so the gain is small (~5-10%). However:
1. For fine-tuning (tool-calling, ADE, Bangla conversations): implement SPFHP packing. These datasets have highly variable lengths
2. Ensure your attention implementation supports block-diagonal masking (prevent cross-document leakage within packed sequences)
3. The Apple "Dataset Decomposition" finding is interesting: train with shorter sequences first (cheap attention), then long sequences. This is both a packing optimization and a curriculum

---

## 9. Gradient Checkpointing + Selective Recomputation

**What:** Trade compute for memory by recomputing activations during backward pass.

### Key Papers
| Paper | ID | Year | Finding |
|-------|-----|------|---------|
| Reducing Activation Recomputation (NVIDIA Megatron) | arXiv:2205.05198 | 2022 | **Selective recomputation: 5x memory reduction, only 2% throughput loss.** Save cheap-to-store activations, recompute expensive ones |
| MEMO | arXiv:2407.12117 | 2024 | Offload activations to CPU memory. 7B LLM with 1M sequence on 8 GPUs |
| Video-Ma2mba Multi-Axis Checkpointing | arXiv:2411.19460 | 2024 | Multi-axis gradient checkpointing for different model dimensions |

### What to Checkpoint vs Recompute

**Save (cheap to store, expensive to recompute):**
- Linear layer outputs (used in backward, expensive matmul to redo)
- LayerNorm statistics (mean, variance -- tiny to store)

**Recompute (expensive to store, cheap to redo):**
- Attention scores (QK^T -- massive memory, fast to recompute with Flash Attention)
- Activation function outputs (GeLU, SiLU -- element-wise, trivially fast)
- Dropout masks (just re-run with same seed)

### Optimal Strategy for 12GB VRAM (RTX 3060)

For Noor-Edge (2.8B params, ~1B active):
- **Full model fits in FP16 without checkpointing** (~5.6GB for weights + optimizer states with Muon 1x)
- Activations for batch of 2K tokens: ~1-2GB per layer
- **Checkpoint every 2 layers** (already in your training config) is a good default
- Selective recomputation (Megatron-style): save linear outputs, recompute attention + activations
- Expected overhead: **~20% slower training, ~3-5x less activation memory**

For Noor-Pro (12B MoE, ~3B active):
- Requires expert offload (already planned: SSD-backed LRU)
- Checkpoint **every layer** (not every 2) due to MoE memory overhead
- Activation CPU offload for non-active-expert layers
- Flash Attention essential (never materialize full attention matrix)

### Memory Budget with Selective Checkpointing (RTX 3060, 12GB)

| Component | Without Checkpointing | With Selective Checkpointing |
|-----------|----------------------|------------------------------|
| Model weights (FP16) | 2.0 GB | 2.0 GB |
| Muon momentum (1x) | 2.0 GB | 2.0 GB |
| Gradients (FP16) | 2.0 GB | 2.0 GB |
| Activations (micro-batch) | 4.0 GB | 0.8 GB |
| CUDA/framework overhead | 0.5 GB | 0.5 GB |
| **Total** | **10.5 GB** | **7.3 GB** |
| Headroom for batch size | 1.5 GB | 4.7 GB |

### Implementation Complexity: **Medium**
- Basic gradient checkpointing (checkpoint every N layers): easy
- Selective recomputation (choose what to save vs recompute): medium, requires per-operation decisions
- CPU offload: hard, requires async memory management
- In Rust/Mojo: implement as a wrapper around the backward pass that optionally recomputes

### Recommendation for Noor
**ALREADY PLANNED -- refine strategy.** Your config says "every 2 layers" which is a good start. Optimize by:
1. Switch to **selective recomputation**: save linear layer outputs and norm statistics, recompute attention scores and activation functions
2. This gives most of the memory savings of full checkpointing with only ~2% overhead (vs ~20% for full checkpointing)
3. On RTX 3060: this frees up ~3GB, enabling 3-4x larger micro-batches (better gradient quality)
4. Implement as Zig kernels that tag which tensors to retain vs mark for recomputation

---

## 10. Multi-Token Prediction

**What:** Train the model to predict multiple future tokens per position, not just the next one.

### Key Papers
| Paper | ID | Year | Finding |
|-------|-----|------|---------|
| Better & Faster LLMs via Multi-token Prediction (Meta) | arXiv:2404.19737 | 2024 | The foundational paper. **+12% HumanEval, +17% MBPP** at 13B scale. 3x faster inference with speculative decoding |
| Multi-Token Prediction Needs Registers | arXiv:2505.10518 | 2025 | Register tokens improve MTP stability and quality |
| L-MTP: Leap Multi-Token Prediction | arXiv:2505.17505 | 2025 | Predict non-adjacent future tokens for better long-range planning |
| Training-Free MTP via Embedding-Space Probing | arXiv:2603.17942 | 2026 | Use mask tokens in embedding space for parallel prediction without training changes |
| FastMTP | arXiv:2509.18362 | 2025 | Accelerating MTP inference with optimized speculative decoding |

### Measured Impact
- **Code tasks: +12% HumanEval, +17% MBPP** (13B model)
- Benefit **increases with model size** (marginal at < 1B, significant at 7B+)
- **3x inference speedup** via speculative decoding with MTP heads
- Improves development of induction heads and algorithmic reasoning
- Used in production: DeepSeek-V3/R1, Llama 4 (reported to use MTP)
- **Training cost:** adds N-1 extra prediction heads (small overhead, ~5% more FLOPs per step)

### Architecture
```
Standard:    input -> transformer -> head_1 -> predict token t+1
Multi-token: input -> transformer -> head_1 -> predict token t+1
                                  -> head_2 -> predict token t+2
                                  -> head_3 -> predict token t+3
                                  -> head_4 -> predict token t+4
```
Each head shares the transformer trunk but has its own small projection layer.
Loss = sum of cross-entropy losses across all heads (equal weight or decaying).

### Implementation Complexity: **Medium**
- Add N-1 extra linear projection heads to the output layer
- Modify loss function to sum cross-entropy across all heads
- At inference: use head_2..N for speculative decoding (draft tokens verified by head_1)
- In Rust/Mojo: straightforward -- just additional matmuls + loss terms

### Recommendation for Noor
**MEDIUM-HIGH PRIORITY.** The sample efficiency gain is real but strongest for code tasks and larger models. For Noor:
1. Implement 4-token prediction (the sweet spot from Meta's paper)
2. **Deploy at Noor-Pro scale (12B)** where the benefit is meaningful
3. For Noor-Edge (2.8B): may not help much -- test on proxy first
4. The inference speedup (3x via speculative decoding) is extremely valuable for on-device deployment
5. Implementation is straightforward: 3 extra linear heads + modified loss. Maybe 200 lines of Mojo

---

## Priority-Ranked Summary for Noor

| Rank | Technique | Expected Savings | Complexity | Status |
|------|-----------|-----------------|------------|--------|
| 1 | **Data Quality Filtering** (FineWeb-Edu style) | 50-80% less data needed | Medium | NEW -- implement ASAP |
| 2 | **Deduplication** (MinHash + SoftDedup) | 30-50% data reduction | Easy-Medium | NEW -- run on master JSONL |
| 3 | **Data Mixing Ratios** (DoReMi / static) | 2.6x speedup | Easy | Partially planned -- refine Bangla ratio |
| 4 | **Synthetic Augmentation** (WRAP-style) | 3-7x data efficiency | Medium | Already planned -- add style variants |
| 5 | **Curriculum Learning** | 18-45% fewer steps | Easy | NEW -- sort shards by difficulty |
| 6 | **Multi-Token Prediction** | +12-17% code, 3x inference | Medium | NEW -- implement for Pro/Max |
| 7 | **Knowledge Distillation** | +1.6% quality | Hard | Already planned -- add sparse logits |
| 8 | **Selective Recomputation** | 3-5x less activation memory | Medium | Planned -- refine to selective |
| 9 | **Packed Sequences** | 5-50% compute savings | Easy | LOW -- mainly for fine-tuning |
| 10 | **Influence Functions** | Precise data curation | Hard | LOW -- defer to fine-tuning phase |

---

## Concrete Action Plan

### Phase 0: Data Pipeline (Before Training)
1. **Deduplicate** the 16GB master JSONL with MinHash (Rust crate `gaoya` or custom)
2. **Train a fastText quality classifier** on your 290K verified lines (positive) vs raw web text (negative)
3. **Score and filter** all data -- keep top 30-50% by quality
4. **Sort remaining data by difficulty** (compression ratio = bytes / Borno tokens)
5. **Set mixing ratios**: 65% English / 20% Bangla / 10% code / 5% math
6. **Generate synthetic Bangla** by rephrasing English knowledge via Gemma E2B

### Phase 1: Training Loop Enhancements
7. **Implement selective recomputation** (save linear outputs, recompute attention)
8. **Add 4-token prediction heads** (for Noor-Pro, optional for Edge)
9. **Implement SPFHP packing** for variable-length fine-tuning data

### Phase 2: Distillation
10. **Pre-compute sparse teacher logits** (top-128 per token from Gemma)
11. **Add reverse KLD distillation loss** to training loop
12. **Validate cross-tokenizer compatibility** between Borno and Gemma tokenizer

### Estimated Combined Impact
If all top-5 techniques are applied:
- Data quality filtering: **2-5x** fewer tokens needed
- Deduplication: additional **1.3-1.5x** reduction
- Optimal mixing: **2.6x** faster convergence
- Synthetic augmentation: **3x** more effective tokens from same raw data
- Curriculum: **1.2-1.5x** fewer training steps

**Conservative combined estimate: 3-5x more efficient training overall.**
This means your planned 2B token pretraining might achieve the quality of a 6-10B token run with random data and default settings.

---

## Sources

### Curriculum Learning
- [Beyond Random Sampling (2025)](https://arxiv.org/abs/2506.11300)
- [Dataset Decomposition (NeurIPS 2024)](https://arxiv.org/abs/2405.13226)
- [LR Decay Wastes Best Data (2025)](https://arxiv.org/abs/2511.18903)
- [Curriculum Learning LLM Pretraining Analysis (2026)](https://arxiv.org/abs/2601.21698)

### Influence Functions / Data Attribution
- [Studying LLM Generalization with Influence Functions (Anthropic, 2023)](https://arxiv.org/abs/2308.03296)
- [DataInf (ICLR 2024)](https://arxiv.org/abs/2310.00902)
- [TRAK (2023)](https://arxiv.org/abs/2303.14186)

### Online Data Filtering
- [FineWeb Datasets (2024)](https://arxiv.org/abs/2406.17557)
- [Ultra-FineWeb (2025)](https://arxiv.org/abs/2505.05427)
- [FinerWeb-10BT (2025)](https://hf.co/papers/2501.07314)
- [PreSelect (2025)](https://arxiv.org/abs/2503.00808)
- [DATAMASK (2025)](https://arxiv.org/abs/2512.24265)

### Deduplication
- [SemDeDup (Meta, 2023)](https://arxiv.org/abs/2303.09540)
- [Dedup Makes LMs Better (2021)](https://arxiv.org/abs/2107.06499)
- [SoftDedup (2024)](https://arxiv.org/abs/2407.06654)
- [LSHBloom (2024)](https://arxiv.org/abs/2411.04257)
- [Mix MinHash Match (2025)](https://arxiv.org/abs/2512.18834)

### Data Mixing
- [DoReMi (NeurIPS 2023)](https://arxiv.org/abs/2305.10429)
- [SampleMix (2025)](https://arxiv.org/abs/2503.01506)
- [Olmix (2026)](https://hf.co/papers/2602.12237)
- [R&B (2025)](https://arxiv.org/abs/2505.00358)
- [Multilingual Mixing (EPFL, 2025)](https://arxiv.org/abs/2502.10361)
- [Revisiting Multilingual Mixtures (2025)](https://arxiv.org/abs/2510.25947)

### Synthetic Data
- [WRAP (2024)](https://arxiv.org/abs/2401.16380)
- [BeyondWeb (2025)](https://arxiv.org/abs/2508.10975)
- [FineInstructions (2026)](https://hf.co/papers/2601.22146)
- [Cosmopedia (HuggingFace, 2024)](https://huggingface.co/blog/cosmopedia)
- [Synthetic Continued Pretraining / EntiGraph (ICLR 2025)](https://arxiv.org/abs/2409.07431)
- [REWIRE (2025)](https://arxiv.org/abs/2506.04689)

### Knowledge Distillation
- [Pre-training Distillation Design Space (ACL 2025)](https://arxiv.org/abs/2410.16215)
- [MiniLLM (2023)](https://arxiv.org/abs/2306.08543)
- [Sparse Logit Sampling (2025)](https://arxiv.org/abs/2503.16870)
- [Cross-Tokenizer Distillation ULD (2024)](https://arxiv.org/abs/2402.12030)

### Packed Sequences
- [Efficient Sequence Packing (2021)](https://arxiv.org/abs/2107.02027)
- [Packing with Flash Attention (2024)](https://arxiv.org/abs/2407.09105)
- [Best-fit Packing (2024)](https://arxiv.org/abs/2404.10830)
- [Fewer Truncations Improve LM (2024)](https://arxiv.org/abs/2404.10830)

### Gradient Checkpointing
- [Reducing Activation Recomputation (NVIDIA Megatron, 2022)](https://arxiv.org/abs/2205.05198)
- [MEMO (2024)](https://arxiv.org/abs/2407.12117)

### Multi-Token Prediction
- [Better & Faster LLMs via MTP (Meta, 2024)](https://arxiv.org/abs/2404.19737)
- [MTP Needs Registers (2025)](https://arxiv.org/abs/2505.10518)
- [L-MTP (2025)](https://arxiv.org/abs/2505.17505)
- [Training-Free MTP (2026)](https://arxiv.org/abs/2603.17942)
