# Noor-Edge Training Summary

| Phase | Steps | Start Loss | Final Loss | Min Loss | Avg tok/s | Duration |
|-------|-------|-----------|-----------|---------|-----------|----------|
| Phase 1: Base Pretrain | 2000–19990 | 7.4654 | 5.6430 | 1.0937 | 2852 | 248.1 min |

## Model Configuration
- **Model:** Noor-Edge (2.8B total, 430M active)
- **Architecture:** 24 layers, d_model=1024, PLE dim=128, GQA 8Q/2KV
- **Hardware:** RunPod A100 80GB
- **Precision:** f32
- **Tokenizer:** Borno v1 (64K BPE, Bangla-native)

## Figures
- `loss_curve.png` — Combined loss across all phases
- `loss_phase_1_base_pretrain.png` — Phase 1 detail
- `loss_phase_2_bangla_cc.png` — Phase 2 detail
- `throughput.png` — Tokens/sec over training
- `lr_schedule.png` — Learning rate schedule

*Generated: 2026-04-08 15:03*