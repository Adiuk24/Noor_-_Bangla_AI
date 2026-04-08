#!/usr/bin/env python3
"""
Noor Training Log → Publication Graphs
Generates loss curves, throughput, and LR schedule plots for paper.
Usage: python3 scripts/plot_training.py --logs logs/ --output paper/figures/
"""

import re
import os
import sys
import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("Install matplotlib: pip3 install matplotlib")
    sys.exit(1)

# Paper-quality style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

PHASE_COLORS = {
    'Phase 1: Base Pretrain': '#2196F3',
    'Phase 2: Bangla CC': '#4CAF50',
    'Phase 3: Reasoning': '#FF9800',
    'Phase 4: Instruction': '#9C27B0',
}


def parse_log(filepath):
    """Parse a Noor training log file into structured data."""
    steps, losses, lrs, toks, elapsed = [], [], [], [], []

    pattern = re.compile(
        r'step=\s*(\d+)\s*\|\s*loss=([\d.]+)\s*\|\s*lr=([\d.e+-]+)\s*\|\s*'
        r'([\d.]+)s/step\s*\|\s*(\d+)\s*tok/s\s*\|\s*([\d.]+)min'
    )

    with open(filepath) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                lrs.append(float(m.group(3)))
                toks.append(int(m.group(5)))
                elapsed.append(float(m.group(6)))

    return {
        'steps': steps,
        'losses': losses,
        'lrs': lrs,
        'toks_per_sec': toks,
        'elapsed_min': elapsed,
    }


def smooth(values, window=50):
    """Simple moving average for smoother curves."""
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def plot_loss_curve(phases, output_dir):
    """Plot training loss across all phases."""
    fig, ax = plt.subplots()

    global_step_offset = 0
    phase_boundaries = []

    for phase_name, data in phases.items():
        color = PHASE_COLORS.get(phase_name, '#666666')
        steps = [s + global_step_offset for s in data['steps']]

        # Raw loss (transparent)
        ax.plot(steps, data['losses'], alpha=0.15, color=color, linewidth=0.5)
        # Smoothed loss
        ax.plot(steps, smooth(data['losses']), color=color, linewidth=2, label=phase_name)

        if global_step_offset > 0:
            phase_boundaries.append(global_step_offset)

        if steps:
            global_step_offset = steps[-1]

    for boundary in phase_boundaries:
        ax.axvline(x=boundary, color='#999999', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Noor-Edge (430M) — Training Loss')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)

    filepath = os.path.join(output_dir, 'loss_curve.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_loss_per_phase(phases, output_dir):
    """Individual loss plot per phase (for detailed analysis)."""
    paths = []
    for phase_name, data in phases.items():
        if not data['steps']:
            continue

        fig, ax = plt.subplots()
        color = PHASE_COLORS.get(phase_name, '#2196F3')

        ax.plot(data['steps'], data['losses'], alpha=0.2, color=color, linewidth=0.5)
        ax.plot(data['steps'], smooth(data['losses']), color=color, linewidth=2)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'Noor-Edge — {phase_name}')
        ax.set_ylim(bottom=0)

        # Add min/max/final annotations
        min_loss = min(data['losses'])
        max_loss = max(data['losses'])
        final_loss = data['losses'][-1]
        start_loss = data['losses'][0]

        ax.annotate(f'Start: {start_loss:.2f}', xy=(data['steps'][0], start_loss),
                    fontsize=9, color='gray')
        ax.annotate(f'Final: {final_loss:.2f}', xy=(data['steps'][-1], final_loss),
                    fontsize=9, color=color, fontweight='bold',
                    xytext=(-60, 15), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=color))

        safe_name = phase_name.lower().replace(' ', '_').replace(':', '')
        filepath = os.path.join(output_dir, f'loss_{safe_name}.png')
        fig.savefig(filepath)
        plt.close(fig)
        print(f"  Saved: {filepath}")
        paths.append(filepath)
    return paths


def plot_throughput(phases, output_dir):
    """Plot tokens/sec throughput across phases."""
    fig, ax = plt.subplots()

    global_step_offset = 0

    for phase_name, data in phases.items():
        color = PHASE_COLORS.get(phase_name, '#666666')
        steps = [s + global_step_offset for s in data['steps']]

        ax.plot(steps, smooth(data['toks_per_sec'], window=30),
                color=color, linewidth=1.5, label=phase_name)

        if steps:
            global_step_offset = steps[-1]

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Tokens/sec')
    ax.set_title('Noor-Edge — Training Throughput (A100 80GB)')
    ax.legend(loc='lower right')

    filepath = os.path.join(output_dir, 'throughput.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_lr_schedule(phases, output_dir):
    """Plot learning rate schedule across phases."""
    fig, ax = plt.subplots()

    global_step_offset = 0

    for phase_name, data in phases.items():
        color = PHASE_COLORS.get(phase_name, '#666666')
        steps = [s + global_step_offset for s in data['steps']]

        ax.plot(steps, data['lrs'], color=color, linewidth=2, label=phase_name)

        if steps:
            global_step_offset = steps[-1]

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Noor-Edge — Learning Rate Schedule')
    ax.legend(loc='upper right')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))

    filepath = os.path.join(output_dir, 'lr_schedule.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def generate_summary_table(phases, output_dir):
    """Generate a markdown summary table."""
    lines = [
        "# Noor-Edge Training Summary",
        "",
        "| Phase | Steps | Start Loss | Final Loss | Min Loss | Avg tok/s | Duration |",
        "|-------|-------|-----------|-----------|---------|-----------|----------|",
    ]

    for phase_name, data in phases.items():
        if not data['steps']:
            continue
        steps = f"{data['steps'][0]}–{data['steps'][-1]}"
        start_loss = f"{data['losses'][0]:.4f}"
        final_loss = f"{data['losses'][-1]:.4f}"
        min_loss = f"{min(data['losses']):.4f}"
        avg_toks = f"{sum(data['toks_per_sec']) // len(data['toks_per_sec'])}"
        duration = f"{data['elapsed_min'][-1]:.1f} min"
        lines.append(f"| {phase_name} | {steps} | {start_loss} | {final_loss} | {min_loss} | {avg_toks} | {duration} |")

    lines.extend([
        "",
        "## Model Configuration",
        "- **Model:** Noor-Edge (2.8B total, 430M active)",
        "- **Architecture:** 24 layers, d_model=1024, PLE dim=128, GQA 8Q/2KV",
        "- **Hardware:** RunPod A100 80GB",
        "- **Precision:** f32",
        "- **Tokenizer:** Borno v1 (64K BPE, Bangla-native)",
        "",
        "## Figures",
        "- `loss_curve.png` — Combined loss across all phases",
        "- `loss_phase_1_base_pretrain.png` — Phase 1 detail",
        "- `loss_phase_2_bangla_cc.png` — Phase 2 detail",
        "- `throughput.png` — Tokens/sec over training",
        "- `lr_schedule.png` — Learning rate schedule",
        "",
        f"*Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ])

    filepath = os.path.join(output_dir, 'training_summary.md')
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Generate Noor training graphs for paper')
    parser.add_argument('--logs', default='logs/', help='Directory containing training logs')
    parser.add_argument('--output', default='paper/figures/', help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Discover and parse logs
    phases = {}
    log_dir = Path(args.logs)

    log_map = {
        'phase1_base_20k.log': 'Phase 1: Base Pretrain',
        'phase2_bangla.log': 'Phase 2: Bangla CC',
        'phase2_bangla_final.log': 'Phase 2: Bangla CC',
        'phase3_reasoning.log': 'Phase 3: Reasoning',
        'phase4_instruction.log': 'Phase 4: Instruction',
    }

    for filename, phase_name in log_map.items():
        filepath = log_dir / filename
        if filepath.exists():
            print(f"Parsing: {filepath}")
            data = parse_log(filepath)
            if data['steps']:
                phases[phase_name] = data
                print(f"  → {len(data['steps'])} steps, loss {data['losses'][0]:.2f} → {data['losses'][-1]:.2f}")

    if not phases:
        print(f"No logs found in {args.logs}/")
        print(f"Expected files: {list(log_map.keys())}")
        sys.exit(1)

    print(f"\nGenerating figures → {args.output}")
    plot_loss_curve(phases, args.output)
    plot_loss_per_phase(phases, args.output)
    plot_throughput(phases, args.output)
    plot_lr_schedule(phases, args.output)
    generate_summary_table(phases, args.output)

    print("\nDone. All figures ready for paper.")


if __name__ == '__main__':
    main()
