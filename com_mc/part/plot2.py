#!/usr/bin/env python3
"""
IMC Partitioning Results Plotter

Reads partitioning_results.csv and generates one acceptance ratio
graph per processor count.

Usage:
  - Eclipse / IDE: Edit the PARAMETERS section below, then Run.
  - Terminal:      python imc_plot_results.py
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ╔══════════════════════════════════════════════════════════════╗
# ║                    PARAMETERS (EDIT HERE)                    ║
# ╚══════════════════════════════════════════════════════════════╝

INPUT_CSV  = "/Users/jaewoo/data/com/partitioning_results.csv"
OUTPUT_DIR = "/Users/jaewoo/data/com"          # Graphs saved here
FIG_FORMAT = "pdf"                              # "pdf", "png", or "eps"
DPI        = 300                                # Resolution (for png)

# ╔══════════════════════════════════════════════════════════════╗
# ║                  END OF PARAMETERS                           ║
# ╚══════════════════════════════════════════════════════════════╝


# ============================================================
# Style
# ============================================================

ALGO_STYLES = {
    "FFD":    {"marker": "o", "linestyle": "-",  "color": "#1f77b4"},
    "BFD":    {"marker": "s", "linestyle": "--", "color": "#ff7f0e"},
    "WFD":    {"marker": "^", "linestyle": "-.", "color": "#2ca02c"},
    "CU-UDP": {"marker": "D", "linestyle": ":",  "color": "#d62728"},
}


# ============================================================
# Load
# ============================================================

def load_results(filepath):
    """
    Returns:
        data[m] = {"util_caps": [...], "FFD": [...], "BFD": [...], ...}
        algorithms: list of algorithm names found in the header
    """
    data = defaultdict(lambda: defaultdict(list))
    algorithms = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # header = ["num_processors", "util_cap", "FFD", "BFD", "WFD", ...]
        algorithms = header[2:]

        for row in reader:
            if not row:
                continue
            m = int(row[0])
            cap = float(row[1])
            data[m]["util_caps"].append(cap)
            for i, alg in enumerate(algorithms):
                data[m][alg].append(float(row[2 + i]))

    return dict(data), algorithms


# ============================================================
# Plot
# ============================================================

def plot_per_processor(data, algorithms, output_dir, fig_format, dpi):
    os.makedirs(output_dir, exist_ok=True)

    for m in sorted(data.keys()):
        d = data[m]
        caps = d["util_caps"]

        fig, ax = plt.subplots(figsize=(6, 4))

        for alg in algorithms:
            style = ALGO_STYLES.get(alg, {})
            ax.plot(caps, d[alg],
                    label=alg,
                    marker=style.get("marker", "o"),
                    linestyle=style.get("linestyle", "-"),
                    color=style.get("color", None),
                    markersize=6, linewidth=1.8)

        ax.set_xlabel("Utilization Cap", fontsize=12)
        ax.set_ylabel("Acceptance Ratio", fontsize=12)
        ax.set_title(f"m = {m} processors", fontsize=13)
        ax.set_ylim(0.3, 1.05)
        ax.set_xlim(caps[0] - 0.05 * m, caps[-1] + 0.05 * m)
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        filename = f"acceptance_ratio_m{m}.{fig_format}"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filepath}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("IMC Partitioning Results Plotter")
    print("=" * 60)
    print(f"Input:  {INPUT_CSV}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Format: {FIG_FORMAT}  DPI: {DPI}")
    print()

    data, algorithms = load_results(INPUT_CSV)
    print(f"Loaded data for m = {sorted(data.keys())}")
    print(f"Algorithms: {algorithms}")
    print()

    plot_per_processor(data, algorithms, OUTPUT_DIR, FIG_FORMAT, DPI)

    print(f"\nDone!")


if __name__ == "__main__":
    main()