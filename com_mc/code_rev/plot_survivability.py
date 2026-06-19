"""
Plot the runtime survivability results (paper Section 4.3).

Reads the three CSVs produced by runtime_simulation.py and generates one
figure per processor count for each experiment:

    Figure 5  - degraded job ratio vs. target utilization
    Figure 6  - degraded job ratio vs. mode-switch probability
    Figure 7  - degraded job ratio vs. migration overhead (broken axis)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
RESULT_DIR = "/Users/jaewoo/data/com"


def plot_util(result_dir):
    csv_file_path = os.path.join(result_dir, "imc_simulation_recovery_results.csv")
    if not os.path.exists(csv_file_path):
        print(f"[skip] CSV not found: {csv_file_path}")
        return

    df = pd.read_csv(csv_file_path)
    for m in sorted(df['m'].unique()):
        sub_df = df[df['m'] == m]

        plt.figure(figsize=(10, 6))
        plt.plot(sub_df["Target"], sub_df["Degrade_OFF"], marker='x',
                 linestyle='--', color='red', linewidth=2,
                 label='Migration OFF')
        plt.plot(sub_df["Target"], sub_df["Degrade_Mig_Rec"], marker='o',
                 linestyle='-', color='blue', linewidth=2,
                 label='Migration Rec')
        plt.plot(sub_df["Target"], sub_df["Degrade_Mig_NoRec"], marker='^',
                 linestyle='-.', color='green', linewidth=2,
                 label='Migration NoRec')

        plt.xlabel('Target Utilization / Core', fontsize=14)
        plt.ylabel('Degraded Job Ratio (%)', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(sub_df["Target"])

        max_y = max(sub_df["Degrade_OFF"].max(),
                    sub_df["Degrade_Mig_Rec"].max(),
                    sub_df["Degrade_Mig_NoRec"].max())
        plt.ylim(-0.5, max_y + 2.0)
        plt.tight_layout()

        pdf_filename = f"imc_simulation_m{m}.pdf"
        pdf_file_path = os.path.join(result_dir, pdf_filename)
        plt.savefig(pdf_file_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"[Util] Plot for m={m} saved: {pdf_filename}")


def plot_prob(result_dir):
    csv_file_path = os.path.join(result_dir, "imc_prob_recovery_results.csv")
    if not os.path.exists(csv_file_path):
        print(f"[skip] CSV not found: {csv_file_path}")
        return

    df = pd.read_csv(csv_file_path)
    for m in sorted(df['m'].unique()):
        sub_df = df[df['m'] == m]

        plt.figure(figsize=(10, 6))
        plt.plot(sub_df["Prob"], sub_df["Degrade_OFF"], marker='x',
                 linestyle='--', color='red', linewidth=2,
                 label='Migration OFF')
        plt.plot(sub_df["Prob"], sub_df["Degrade_Mig_Rec"], marker='o',
                 linestyle='-', color='blue', linewidth=2,
                 label='Migration Rec')
        plt.plot(sub_df["Prob"], sub_df["Degrade_Mig_NoRec"], marker='^',
                 linestyle='-.', color='green', linewidth=2,
                 label='Migration NoRec')

        plt.xlabel('Mode Switch Probability', fontsize=14)
        plt.ylabel('Degraded Job Ratio (%)', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(sub_df["Prob"])

        max_y = max(sub_df["Degrade_OFF"].max(),
                    sub_df["Degrade_Mig_Rec"].max(),
                    sub_df["Degrade_Mig_NoRec"].max())
        plt.ylim(-0.5, max_y + 2.0)

        pdf_filename = f"imc_prob_m{m}.pdf"
        pdf_file_path = os.path.join(result_dir, pdf_filename)
        plt.savefig(pdf_file_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"[Prob] Plot for m={m} saved: {pdf_filename}")


def plot_mig(result_dir):
    csv_file_path = os.path.join(result_dir, "imc_overhead_recovery_results.csv")
    if not os.path.exists(csv_file_path):
        print(f"[skip] CSV not found: {csv_file_path}")
        return

    df = pd.read_csv(csv_file_path)
    for m in sorted(df['m'].unique()):
        sub_df = df[df['m'] == m]

        # Broken axis: Migration OFF sits far above Rec / NoRec.
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 6),
            gridspec_kw={'height_ratios': [1, 2]})
        fig.subplots_adjust(hspace=0.05)

        alpha_pct = [a * 100 for a in sub_df["Alpha"]]

        # Top axis: Migration OFF.
        line1 = ax1.plot(alpha_pct, sub_df["Degrade_OFF"], marker='s',
                         linestyle='--', color='red', linewidth=2,
                         label='Migration OFF')
        off_min, off_max = sub_df["Degrade_OFF"].min(), sub_df["Degrade_OFF"].max()
        ax1.set_ylim(off_min - 0.2, off_max + 0.2)

        # Bottom axis: Rec, NoRec.
        line2 = ax2.plot(alpha_pct, sub_df["Degrade_Mig_Rec"], marker='o',
                         linestyle='-', color='blue', linewidth=2,
                         label='Migration Rec')
        line3 = ax2.plot(alpha_pct, sub_df["Degrade_Mig_NoRec"], marker='^',
                         linestyle='-', color='green', linewidth=2,
                         label='Migration NoRec')

        bot_min = min(sub_df["Degrade_Mig_Rec"].min(),
                      sub_df["Degrade_Mig_NoRec"].min())
        bot_max = max(sub_df["Degrade_Mig_Rec"].max(),
                      sub_df["Degrade_Mig_NoRec"].max())
        ax2.set_ylim(bot_min - 0.1, bot_max + 0.1)

        # Broken-axis diagonal marks.
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.tick_params(labelbottom=False, bottom=False)

        d = 0.015
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ax2.set_xlabel('Migration Overhead $\\alpha$ (%)', fontsize=14)
        ax2.set_xticks(alpha_pct)

        fig.text(0.04, 0.5, 'DJR (%)', va='center', rotation='vertical',
                 fontsize=14)
        ax1.set_title(f'$m={m}$', fontsize=14)

        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='center left', fontsize=14)

        ax1.grid(True, linestyle='-', alpha=0.3)
        ax2.grid(True, linestyle='-', alpha=0.3)

        pdf_filename = f"imc_overhead_m{m}.pdf"
        pdf_file_path = os.path.join(result_dir, pdf_filename)
        plt.savefig(pdf_file_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"[Mig] Plot for m={m} saved: {pdf_filename}")


def main():
    result_dir = RESULT_DIR
    plot_util(result_dir)
    plot_prob(result_dir)
    plot_mig(result_dir)


if __name__ == "__main__":
    main()
