"""
Plot the runtime survivability results (paper Section 4.3).

Reads the three CSVs produced by runtime_simulation.py and generates one
figure per processor count for each experiment:

    Figure 5  - degraded job ratio vs. target utilization      (with 95% CI bars)
    Figure 6  - degraded job ratio vs. mode-switch probability  (with 95% CI bars)
    Figure 7  - degraded job ratio vs. migration overhead       (broken axis, CI bars)
    Lost-opt  - lost optional-execution ratio vs. target utilization (reviewer R2-6)

The CSVs now carry per-task-set DJR standard deviations (Std_*) and the
lost-optional ratios (Lost_*). Error bars are the 95% CI of the mean,
1.96 * Std / sqrt(N), where N is the number of task sets per point.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
RESULT_DIR = "/Users/jaewoo/data/com"

# Number of task sets per simulation point (= len(prepared) in the sim, MAX_SIM_SETS).
# Used only when the CSV has no explicit 'N_Sets' column.
N_SETS = 1000


def _n_sets(sub_df):
    """Per-point task-set count: prefer an explicit column, else the constant."""
    if "N_Sets" in sub_df.columns:
        return sub_df["N_Sets"].to_numpy()
    return N_SETS


def _ci95(std, n):
    """95% CI half-width of the mean. Swap for `return std` to show +/-1 std instead."""
    return 1.96 * np.asarray(std, dtype=float) / np.sqrt(n)


def plot_util(result_dir):
    csv_file_path = os.path.join(result_dir, "imc_simulation_recovery_results.csv")
    if not os.path.exists(csv_file_path):
        print(f"[skip] CSV not found: {csv_file_path}")
        return

    df = pd.read_csv(csv_file_path)
    for m in sorted(df['m'].unique()):
        sub_df = df[df['m'] == m]
        n = _n_sets(sub_df)

        plt.figure(figsize=(10, 6))
        plt.errorbar(sub_df["Target"], sub_df["Degrade_OFF"],
                     yerr=_ci95(sub_df["Std_OFF"], n),
                     marker='x', linestyle='--', color='red', linewidth=2,
                     capsize=3, label='Migration OFF')
        plt.errorbar(sub_df["Target"], sub_df["Degrade_Mig_Rec"],
                     yerr=_ci95(sub_df["Std_Mig_Rec"], n),
                     marker='o', linestyle='-', color='blue', linewidth=2,
                     capsize=3, label='Migration Rec')
        plt.errorbar(sub_df["Target"], sub_df["Degrade_Mig_NoRec"],
                     yerr=_ci95(sub_df["Std_Mig_NoRec"], n),
                     marker='^', linestyle='-.', color='green', linewidth=2,
                     capsize=3, label='Migration NoRec')

        plt.xlabel('Target Utilization / Core', fontsize=14)
        plt.ylabel('Degraded Job Ratio (%)', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(sub_df["Target"])

        max_y = max((sub_df["Degrade_OFF"] + _ci95(sub_df["Std_OFF"], n)).max(),
                    (sub_df["Degrade_Mig_Rec"] + _ci95(sub_df["Std_Mig_Rec"], n)).max(),
                    (sub_df["Degrade_Mig_NoRec"] + _ci95(sub_df["Std_Mig_NoRec"], n)).max())
        plt.ylim(-0.5, max_y + 2.0)
        plt.tight_layout()

        pdf_filename = f"imc_simulation_m{m}.pdf"
        plt.savefig(os.path.join(result_dir, pdf_filename), format='pdf', bbox_inches='tight')
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
        n = _n_sets(sub_df)

        plt.figure(figsize=(10, 6))
        plt.errorbar(sub_df["Prob"], sub_df["Degrade_OFF"],
                     yerr=_ci95(sub_df["Std_OFF"], n),
                     marker='x', linestyle='--', color='red', linewidth=2,
                     capsize=3, label='Migration OFF')
        plt.errorbar(sub_df["Prob"], sub_df["Degrade_Mig_Rec"],
                     yerr=_ci95(sub_df["Std_Mig_Rec"], n),
                     marker='o', linestyle='-', color='blue', linewidth=2,
                     capsize=3, label='Migration Rec')
        plt.errorbar(sub_df["Prob"], sub_df["Degrade_Mig_NoRec"],
                     yerr=_ci95(sub_df["Std_Mig_NoRec"], n),
                     marker='^', linestyle='-.', color='green', linewidth=2,
                     capsize=3, label='Migration NoRec')

        plt.xlabel('Mode Switch Probability', fontsize=14)
        plt.ylabel('Degraded Job Ratio (%)', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(sub_df["Prob"])

        max_y = max((sub_df["Degrade_OFF"] + _ci95(sub_df["Std_OFF"], n)).max(),
                    (sub_df["Degrade_Mig_Rec"] + _ci95(sub_df["Std_Mig_Rec"], n)).max(),
                    (sub_df["Degrade_Mig_NoRec"] + _ci95(sub_df["Std_Mig_NoRec"], n)).max())
        plt.ylim(-0.5, max_y + 2.0)
        plt.tight_layout()

        pdf_filename = f"imc_prob_m{m}.pdf"
        plt.savefig(os.path.join(result_dir, pdf_filename), format='pdf', bbox_inches='tight')
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
        n = _n_sets(sub_df)

        ci_off = _ci95(sub_df["Std_OFF"], n)
        ci_rec = _ci95(sub_df["Std_Mig_Rec"], n)
        ci_nor = _ci95(sub_df["Std_Mig_NoRec"], n)

        # Broken axis: Migration OFF sits far above Rec / NoRec.
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 6),
            gridspec_kw={'height_ratios': [1, 2]})
        fig.subplots_adjust(hspace=0.05)

        alpha_pct = [a * 100 for a in sub_df["Alpha"]]

        # Top axis: Migration OFF.
        c1 = ax1.errorbar(alpha_pct, sub_df["Degrade_OFF"], yerr=ci_off,
                          marker='s', linestyle='--', color='red', linewidth=2,
                          capsize=3, label='Migration OFF')
        off_min, off_max = sub_df["Degrade_OFF"].min(), sub_df["Degrade_OFF"].max()
        ax1.set_ylim(off_min - ci_off.max() - 0.2, off_max + ci_off.max() + 0.2)

        # Bottom axis: Rec, NoRec.
        c2 = ax2.errorbar(alpha_pct, sub_df["Degrade_Mig_Rec"], yerr=ci_rec,
                          marker='o', linestyle='-', color='blue', linewidth=2,
                          capsize=3, label='Migration Rec')
        c3 = ax2.errorbar(alpha_pct, sub_df["Degrade_Mig_NoRec"], yerr=ci_nor,
                          marker='^', linestyle='-', color='green', linewidth=2,
                          capsize=3, label='Migration NoRec')

        bot_min = min((sub_df["Degrade_Mig_Rec"] - ci_rec).min(),
                      (sub_df["Degrade_Mig_NoRec"] - ci_nor).min())
        bot_max = max((sub_df["Degrade_Mig_Rec"] + ci_rec).max(),
                      (sub_df["Degrade_Mig_NoRec"] + ci_nor).max())
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

        fig.text(0.04, 0.5, 'DJR (%)', va='center', rotation='vertical', fontsize=14)
        ax1.set_title(f'$m={m}$', fontsize=14)

        handles = [c1, c2, c3]
        labels = [h.get_label() for h in handles]
        ax2.legend(handles, labels, loc='center left', fontsize=14)

        ax1.grid(True, linestyle='-', alpha=0.3)
        ax2.grid(True, linestyle='-', alpha=0.3)

        pdf_filename = f"imc_overhead_m{m}.pdf"
        plt.savefig(os.path.join(result_dir, pdf_filename), format='pdf', bbox_inches='tight')
        plt.close()
        print(f"[Mig] Plot for m={m} saved: {pdf_filename}")


def plot_lost_optional(result_dir):
    """Reviewer R2-6: lost optional-execution ratio vs. target utilization."""
    csv_file_path = os.path.join(result_dir, "imc_simulation_recovery_results.csv")
    if not os.path.exists(csv_file_path):
        print(f"[skip] CSV not found: {csv_file_path}")
        return
    df = pd.read_csv(csv_file_path)
    if "Lost_Mig_Rec" not in df.columns:
        print("[skip] no Lost_* columns in CSV")
        return

    for m in sorted(df['m'].unique()):
        sub_df = df[df['m'] == m]

        plt.figure(figsize=(10, 6))
        plt.plot(sub_df["Target"], sub_df["Lost_OFF"], marker='x',
                 linestyle='--', color='red', linewidth=2, label='Migration OFF')
        plt.plot(sub_df["Target"], sub_df["Lost_Mig_Rec"], marker='o',
                 linestyle='-', color='blue', linewidth=2, label='Migration Rec')
        plt.plot(sub_df["Target"], sub_df["Lost_Mig_NoRec"], marker='^',
                 linestyle='-.', color='green', linewidth=2, label='Migration NoRec')

        plt.xlabel('Target Utilization / Core', fontsize=14)
        plt.ylabel('Lost Optional Execution (%)', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(sub_df["Target"])
        max_y = max(sub_df["Lost_OFF"].max(), sub_df["Lost_Mig_Rec"].max(),
                    sub_df["Lost_Mig_NoRec"].max())
        plt.ylim(-0.5, max_y + 2.0)
        plt.tight_layout()

        pdf_filename = f"imc_lostopt_m{m}.pdf"
        plt.savefig(os.path.join(result_dir, pdf_filename), format='pdf', bbox_inches='tight')
        plt.close()
        print(f"[Lost] Plot for m={m} saved: {pdf_filename}")


def main():
    result_dir = RESULT_DIR
    plot_util(result_dir)
    plot_prob(result_dir)
    plot_mig(result_dir)
    plot_lost_optional(result_dir)   # reviewer R2-6; comment out if not needed


if __name__ == "__main__":
    main()