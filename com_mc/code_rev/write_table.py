"""
Compute the migration / recovery count table for the paper (reviewer R2-3).

Reads the runtime CSVs produced by runtime_simulation.py (updated main(), which
emits Avg_Mig / Max_Mig / Avg_Rec / Max_Rec under the full IMC-PALM Migration Rec
protocol) and reports, per processor count m:

  * the default operating point  (Ut = 0.70, P^MS = 0.2, alpha = 0)  -> main table
  * the range of counts across the utilization sweep and the P^MS sweep -> context

"Per simulation" = per task-set run; Avg/Max are taken over all task sets at a point.

Outputs (written to RESULT_DIR):
  migration_counts_table.csv   tidy numbers
  migration_counts_table.tex   booktabs fragment ready to \\input
"""

import os
import pandas as pd
import numpy as np

RESULT_DIR = "/Users/jaewoo/data/com"

UTIL_CSV = "imc_simulation_recovery_results.csv"   # default P^MS=0.2, alpha=0; varies Ut
PROB_CSV = "imc_prob_recovery_results.csv"          # fixed Ut=0.70; varies P^MS
DEFAULT_UT = 0.70                                    # default operating point

CNT_COLS = ["Avg_Mig", "Max_Mig", "Avg_Rec", "Max_Rec"]


def _load(result_dir, fname):
    fp = os.path.join(result_dir, fname)
    if not os.path.exists(fp):
        print(f"[skip] CSV not found: {fp}")
        return None
    df = pd.read_csv(fp)
    if "Avg_Mig" not in df.columns:
        print(f"[skip] {fname} has no migration-count columns — "
              f"re-run runtime_simulation.py with the updated main().")
        return None
    return df


def default_point_table(df, xcol, xval):
    """One row per m at the default operating point."""
    rows = []
    for m in sorted(df["m"].unique()):
        sub = df[(df["m"] == m) & np.isclose(df[xcol], xval)]
        if sub.empty:
            print(f"[warn] m={m}: no row at {xcol}={xval}")
            continue
        r = sub.iloc[0]
        rows.append({"m": int(m),
                     "Avg_Mig": round(float(r["Avg_Mig"]), 2),
                     "Max_Mig": int(r["Max_Mig"]),
                     "Avg_Rec": round(float(r["Avg_Rec"]), 2),
                     "Max_Rec": int(r["Max_Rec"])})
    return pd.DataFrame(rows)


def range_summary(df, sweep_label):
    """Min–max of the counts across the sweep, per m (context, not the main table)."""
    rows = []
    for m in sorted(df["m"].unique()):
        sub = df[df["m"] == m]
        rows.append({"m": int(m), "sweep": sweep_label,
                     "Avg_Mig_min": round(sub["Avg_Mig"].min(), 2),
                     "Avg_Mig_max": round(sub["Avg_Mig"].max(), 2),
                     "Max_Mig": int(sub["Max_Mig"].max()),
                     "Avg_Rec_min": round(sub["Avg_Rec"].min(), 2),
                     "Avg_Rec_max": round(sub["Avg_Rec"].max(), 2),
                     "Max_Rec": int(sub["Max_Rec"].max())})
    return pd.DataFrame(rows)


def to_latex(tab):
    head = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Average and maximum numbers of migrations and recoveries per "
        "simulation under the full IMC-PALM (Migration Rec) protocol at the default "
        "operating point ($U_t=0.70$, $P^{MS}=0.2$, $\\alpha=0$).}\n"
        "\\label{tab:migration-counts}\n"
        "\\begin{tabular}{ccccc}\n"
        "\\toprule\n"
        "$m$ & Avg.\\ migrations & Max.\\ migrations & "
        "Avg.\\ recoveries & Max.\\ recoveries \\\\\n"
        "\\midrule\n"
    )
    body = "".join(
        f"{int(r.m)} & {r.Avg_Mig:.2f} & {int(r.Max_Mig)} & "
        f"{r.Avg_Rec:.2f} & {int(r.Max_Rec)} \\\\\n"
        for r in tab.itertuples()
    )
    tail = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return head + body + tail


def main():
    result_dir = RESULT_DIR
    util = _load(result_dir, UTIL_CSV)
    if util is None:
        return

    main_tab = default_point_table(util, "Target", DEFAULT_UT)
    print("\n=== Migration / recovery counts per simulation "
          f"(default point: Ut={DEFAULT_UT}, P^MS=0.2, alpha=0; Migration Rec) ===")
    print(main_tab.to_string(index=False))

    summaries = [range_summary(util, "utilization 0.70-0.95")]
    prob = _load(result_dir, PROB_CSV)
    if prob is not None:
        summaries.append(range_summary(prob, "P^MS 0.0-1.0"))
    range_tab = pd.concat(summaries, ignore_index=True)
    print("\n=== Range of counts across sweeps (context) ===")
    print(range_tab.to_string(index=False))

    main_tab.to_csv(os.path.join(result_dir, "migration_counts_table.csv"), index=False)
    with open(os.path.join(result_dir, "migration_counts_table.tex"), "w") as f:
        f.write(to_latex(main_tab))
    print(f"\nSaved: migration_counts_table.csv, migration_counts_table.tex (in {result_dir})")


if __name__ == "__main__":
    main()