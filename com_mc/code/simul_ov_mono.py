#!/usr/bin/env python3
# ============================================================
#  Overhead Sensitivity — Statistical Significance Harness
#
#  PURPOSE
#  -------
#  The original simul_unified2.py runs the overhead experiment
#  (Fig. 7) ONCE with random.seed(42) fixed per task set, then
#  SUMS the per-set degraded-job counts into a single ratio per
#  (m, alpha). That produces one point estimate with no variance,
#  so the small (~0.2 %p) non-monotonicity of Migration NoRec in
#  alpha cannot be tested for statistical significance.
#
#  This harness reuses the EXACT scheduling / migration / recovery
#  logic from simul_unified2.py (imported unchanged) and only
#  changes the OUTER experiment loop:
#
#    * The mode-switch RNG is reseeded with a DIFFERENT seed on
#      every (seed, task_set, mode) run, so each seed is an
#      independent realization of the i.i.d. mode-switch process.
#    * For each seed we record the system-wide degraded job ratio
#      per (m, alpha, mode) -> a SAMPLE, not a single value.
#    * Across seeds we compute mean, std, a paired bootstrap CI
#      for the alpha=0 -> alpha=1% NoRec change ("the rise"), and
#      a sign/Wilcoxon-style summary for the rise-then-fall shape.
#
#  IMPORTANT — reproducibility caveat
#  ----------------------------------
#  This script GENERATES synthetic task sets from the procedure
#  described in Section 4.1 of the paper, because the original
#  pre-generated JSON workloads (stasks_m_*_target_*.json) are not
#  bundled. Absolute degraded-job-ratio values will therefore NOT
#  match the paper's Fig. 7 numbers (e.g. 11.94 %); they depend on
#  the workload RNG. The question this harness answers is
#  scale-invariant: "is the NoRec non-monotonicity in alpha larger
#  than the seed-to-seed sampling variation?" — and that conclusion
#  does not require reproducing the exact absolute numbers.
#
#  USAGE
#    python3 overhead_significance.py --seeds 30 --max-sets 200 \
#            --sim-ticks 10000 --out results_overhead_sig.csv
# ============================================================

import os
import sys
import csv
import math
import random
import argparse
import statistics

# --- import the ORIGINAL simulator logic unchanged ----------
# We import by file path so the original file needs no edits.
import importlib.util

def load_original(sim_path):
    spec = importlib.util.spec_from_file_location("simul_unified3", sim_path)
    mod = importlib.util.module_from_spec(spec)
    # Prevent the original main() from executing on import.
    src = open(sim_path).read()
    src = src.replace('if __name__ == "__main__":\n    main()',
                       'if __name__ == "__main__":\n    pass')
    exec(compile(src, sim_path, "exec"), mod.__dict__)
    return mod


# ------------------------------------------------------------
#  Workload generation — verbatim from Section 4.1 of the paper
# ------------------------------------------------------------
def gen_task_set(m, U_t, wl_rng):
    """Generate one synthetic task set per the paper's Section 4.1.

    - period log-uniform in [10, 500]
    - P(HC) = 0.5
    - HC: u^H ~ U[0.05, 0.30], u^L = u^H / R, R ~ U[1.0, 3.0]
    - LC: u^A ~ U[0.05, 0.30], u^D ~ U[0.001, u^A/2]
    - n ~ U[4m, 6m]
    - then uniformly scale all utilizations so that
      max(collective LO util, collective HI util) == m * U_t
    """
    n = wl_rng.randint(4 * m, 6 * m)
    tasks = []
    for i in range(n):
        log_lo, log_hi = math.log(10.0), math.log(500.0)
        period = int(round(math.exp(wl_rng.uniform(log_lo, log_hi))))
        period = max(10, min(500, period))
        is_hc = (wl_rng.random() < 0.5)
        if is_hc:
            u_hi = wl_rng.uniform(0.05, 0.30)
            R = wl_rng.uniform(1.0, 3.0)
            u_lo = u_hi / R                     # u^L <= u^H for HC
            tasks.append({"id": i, "crit": "HC", "period": period,
                          "u_LO": u_lo, "u_HI": u_hi})
        else:
            u_a = wl_rng.uniform(0.05, 0.30)
            u_d = wl_rng.uniform(0.001, u_a / 2.0)
            # In this simulator's task dict convention, for an LC task
            # u_LO carries the ACTIVE budget and u_HI carries the
            # DEGRADED (mandatory) budget (see Processor.add / try_add).
            tasks.append({"id": i, "crit": "LC", "period": period,
                          "u_LO": u_a, "u_HI": u_d})

    # collective utilizations
    U_LO = sum((t["u_LO"] if t["crit"] == "HC" else t["u_LO"]) for t in tasks
               if t["crit"] == "HC") + \
           sum(t["u_LO"] for t in tasks if t["crit"] == "LC")
    U_HI = sum(t["u_HI"] for t in tasks if t["crit"] == "HC") + \
           sum(t["u_HI"] for t in tasks if t["crit"] == "LC")
    cur_max = max(U_LO, U_HI)
    if cur_max <= 0:
        return None
    scale = (m * U_t) / cur_max
    for t in tasks:
        t["u_LO"] *= scale
        t["u_HI"] *= scale
        t["c_LO"] = max(1, int(round(t["u_LO"] * t["period"])))
        t["c_HI"] = max(1, int(round(t["u_HI"] * t["period"])))
    return tasks


def build_prepared(orig, m, U_t, n_sets, wl_seed):
    """Generate and FFD-prepare n_sets schedulable task sets."""
    wl_rng = random.Random(wl_seed)
    prepared = []
    attempts = 0
    while len(prepared) < n_sets and attempts < n_sets * 50:
        attempts += 1
        ts = gen_task_set(m, U_t, wl_rng)
        if ts is None:
            continue
        import copy
        if orig.partition_ffd_new(copy.deepcopy(ts), m) is None:
            continue
        prepared.append(copy.deepcopy(ts))
    return prepared


# ------------------------------------------------------------
#  One realization: run all modes for one mode-switch seed
# ------------------------------------------------------------
def eval_point_seeded(orig, prepared, m, sim_ticks, switch_prob, alpha,
                       ms_seed):
    """Same aggregation as orig.eval_point, but the mode-switch RNG
    is seeded with ms_seed (a DIFFERENT seed per realization) instead
    of the fixed 42, giving an independent draw of the i.i.d.
    mode-switch process. Returns system-wide DJR (%) per mode."""
    import copy
    acc = {mode: {"total": 0, "degrade": 0}
           for mode in ["off", "mig_rec", "mig_norec"]}
    for task_set in prepared:
        for mode in ["off", "mig_rec", "mig_norec"]:
            random.seed(ms_seed)            # <-- the only change vs orig
            t_total, d_total = orig.run_simulation(
                task_set, m, sim_ticks,
                mig_mode=mode, switch_prob=switch_prob, mig_alpha=alpha)
            acc[mode]["total"] += t_total
            acc[mode]["degrade"] += d_total
    if acc["off"]["total"] == 0:
        return None
    return {
        "off":   100.0 * acc["off"]["degrade"]   / acc["off"]["total"],
        "rec":   100.0 * acc["mig_rec"]["degrade"]   / acc["mig_rec"]["total"],
        "norec": 100.0 * acc["mig_norec"]["degrade"] / acc["mig_norec"]["total"],
    }


# ------------------------------------------------------------
#  Bootstrap paired CI
# ------------------------------------------------------------
def paired_bootstrap_ci(diffs, n_boot=20000, ci=0.95, rng=None):
    """Percentile bootstrap CI for the mean of a paired difference
    sample (one diff per seed)."""
    if rng is None:
        rng = random.Random(12345)
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot)]
    return (statistics.fmean(diffs), lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-path",
                    default="simul_unified2.py",
                    help="path to the original simul_unified2.py")
    ap.add_argument("--seeds", type=int, default=30,
                    help="number of independent mode-switch seeds")
    ap.add_argument("--max-sets", type=int, default=200,
                    help="task sets per (m) point")
    ap.add_argument("--sim-ticks", type=int, default=10000)
    ap.add_argument("--switch-prob", type=float, default=0.20)
    ap.add_argument("--U-t", type=float, default=0.70)
    ap.add_argument("--m-values", default="2,4,8")
    ap.add_argument("--alphas", default="0.0,0.01,0.03,0.05,0.10")
    ap.add_argument("--wl-seed", type=int, default=20260518,
                    help="seed for synthetic workload generation "
                         "(fixed so the SAME task sets are used "
                         "across all alpha/seed combinations)")
    ap.add_argument("--out", default="results_overhead_sig.csv")
    ap.add_argument("--per-seed-out", default="results_overhead_perseed.csv")
    args = ap.parse_args()

    orig = load_original(args.sim_path)
    m_values = [int(x) for x in args.m_values.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    # Per-seed long-format records
    per_seed_rows = []
    # Aggregated summary rows
    summary_rows = []

    for m in m_values:
        # Same workload for every (alpha, seed) at this m, so the only
        # varying factor is the mode-switch realization (paired design).
        prepared = build_prepared(orig, m, args.U_t, args.max_sets,
                                   args.wl_seed + m)
        if len(prepared) == 0:
            print(f"[warn] m={m}: no schedulable task sets generated",
                  file=sys.stderr)
            continue
        print(f"[info] m={m}: {len(prepared)} task sets prepared",
              file=sys.stderr)

        # norec_samples[alpha] = [djr_seed0, djr_seed1, ...]
        norec_samples = {a: [] for a in alphas}
        rec_samples = {a: [] for a in alphas}
        off_samples = {a: [] for a in alphas}

        for si in range(args.seeds):
            ms_seed = 1000 + si      # distinct, reproducible seeds
            for a in alphas:
                r = eval_point_seeded(orig, prepared, m, args.sim_ticks,
                                      args.switch_prob, a, ms_seed)
                if r is None:
                    continue
                norec_samples[a].append(r["norec"])
                rec_samples[a].append(r["rec"])
                off_samples[a].append(r["off"])
                per_seed_rows.append([m, a, si, ms_seed,
                                      r["off"], r["rec"], r["norec"]])
            print(f"[info] m={m} seed#{si} done", file=sys.stderr)

        # ---- summary statistics per (m, alpha) ----
        for a in alphas:
            ns = norec_samples[a]
            rs = rec_samples[a]
            if not ns:
                continue
            summary_rows.append([
                m, a, len(ns),
                statistics.fmean(off_samples[a]),
                statistics.fmean(rs),
                statistics.pstdev(rs) if len(rs) > 1 else 0.0,
                statistics.fmean(ns),
                statistics.pstdev(ns) if len(ns) > 1 else 0.0,
            ])

        # ---- the key hypothesis test ----
        # "the rise": paired difference NoRec(alpha=alpha1) - NoRec(alpha=0)
        if len(alphas) >= 2 and norec_samples[alphas[0]] and \
           norec_samples[alphas[1]]:
            a0, a1 = alphas[0], alphas[1]
            n = min(len(norec_samples[a0]), len(norec_samples[a1]))
            diffs_rise = [norec_samples[a1][i] - norec_samples[a0][i]
                          for i in range(n)]
            mean_d, lo, hi = paired_bootstrap_ci(diffs_rise)
            # also the rise-then-fall: NoRec(a1) - NoRec(a_max)
            amax = alphas[-1]
            n2 = min(len(norec_samples[a1]), len(norec_samples[amax]))
            diffs_fall = [norec_samples[a1][i] - norec_samples[amax][i]
                          for i in range(n2)]
            mean_f, lof, hif = paired_bootstrap_ci(diffs_fall)
            n_pos = sum(1 for d in diffs_rise if d > 0)
            print(f"\n=== m={m}: NoRec non-monotonicity test ===")
            print(f"  RISE  NoRec(a={a1}) - NoRec(a=0):  "
                  f"mean={mean_d:+.4f} %p  95%CI=[{lo:+.4f}, {hi:+.4f}]  "
                  f"sign>0 in {n_pos}/{n} seeds  "
                  f"-> {'significant' if (lo>0 or hi<0) else 'NOT significant (CI spans 0)'}")
            print(f"  FALL  NoRec(a={a1}) - NoRec(a={amax}): "
                  f"mean={mean_f:+.4f} %p  95%CI=[{lof:+.4f}, {hif:+.4f}]  "
                  f"-> {'significant' if (lof>0 or hif<0) else 'NOT significant (CI spans 0)'}")

    # ---- write outputs ----
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "Alpha", "N_seeds",
                    "OFF_mean",
                    "Rec_mean", "Rec_std",
                    "NoRec_mean", "NoRec_std"])
        for row in summary_rows:
            w.writerow([row[0], row[1], row[2],
                        f"{row[3]:.6f}",
                        f"{row[4]:.6f}", f"{row[5]:.6f}",
                        f"{row[6]:.6f}", f"{row[7]:.6f}"])

    with open(args.per_seed_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "Alpha", "SeedIdx", "MS_Seed",
                    "DJR_OFF", "DJR_Rec", "DJR_NoRec"])
        for row in per_seed_rows:
            w.writerow([row[0], row[1], row[2], row[3],
                        f"{row[4]:.6f}", f"{row[5]:.6f}", f"{row[6]:.6f}"])

    print(f"\n[done] summary -> {args.out}")
    print(f"[done] per-seed -> {args.per_seed_out}")


if __name__ == "__main__":
    main()