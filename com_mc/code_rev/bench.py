"""
Micro-benchmark for the runtime admission check and mode-switch decision latency
(reviewer R2-3).

Measures, on representative partitioned task-set states:
  (1) admission-check cost   - one Processor.try_add_fixed_x(task) call
  (2) decision latency       - the full first-fit migration scan triggered by a
                               single mode switch: for each LC task on the affected
                               processor, scan the (m-1) neighbours until one admits
                               it (Algorithm 2). Worst case is n_L * (m-1) checks.
                               Neighbour utilisation is held fixed across candidates
                               (each evaluated independently) for a clean, repeatable
                               estimate; the n_L*(m-1) bound covers the adversarial scan.

Prerequisite: runtime_simulation.py must be importable. Remove the
`from code_rev.config import DATA_DIR` line at its top first; main() there is guarded
by `if __name__ == "__main__"`, so importing it does NOT run the simulation sweeps.

IMPORTANT: numbers are measured on the *host* CPU running this script, in pure
Python. A production scheduler (C, on the target ECU) is far faster, and the target
clock differs from the host. Treat the absolute figures as an upper bound on the
algorithmic cost of this reference implementation; for on-target numbers, run the
equivalent loop on the deployment hardware. Set CPU_GHZ to the host clock to read
the cycle estimate.
"""

import time
import gc
import random
import statistics

from code_rev.runtime_simulation import Processor, partition_ffd_new

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
M_VALUES = [2, 4, 8]
UT = 0.70                 # default operating point
P_H = 0.5                 # HC-task probability (paper Section 4.1)
N_SETS = 20               # partitioned task sets per m used to build realistic states
ADM_INNER = 2000          # admission calls per timing window
ADM_REPEATS = 50
DEC_REPEATS = 200         # repeats per decision unit
CPU_GHZ = 1.0             # set to your host clock for the cycle estimate
SEED0 = 12345


# ------------------------------------------------------------------
# Workload generation (faithful to paper Section 4.1; only u_LO/u_HI/crit matter here)
# ------------------------------------------------------------------
def gen_taskset(m, Ut, seed):
    rng = random.Random(seed)
    n = rng.randint(4 * m, 6 * m)
    tasks = []
    for i in range(n):
        if rng.random() < P_H:                       # HC: u_L = u_H / R, R in [1,3]
            u_h = rng.uniform(0.05, 0.30)
            u_l = u_h / rng.uniform(1.0, 3.0)
            tasks.append({"id": i, "crit": "HC", "u_LO": u_l, "u_HI": u_h})
        else:                                        # LC: u_LO=active, u_HI=degraded
            u_a = rng.uniform(0.05, 0.30)
            u_d = rng.uniform(0.001, u_a / 2.0)
            tasks.append({"id": i, "crit": "LC", "u_LO": u_a, "u_HI": u_d})
    # scale so max(collective LO, collective HI) == m * Ut
    U_lo = sum(t["u_LO"] if t["crit"] == "HC" else t["u_LO"] for t in tasks)
    U_hi = sum(t["u_HI"] if t["crit"] == "HC" else t["u_HI"] for t in tasks)
    peak = max(U_lo, U_hi)
    if peak > 0:
        s = (m * Ut) / peak
        for t in tasks:
            t["u_LO"] *= s
            t["u_HI"] *= s
    return tasks


def gen_partitioned(m, Ut, k, seed0):
    """Return k successfully partitioned processor lists for processor count m."""
    out = []
    s = seed0
    attempts = 0
    while len(out) < k and attempts < k * 50:
        attempts += 1
        tasks = gen_taskset(m, Ut, s)
        s += 1
        procs = partition_ffd_new([dict(t) for t in tasks], m)
        if procs is not None:
            out.append((procs, tasks))
    return out


# ------------------------------------------------------------------
# Scenario construction
# ------------------------------------------------------------------
def admission_scenarios(partitioned):
    """List of (target_processor, candidate_LC_task) pairs across realistic states."""
    scen = []
    for procs, tasks in partitioned:
        lcs = [t for t in tasks if t["crit"] == "LC"]
        if not lcs:
            continue
        lcs_sorted = sorted(lcs, key=lambda t: t["u_LO"])
        cand = lcs_sorted[len(lcs_sorted) // 2]       # median-utilization LC task
        for p in procs:                                # each processor as a target
            scen.append((p, cand))
    return scen


def decision_units(partitioned):
    """List of (candidates_sorted, neighbour_processors) per (set, affected proc)."""
    units = []
    for procs, tasks in partitioned:
        for p in procs:
            cands = sorted([t for t in p.tasks if t["crit"] == "LC"],
                           key=lambda t: t["u_LO"])
            if not cands:
                continue
            neighbours = [q for q in procs if q is not p]
            units.append((cands, neighbours))
    return units


# ------------------------------------------------------------------
# Timing
# ------------------------------------------------------------------
def bench_admission(scen):
    seq = [scen[i % len(scen)] for i in range(ADM_INNER)]
    for p, t in seq[:min(len(seq), 2000)]:            # warm-up
        p.try_add_fixed_x(t)
    rounds = []
    for _ in range(ADM_REPEATS):
        gc.collect(); gc.disable()
        t0 = time.perf_counter_ns()
        hits = 0
        for p, t in seq:
            if p.try_add_fixed_x(t):
                hits += 1
        t1 = time.perf_counter_ns()
        gc.enable()
        rounds.append((t1 - t0) / ADM_INNER)
    return min(rounds), statistics.median(rounds)


def _run_decision(cands, neighbours):
    for lc in cands:                                  # first-fit scan (Algorithm 2)
        for nb in neighbours:
            if nb.try_add_fixed_x(lc):
                break


def bench_decision(units):
    per_unit, n_ls = [], []
    for cands, neighbours in units:
        _run_decision(cands, neighbours)              # warm-up
        rs = []
        for _ in range(DEC_REPEATS):
            gc.collect(); gc.disable()
            t0 = time.perf_counter_ns()
            _run_decision(cands, neighbours)
            t1 = time.perf_counter_ns()
            gc.enable()
            rs.append(t1 - t0)
        per_unit.append(statistics.median(rs))
        n_ls.append(len(cands))
    return per_unit, n_ls


def cyc(ns):
    return ns * CPU_GHZ


def main():
    print(f"Host micro-benchmark (pure Python). CPU_GHZ assumed = {CPU_GHZ} for cycle est.\n")
    for m in M_VALUES:
        partitioned = gen_partitioned(m, UT, N_SETS, SEED0 + m * 1000)
        if not partitioned:
            print(f"m={m}: no schedulable sets generated; skipping.")
            continue

        adm_min, adm_med = bench_admission(admission_scenarios(partitioned))

        units = decision_units(partitioned)
        dec, n_ls = bench_decision(units)
        dec_med = statistics.median(dec)
        dec_p95 = sorted(dec)[int(0.95 * (len(dec) - 1))]
        dec_max = max(dec)
        avg_nl = statistics.mean(n_ls)
        max_nl = max(n_ls)
        worst_checks = max_nl * (m - 1)

        print(f"=== m = {m}  (Ut={UT}, P_H={P_H}, {len(partitioned)} sets) ===")
        print(f"  admission check  : median {adm_med:8.1f} ns/call  "
              f"(min {adm_min:.1f})  ~{cyc(adm_med):.0f} host-cycles")
        print(f"  decision latency : median {dec_med:8.1f} ns  "
              f"p95 {dec_p95:.1f}  max {dec_max:.1f}  "
              f"(~{cyc(dec_med):.0f} host-cycles median)")
        print(f"  per mode switch  : avg n_L={avg_nl:.1f}, max n_L={max_nl}, "
              f"neighbours={m-1}; worst-case admission checks = n_L*(m-1) = {worst_checks}")
        print(f"  (decision latency above already includes the first-fit scan and "
              f"loop overhead; the check count bounds the O(n_L*m) work.)\n")


if __name__ == "__main__":
    main()