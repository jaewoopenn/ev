#!/usr/bin/env python3
"""
IMC (Imprecise Mixed-Criticality) Task Set Generator
for Partitioned Multiprocessor Scheduling

Task generation methodology based on:
  - Ramanathan & Easwaran, "Utilization Difference Based Partitioned
    Scheduling of Mixed-Criticality Systems" (2020)
  - Emberson, Stafford & Davis, "Techniques for the Synthesis of
    Multiprocessor Tasksets" (WATERS 2010)

Adapted for the IMC model from:
  - Liu et al., "EDF-VD Scheduling of Mixed-Criticality Systems
    with Degraded Quality Guarantees" (RTSS 2016)

Usage:
  - Eclipse / IDE: Edit the PARAMETERS section below, then Run.
  - Terminal:      python imc_taskset_generator.py
"""

import numpy as np
import json
import os
from dataclasses import dataclass
from typing import List, Optional
import csv
import time


# ╔══════════════════════════════════════════════════════════════╗
# ║                    PARAMETERS (EDIT HERE)                    ║
# ╚══════════════════════════════════════════════════════════════╝

# --- Output ---
OUTPUT_DIR       = "/Users/jaewoo/data/com/data"   # Root output directory
OUTPUT_FORMAT    = "both"                      # "json", "csv", or "both"

# --- Processor configurations ---
PROCESSORS       = [2, 4, 8]                   # List of processor counts

# --- Workload generation ---
N_WORKLOADS      = 1000                        # Number of workloads per UB value
SEED             = 42                          # Random seed for reproducibility

# --- Task criticality ---
P_HC             = 0.5                         # Fraction of HC tasks

# --- Task count ---
# n in [m + 1, N_TASKS_FACTOR * m]  (UDP paper uses factor = 5)
N_TASKS_FACTOR   = 5

# --- Period ---
PERIOD_MIN       = 10                          # Min period (log-uniform)
PERIOD_MAX       = 500                         # Max period (log-uniform)

# --- Individual task utilization bounds ---
U_MIN            = 0.001                       # Min individual task utilization
U_MAX            = 0.99                        # Max individual task utilization

# --- Normalized utilization sweep ---
#   UB_norm = max(U_LC^A + U_HC^L, U_LC^D + U_HC^H) / m
#   Sweep from UB_NORM_LO to UB_NORM_HI with step UB_NORM_STEP
UB_NORM_LO       = 0.45
UB_NORM_HI       = 0.95
UB_NORM_STEP      = 0.1

# --- Utilization component sweep (normalized, per processor) ---
#   U_HC^H_norm in U_HH_RANGE
#   U_HC^L_norm in [U_HL_START, U_HL_START+0.1, ..., U_HC^H_norm]
#   U_LC^A_norm such that UB_norm = max(U_LC^A_norm + U_HC^L_norm, U_LC^D_norm + U_HC^H_norm)
U_HH_RANGE       = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
U_HL_START        = 0.05                       # Starting value for U_HC^L_norm
U_HL_STEP         = 0.1                        # Step for U_HC^L_norm

# --- LC degradation ratio ---
#   For each LC task: u_D = lambda * u_A, lambda ~ uniform[LAMBDA_MIN, LAMBDA_MAX]
LAMBDA_MIN        = 0.0
LAMBDA_MAX        = 1.0

# ╔══════════════════════════════════════════════════════════════╗
# ║                  END OF PARAMETERS                           ║
# ╚══════════════════════════════════════════════════════════════╝


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Task:
    """A single IMC task."""
    task_id: int
    criticality: str        # "HC" or "LC"
    period: int             # T_i
    deadline: int           # D_i = T_i (implicit deadline)
    C_LO: int               # WCET in LO mode
    C_HI: int               # WCET in HI mode
    u_LO: float             # utilization in LO mode
    u_HI: float             # utilization in HI mode


@dataclass
class TaskSet:
    """A complete IMC task set (workload)."""
    taskset_id: int
    num_processors: int
    UB: float               # max(U_LO, U_HI) = target total utilization
    tasks: List[Task]
    # Computed utilization metrics
    U_LC_A: float = 0.0
    U_HC_L: float = 0.0
    U_LC_D: float = 0.0
    U_HC_H: float = 0.0
    U_LO: float = 0.0
    U_HI: float = 0.0
    actual_UB: float = 0.0

    def compute_utilizations(self):
        self.U_LC_A = sum(t.u_LO for t in self.tasks if t.criticality == "LC")
        self.U_HC_L = sum(t.u_LO for t in self.tasks if t.criticality == "HC")
        self.U_LC_D = sum(t.u_HI for t in self.tasks if t.criticality == "LC")
        self.U_HC_H = sum(t.u_HI for t in self.tasks if t.criticality == "HC")
        self.U_LO = self.U_LC_A + self.U_HC_L
        self.U_HI = self.U_LC_D + self.U_HC_H
        self.actual_UB = max(self.U_LO, self.U_HI)


# ============================================================
# Utilization Generation (Stafford's randfixedsum)
# ============================================================

def randfixedsum(n: int, u_total: float, rng: np.random.Generator,
                 u_min: float = 0.001, u_max: float = 0.99) -> Optional[np.ndarray]:
    """
    Generate n random values in [u_min, u_max] that sum to u_total.

    Uses the UUniFast-Discard approach with shifted range.
    Stafford's exact randfixedsum is complex; this is a practical
    equivalent that produces well-distributed utilizations.

    Returns None if infeasible (u_total not achievable with n values in range).
    """
    # Feasibility check
    if u_total < n * u_min or u_total > n * u_max:
        return None

    # Shift to [0, u_max - u_min], generate, then shift back
    shifted_total = u_total - n * u_min
    shifted_max = u_max - u_min

    if shifted_total < 0 or shifted_total > n * shifted_max:
        return None

    # UUniFast-Discard on shifted range
    for _ in range(200):
        utils = _uunifast(n, shifted_total, rng)
        if np.all(utils >= 0) and np.all(utils <= shifted_max):
            return utils + u_min

    # Fallback: Dirichlet-based
    for _ in range(200):
        utils = rng.dirichlet(np.ones(n)) * shifted_total
        if np.all(utils >= 0) and np.all(utils <= shifted_max):
            return utils + u_min

    # Last resort
    utils = np.full(n, shifted_total / n)
    return utils + u_min


def _uunifast(n: int, u_total: float, rng: np.random.Generator) -> np.ndarray:
    """Standard UUniFast algorithm."""
    utils = np.zeros(n)
    sum_u = u_total
    for i in range(n - 1):
        exp = 1.0 / (n - 1 - i)
        next_sum_u = sum_u * (rng.random() ** exp)
        utils[i] = sum_u - next_sum_u
        sum_u = next_sum_u
    utils[n - 1] = sum_u
    return utils


def log_uniform_int(lo: int, hi: int, rng: np.random.Generator) -> int:
    """Draw an integer from a log-uniform distribution over [lo, hi]."""
    log_lo = np.log(lo)
    log_hi = np.log(hi)
    return int(round(np.exp(rng.uniform(log_lo, log_hi))))


# ============================================================
# Task Set Generator
# ============================================================

class IMCTaskSetGenerator:
    """
    Generates IMC task sets following the UDP paper methodology.

    For a given UB (= max(U_LO, U_HI)):
      1. Enumerate valid (U_HC^H_norm, U_HC^L_norm) combinations
      2. For each combination, compute U_LC^A_norm and U_LC^D_norm
      3. Generate individual task utilizations and periods
      4. Collect task sets until N_WORKLOADS reached
    """

    def __init__(
        self,
        num_processors: int,
        p_hc: float = 0.5,
        n_tasks_factor: int = 5,
        period_range: tuple = (10, 500),
        u_min: float = 0.001,
        u_max: float = 0.99,
        lambda_range: tuple = (0.0, 1.0),
        u_hh_range: list = None,
        u_hl_start: float = 0.05,
        u_hl_step: float = 0.1,
        seed: Optional[int] = None
    ):
        self.m = num_processors
        self.p_hc = p_hc
        self.n_tasks_factor = n_tasks_factor
        self.period_range = period_range
        self.u_min = u_min
        self.u_max = u_max
        self.lambda_range = lambda_range
        self.u_hh_range = u_hh_range or U_HH_RANGE
        self.u_hl_start = u_hl_start
        self.u_hl_step = u_hl_step
        self.rng = np.random.default_rng(seed)

    def _generate_one_taskset(
        self,
        taskset_id: int,
        UB: float,
        U_HC_H_total: float,
        U_HC_L_total: float,
        U_LC_A_total: float
    ) -> Optional[TaskSet]:
        """
        Generate a single task set with specified total utilizations.

        Args:
            taskset_id:     unique id
            UB:             target max(U_LO, U_HI)
            U_HC_H_total:   total u_HI for HC tasks (system-level)
            U_HC_L_total:   total u_LO for HC tasks (system-level)
            U_LC_A_total:   total u_LO for LC tasks (system-level)

        Returns:
            TaskSet or None if generation failed
        """
        m = self.m

        # Determine task count
        n_min = m + 1
        n_max = self.n_tasks_factor * m
        n = self.rng.integers(n_min, n_max + 1)

        # Determine number of HC and LC tasks
        n_hc = max(1, round(n * self.p_hc))
        n_lc = n - n_hc
        if n_lc < 1:
            n_lc = 1
            n_hc = n - 1

        # --- Generate HC task utilizations ---
        # u_LO for HC tasks: sum = U_HC_L_total, each in [u_min, min(u_max, U_HC_L_total)]
        hc_u_lo = randfixedsum(n_hc, U_HC_L_total, self.rng, self.u_min, self.u_max)
        if hc_u_lo is None:
            return None

        # u_HI for HC tasks: sum = U_HC_H_total, each in [corresponding u_LO, u_max]
        # Strategy: distribute U_HC_H_total proportionally to u_LO, then adjust
        if U_HC_L_total > 0:
            ratios = hc_u_lo / hc_u_lo.sum()
        else:
            ratios = np.ones(n_hc) / n_hc
        hc_u_hi = ratios * U_HC_H_total

        # Ensure u_HI >= u_LO and u_HI <= u_max for each HC task
        for i in range(n_hc):
            hc_u_hi[i] = max(hc_u_hi[i], hc_u_lo[i])
            hc_u_hi[i] = min(hc_u_hi[i], self.u_max)

        # Rescale to hit target sum
        current_sum = hc_u_hi.sum()
        if current_sum > 0 and abs(current_sum - U_HC_H_total) > 1e-9:
            scale = U_HC_H_total / current_sum
            hc_u_hi *= scale
            # Re-enforce bounds
            for i in range(n_hc):
                hc_u_hi[i] = max(hc_u_hi[i], hc_u_lo[i])
                hc_u_hi[i] = min(hc_u_hi[i], self.u_max)

        # --- Generate LC task utilizations ---
        # u_LO (= u_A) for LC tasks: sum = U_LC_A_total
        lc_u_lo = randfixedsum(n_lc, U_LC_A_total, self.rng, self.u_min, self.u_max)
        if lc_u_lo is None:
            return None

        # u_HI (= u_D) for LC tasks: u_D = lambda * u_A, lambda ~ uniform
        lc_lambdas = self.rng.uniform(self.lambda_range[0], self.lambda_range[1], n_lc)
        lc_u_hi = lc_u_lo * lc_lambdas

        # --- Generate periods and compute C values ---
        tasks = []
        task_id = 0

        for i in range(n_hc):
            T = log_uniform_int(self.period_range[0], self.period_range[1], self.rng)
            C_LO = max(1, int(np.ceil(hc_u_lo[i] * T)))
            C_HI = max(C_LO, int(np.ceil(hc_u_hi[i] * T)))
            u_lo_actual = C_LO / T
            u_hi_actual = C_HI / T

            tasks.append(Task(
                task_id=task_id, criticality="HC",
                period=T, deadline=T,
                C_LO=C_LO, C_HI=C_HI,
                u_LO=u_lo_actual, u_HI=u_hi_actual
            ))
            task_id += 1

        for i in range(n_lc):
            T = log_uniform_int(self.period_range[0], self.period_range[1], self.rng)
            C_LO = max(1, int(np.ceil(lc_u_lo[i] * T)))
            C_HI = max(0, int(np.round(lc_u_hi[i] * T)))
            u_lo_actual = C_LO / T
            u_hi_actual = C_HI / T

            tasks.append(Task(
                task_id=task_id, criticality="LC",
                period=T, deadline=T,
                C_LO=C_LO, C_HI=C_HI,
                u_LO=u_lo_actual, u_HI=u_hi_actual
            ))
            task_id += 1

        ts = TaskSet(
            taskset_id=taskset_id,
            num_processors=self.m,
            UB=UB,
            tasks=tasks
        )
        ts.compute_utilizations()

        # Post-scaling: adjust so that max(U_LO, U_HI) = UB
        if ts.actual_UB > 1e-9:
            scale = UB / ts.actual_UB
            if abs(scale - 1.0) > 1e-6:
                for t in tasks:
                    T = t.period
                    t.C_LO = max(1, int(round(t.u_LO * scale * T)))
                    t.C_HI = max(0 if t.criticality == "LC" else t.C_LO,
                                 int(round(t.u_HI * scale * T)))
                    t.u_LO = t.C_LO / T
                    t.u_HI = t.C_HI / T
                ts.compute_utilizations()
        return ts

    def _enumerate_util_configs(self, UB_norm: float) -> list:
        """
        Enumerate valid (U_HC^H_norm, U_HC^L_norm, U_LC^A_norm) configurations
        for a given normalized UB.

        Following UDP paper:
          U_HC^H_norm in u_hh_range, filtered by <= UB_norm
          U_HC^L_norm in [u_hl_start, u_hl_start+step, ..., U_HC^H_norm]
          U_LC^A_norm chosen so that UB_norm = max(U_LC^A + U_HC^L, U_LC^D + U_HC^H)/m

        For IMC model, U_LC^D depends on lambda (random), so we set:
          U_LC^A_norm = UB_norm - U_HC^L_norm
          (This ensures U_LO = UB * m when U_LO is the dominant term)
          Then actual UB depends on U_HI = U_LC^D + U_HC^H

        Returns list of (U_HC_H_total, U_HC_L_total, U_LC_A_total) tuples.
        """
        m = self.m
        configs = []

        for u_hh_norm in self.u_hh_range:
            if u_hh_norm > UB_norm:
                continue

            u_hl_norm = self.u_hl_start
            while u_hl_norm <= u_hh_norm + 1e-9:
                # U_LC^A_norm: fill remaining capacity for LO mode
                u_la_norm = UB_norm - u_hl_norm
                if u_la_norm < 0.01:
                    u_hl_norm += self.u_hl_step
                    continue

                # Feasibility: ensure individual tasks can hold these utils
                U_HC_H_total = u_hh_norm * m
                U_HC_L_total = u_hl_norm * m
                U_LC_A_total = u_la_norm * m

                configs.append((U_HC_H_total, U_HC_L_total, U_LC_A_total))
                u_hl_norm += self.u_hl_step

        return configs

    def generate_for_UB(self, UB_norm: float, n_workloads: int,
                        start_id: int = 0) -> List[TaskSet]:
        """
        Generate workloads for a given normalized UB.

        Cycles through utilization configurations and generates task sets
        until n_workloads are collected.
        """
        m = self.m
        UB = UB_norm * m
        configs = self._enumerate_util_configs(UB_norm)

        if not configs:
            print(f"  WARNING: No valid configs for UB_norm={UB_norm:.2f}")
            return []

        workloads = []
        config_idx = 0
        attempts = 0
        max_attempts = n_workloads * 10

        while len(workloads) < n_workloads and attempts < max_attempts:
            cfg = configs[config_idx % len(configs)]
            config_idx += 1
            attempts += 1

            U_HC_H_total, U_HC_L_total, U_LC_A_total = cfg

            ts = self._generate_one_taskset(
                start_id + len(workloads),
                UB,
                U_HC_H_total,
                U_HC_L_total,
                U_LC_A_total
            )

            if ts is not None:
                workloads.append(ts)

        return workloads

    def generate_all(self, n_workloads: int,
                     ub_lo: float, ub_hi: float, ub_step: float) -> dict:
        """
        Generate workloads for all UB values.

        Args:
            n_workloads: workloads per UB
            ub_lo:       min normalized UB
            ub_hi:       max normalized UB
            ub_step:     step for normalized UB

        Returns:
            dict mapping UB (total, not normalized) -> list of TaskSet
        """
        m = self.m
        ub_norms = np.arange(ub_lo, ub_hi + ub_step * 0.01, ub_step)
        ub_norms = np.round(ub_norms, 2)

        n_min = m + 1
        n_max = self.n_tasks_factor * m

        print(f"Generating workloads for m={m} processors")
        print(f"UB_norm values: {ub_norms.tolist()}")
        print(f"Workloads per UB: {n_workloads}")
        print(f"P_HC: {self.p_hc}")
        print(f"Task count: [{n_min}, {n_max}]")
        print(f"Period: log-uniform [{self.period_range[0]}, {self.period_range[1]}]")
        print(f"Lambda range: {self.lambda_range}")
        print(f"u range: [{self.u_min}, {self.u_max}]")
        print("-" * 60)

        all_workloads = {}
        total_id = 0

        for ub_norm in ub_norms:
            ub_norm = float(ub_norm)
            UB = ub_norm * m
            t_start = time.time()

            workloads = self.generate_for_UB(ub_norm, n_workloads, total_id)
            t_elapsed = time.time() - t_start

            if workloads:
                actual_ubs = [ts.actual_UB for ts in workloads]
                u_los = [ts.U_LO for ts in workloads]
                u_his = [ts.U_HI for ts in workloads]
                n_tasks = [len(ts.tasks) for ts in workloads]
                util_diffs = [ts.U_HC_H - ts.U_HC_L for ts in workloads]

                print(
                    f"  UB={UB:.2f} (norm={ub_norm:.2f}) | "
                    f"actual: {np.mean(actual_ubs):.3f}±{np.std(actual_ubs):.3f} | "
                    f"U_LO: {np.mean(u_los):.3f} | "
                    f"U_HI: {np.mean(u_his):.3f} | "
                    f"diff: {np.mean(util_diffs):.3f}±{np.std(util_diffs):.3f} | "
                    f"tasks: {np.mean(n_tasks):.1f} | "
                    f"n={len(workloads)} | "
                    f"time: {t_elapsed:.2f}s"
                )
            else:
                print(f"  UB={UB:.2f} (norm={ub_norm:.2f}) | NO WORKLOADS GENERATED")

            all_workloads[round(UB, 2)] = workloads
            total_id += len(workloads)

        return all_workloads


# ============================================================
# Export Functions
# ============================================================

def export_to_json(all_workloads: dict, output_dir: str, prefix: str = "imc"):
    os.makedirs(output_dir, exist_ok=True)

    for UB, workloads in all_workloads.items():
        if not workloads:
            continue
        m = workloads[0].num_processors
        filename = f"{prefix}_m{m}_cap{UB:.2f}.json"
        filepath = os.path.join(output_dir, filename)

        data = {
            "num_processors": m,
            "util_cap": UB,
            "num_workloads": len(workloads),
            "workloads": []
        }

        for ts in workloads:
            ts_data = {
                "taskset_id": ts.taskset_id,
                "U_LC_A": round(ts.U_LC_A, 6),
                "U_HC_L": round(ts.U_HC_L, 6),
                "U_LC_D": round(ts.U_LC_D, 6),
                "U_HC_H": round(ts.U_HC_H, 6),
                "U_LO": round(ts.U_LO, 6),
                "U_HI": round(ts.U_HI, 6),
                "actual_util_cap": round(ts.actual_UB, 6),
                "tasks": [
                    {
                        "id": int(t.task_id),
                        "crit": t.criticality,
                        "T": int(t.period),
                        "D": int(t.deadline),
                        "C_LO": int(t.C_LO),
                        "C_HI": int(t.C_HI),
                        "u_LO": round(t.u_LO, 6),
                        "u_HI": round(t.u_HI, 6),
                    }
                    for t in ts.tasks
                ]
            }
            data["workloads"].append(ts_data)

        with open(filepath, 'w') as f:
            json.dump(data, f)

        print(f"  Saved: {filepath} ({len(workloads)} workloads)")


def export_summary_csv(all_workloads: dict, output_dir: str, prefix: str = "imc"):
    os.makedirs(output_dir, exist_ok=True)

    m = None
    for wl in all_workloads.values():
        if wl:
            m = wl[0].num_processors
            break
    if m is None:
        return

    filename = f"{prefix}_m{m}_summary.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "util_cap", "num_workloads", "num_processors",
            "avg_n_tasks", "avg_U_LO", "std_U_LO",
            "avg_U_HI", "std_U_HI",
            "avg_actual_cap", "std_actual_cap",
            "avg_U_LC_A", "avg_U_HC_L", "avg_U_LC_D", "avg_U_HC_H",
            "avg_util_diff", "std_util_diff",
            "avg_n_HC", "avg_n_LC"
        ])

        for UB in sorted(all_workloads.keys()):
            workloads = all_workloads[UB]
            if not workloads:
                continue

            n_tasks = [len(ts.tasks) for ts in workloads]
            u_los = [ts.U_LO for ts in workloads]
            u_his = [ts.U_HI for ts in workloads]
            actual_caps = [ts.actual_UB for ts in workloads]
            u_lc_a = [ts.U_LC_A for ts in workloads]
            u_hc_l = [ts.U_HC_L for ts in workloads]
            u_lc_d = [ts.U_LC_D for ts in workloads]
            u_hc_h = [ts.U_HC_H for ts in workloads]
            util_diffs = [ts.U_HC_H - ts.U_HC_L for ts in workloads]
            n_hc = [sum(1 for t in ts.tasks if t.criticality == "HC")
                    for ts in workloads]
            n_lc = [sum(1 for t in ts.tasks if t.criticality == "LC")
                    for ts in workloads]

            writer.writerow([
                f"{UB:.2f}", len(workloads), m,
                f"{np.mean(n_tasks):.2f}",
                f"{np.mean(u_los):.4f}", f"{np.std(u_los):.4f}",
                f"{np.mean(u_his):.4f}", f"{np.std(u_his):.4f}",
                f"{np.mean(actual_caps):.4f}", f"{np.std(actual_caps):.4f}",
                f"{np.mean(u_lc_a):.4f}", f"{np.mean(u_hc_l):.4f}",
                f"{np.mean(u_lc_d):.4f}", f"{np.mean(u_hc_h):.4f}",
                f"{np.mean(util_diffs):.4f}", f"{np.std(util_diffs):.4f}",
                f"{np.mean(n_hc):.2f}", f"{np.mean(n_lc):.2f}"
            ])

    print(f"  Summary saved: {filepath}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("IMC Task Set Generator (UDP-style)")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processors:       {PROCESSORS}")
    print(f"Workloads/UB:     {N_WORKLOADS}")
    print(f"Seed:             {SEED}")
    print(f"P_HC:             {P_HC}")
    print(f"Task count:       [m+1, {N_TASKS_FACTOR}*m]")
    print(f"Period:           log-uniform [{PERIOD_MIN}, {PERIOD_MAX}]")
    print(f"u range:          [{U_MIN}, {U_MAX}]")
    print(f"Lambda range:     [{LAMBDA_MIN}, {LAMBDA_MAX}]")
    print(f"UB_norm range:    [{UB_NORM_LO}, {UB_NORM_HI}], step={UB_NORM_STEP}")

    for m in PROCESSORS:
        print(f"\n{'=' * 60}")
        print(f"Processing m = {m} processors")
        print(f"{'=' * 60}")

        generator = IMCTaskSetGenerator(
            num_processors=m,
            p_hc=P_HC,
            n_tasks_factor=N_TASKS_FACTOR,
            period_range=(PERIOD_MIN, PERIOD_MAX),
            u_min=U_MIN,
            u_max=U_MAX,
            lambda_range=(LAMBDA_MIN, LAMBDA_MAX),
            u_hh_range=U_HH_RANGE,
            u_hl_start=U_HL_START,
            u_hl_step=U_HL_STEP,
            seed=SEED + m
        )

        all_workloads = generator.generate_all(
            n_workloads=N_WORKLOADS,
            ub_lo=UB_NORM_LO,
            ub_hi=UB_NORM_HI,
            ub_step=UB_NORM_STEP
        )

        print(f"\nExporting results...")
        if OUTPUT_FORMAT in ("json", "both"):
            export_to_json(all_workloads, OUTPUT_DIR)
        if OUTPUT_FORMAT in ("csv", "both"):
            export_summary_csv(all_workloads, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print(f"All done! Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()