#!/usr/bin/env python3
"""
IMC (Imprecise Mixed-Criticality) Task Set Generator
for Partitioned Multiprocessor Scheduling

Based on the task model from:
  "EDF-VD Scheduling of Mixed-Criticality Systems
   with Degraded Quality Guarantees" (Liu et al., RTSS 2016)

Extended for multiprocessor (partitioned) with:
  - UUniFast utilization distribution
  - Configurable parameters

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
N_WORKLOADS      = 1000                        # Number of workloads per util_cap
SEED             = 42                          # Random seed for reproducibility

# --- Task criticality ---
P_CRITICALITY    = 0.5                         # Probability a task is HC (0.0 ~ 1.0)

# --- HC task: C_HI / C_LO ratio ---
R_MIN            = 1.5                         # Min R for HC tasks
R_MAX            = 2.5                         # Max R for HC tasks

# --- LC task: C_D / C_A ratio (= lambda) ---
LAMBDA_MIN       = 0.0                         # Min lambda for LC tasks
LAMBDA_MAX       = 1.0                         # Max lambda for LC tasks

# --- Period ---
PERIOD_MIN       = 100                         # Min task period
PERIOD_MAX       = 1000                        # Max task period

# --- Number of tasks per task set (per processor) ---
# N_TASKS_PER_PROC_MIN = 5                       # n_tasks = uniform[this*m, that*m]
# N_TASKS_PER_PROC_MAX = 10                      # n_tasks = uniform[this*m, that*m]
N_TASKS_PER_PROC_MIN = 3                       # n_tasks = uniform[this*m, that*m]
N_TASKS_PER_PROC_MAX = 6                      # n_tasks = uniform[this*m, that*m]

# --- Utilization cap range ---
#   util_cap ranges from UTIL_CAP_LO_RATIO * m  to  UTIL_CAP_HI_RATIO * m
#   with step = 0.1 * m
UTIL_CAP_LO_RATIO = 0.5                       # Lower bound ratio (× m)
UTIL_CAP_HI_RATIO = 1.0                       # Upper bound ratio (× m)
# Step is always 0.1 * m

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
    u_LO: float             # utilization in LO mode = C_LO / T_i
    u_HI: float             # utilization in HI mode = C_HI / T_i
    R: Optional[float] = None    # C_HI/C_LO ratio (HC tasks only)
    lam: Optional[float] = None  # C_HI/C_LO ratio (LC tasks only, = lambda)


@dataclass
class TaskSet:
    """A complete IMC task set (workload)."""
    taskset_id: int
    num_processors: int
    util_cap: float
    pCriticality: float
    tasks: List[Task]
    # Computed utilization metrics
    U_LC_A: float = 0.0     # sum of u_LO for LC tasks (= U^LO_LO)
    U_HC_L: float = 0.0     # sum of u_LO for HC tasks (= U^LO_HI)
    U_LC_D: float = 0.0     # sum of u_HI for LC tasks (= U^HI_LO)
    U_HC_H: float = 0.0     # sum of u_HI for HC tasks (= U^HI_HI)
    U_LO: float = 0.0       # U_LC_A + U_HC_L
    U_HI: float = 0.0       # U_LC_D + U_HC_H
    actual_util_cap: float = 0.0  # max(U_LO, U_HI)

    def compute_utilizations(self):
        """Compute aggregate utilization metrics."""
        self.U_LC_A = sum(t.u_LO for t in self.tasks if t.criticality == "LC")
        self.U_HC_L = sum(t.u_LO for t in self.tasks if t.criticality == "HC")
        self.U_LC_D = sum(t.u_HI for t in self.tasks if t.criticality == "LC")
        self.U_HC_H = sum(t.u_HI for t in self.tasks if t.criticality == "HC")
        self.U_LO = self.U_LC_A + self.U_HC_L
        self.U_HI = self.U_LC_D + self.U_HC_H
        self.actual_util_cap = max(self.U_LO, self.U_HI)


# ============================================================
# UUniFast Algorithm
# ============================================================

def uunifast(n: int, u_total: float, rng: np.random.Generator) -> np.ndarray:
    """
    UUniFast algorithm (Bini & Buttazzo, 2005).
    Generates n utilization values that sum to u_total,
    uniformly distributed over the valid space.
    """
    utils = np.zeros(n)
    sum_u = u_total
    for i in range(n - 1):
        exp = 1.0 / (n - 1 - i)
        next_sum_u = sum_u * (rng.random() ** exp)
        utils[i] = sum_u - next_sum_u
        sum_u = next_sum_u
    utils[n - 1] = sum_u
    return utils


def uunifast_discard(n: int, u_total: float, rng: np.random.Generator,
                     max_attempts: int = 100) -> np.ndarray:
    """
    UUniFast-Discard: rejects task sets where any u_i > 1.
    Falls back to Dirichlet-based generation if discard rate is too high.
    """
    if u_total <= n:
        for _ in range(max_attempts):
            utils = uunifast(n, u_total, rng)
            if np.all(utils <= 1.0) and np.all(utils > 0):
                return utils

    # Fallback: Dirichlet-based generation capped at 1.0
    for _ in range(max_attempts):
        utils = rng.dirichlet(np.ones(n)) * u_total
        if np.all(utils <= 1.0) and np.all(utils > 0):
            return utils

    # Last resort: uniform distribution with slight noise
    base = u_total / n
    if base > 1.0:
        utils = np.ones(n) * (u_total / n)
        return np.clip(utils, 0.001, 1.0)
    noise = rng.uniform(-base * 0.3, base * 0.3, n)
    utils = base + noise
    utils = np.clip(utils, 0.001, 1.0)
    utils = utils * (u_total / utils.sum())
    utils = np.clip(utils, 0.001, 1.0)
    return utils


# ============================================================
# Task Set Generator
# ============================================================

class IMCTaskSetGenerator:
    """
    Generator for IMC task sets.

    Parameters:
        num_processors:         number of processors (e.g., 2, 4, 8)
        pCriticality:           probability that a task is HC
        R_range:                (min, max) for HC task's C_HI/C_LO ratio
        lambda_range:           (min, max) for LC task's C_HI/C_LO ratio
        period_range:           (min, max) for task period
        n_tasks_per_proc_range: (min, max) tasks per processor
        seed:                   random seed for reproducibility
    """

    def __init__(
        self,
        num_processors: int = 4,
        pCriticality: float = 0.5,
        R_range: tuple = (1.5, 2.5),
        lambda_range: tuple = (0.0, 1.0),
        period_range: tuple = (100, 1000),
        n_tasks_per_proc_range: tuple = (5, 10),
        seed: Optional[int] = None
    ):
        self.num_processors = num_processors
        self.pCriticality = pCriticality
        self.R_range = R_range
        self.lambda_range = lambda_range
        self.period_range = period_range
        self.n_tasks_per_proc_range = n_tasks_per_proc_range
        self.rng = np.random.default_rng(seed)

    def generate_single_taskset(
        self, taskset_id: int, util_cap: float
    ) -> TaskSet:
        """
        Generate a single IMC task set such that
        max(U_LO, U_HI) ≈ util_cap.

        Approach:
          1. Decide n tasks, assign criticality, generate R/lambda/period.
          2. UUniFast to distribute base utilizations.
          3. Compute preliminary U_LO and U_HI.
          4. Scale all utilizations so that max(U_LO, U_HI) = util_cap.
          5. Recompute C_LO, C_HI from scaled utilizations.

        Args:
            taskset_id: unique identifier
            util_cap:   target max(U_LO, U_HI)

        Returns:
            TaskSet object
        """
        # Step 1: Determine number of tasks (proportional to processor count)
        n_min = self.n_tasks_per_proc_range[0] * self.num_processors
        n_max = self.n_tasks_per_proc_range[1] * self.num_processors
        n = self.rng.integers(n_min, n_max + 1)

        # Step 2: Assign criticality, R/lambda, period for each task
        task_params = []
        for i in range(n):
            T_i = self.rng.integers(
                self.period_range[0], self.period_range[1] + 1
            )
            is_HC = self.rng.random() < self.pCriticality

            if is_HC:
                R = self.rng.uniform(self.R_range[0], self.R_range[1])
                task_params.append({
                    "id": i, "crit": "HC", "T": int(T_i), "R": R, "lam": None
                })
            else:
                lam = self.rng.uniform(
                    self.lambda_range[0], self.lambda_range[1]
                )
                task_params.append({
                    "id": i, "crit": "LC", "T": int(T_i), "R": None, "lam": lam
                })

        # Step 3: UUniFast to distribute base utilizations
        # Use util_cap as initial target (will be scaled later)
        base_utils = uunifast_discard(n, util_cap, self.rng)

        # Step 3.5: Cap individual HC task utilizations so that u_HI <= 1.0
        # For HC task: u_HI = R * u_base, so we need u_base <= 1/R
        # Redistribute any excess to other tasks
        needs_redistribution = True
        max_iters = 10
        for _ in range(max_iters):
            excess = 0.0
            uncapped_indices = []
            for i in range(n):
                if task_params[i]["crit"] == "HC":
                    R = task_params[i]["R"]
                    cap_val = 1.0 / R  # max u_base so that R*u_base <= 1.0
                    if base_utils[i] > cap_val:
                        excess += base_utils[i] - cap_val
                        base_utils[i] = cap_val
                    else:
                        uncapped_indices.append(i)
                else:
                    # LC task: u_LO = u_base, need u_base <= 1.0 (already ensured)
                    if base_utils[i] <= 1.0:
                        uncapped_indices.append(i)

            if excess < 1e-9 or not uncapped_indices:
                break

            # Redistribute excess proportionally among uncapped tasks
            uncapped_sum = sum(base_utils[j] for j in uncapped_indices)
            if uncapped_sum > 0:
                for j in uncapped_indices:
                    base_utils[j] += excess * (base_utils[j] / uncapped_sum)
            else:
                for j in uncapped_indices:
                    base_utils[j] += excess / len(uncapped_indices)

        # Step 4: Compute preliminary utilizations and scaling factor
        # For HC: u_LO = base, u_HI = R * base
        # For LC: u_LO = base, u_HI = lambda * base
        prelim_U_LO = 0.0  # sum of u_LO (= sum of base_utils = util_cap)
        prelim_U_HI = 0.0  # sum of u_HI
        for i in range(n):
            u_base = base_utils[i]
            if task_params[i]["crit"] == "HC":
                prelim_U_LO += u_base
                prelim_U_HI += task_params[i]["R"] * u_base
            else:
                prelim_U_LO += u_base
                prelim_U_HI += task_params[i]["lam"] * u_base

        prelim_max = max(prelim_U_LO, prelim_U_HI)

        # Scale factor: we want max(U_LO, U_HI) = util_cap
        if prelim_max > 0:
            scale = util_cap / prelim_max
        else:
            scale = 1.0

        # Step 5: Apply scaling and compute final C_LO, C_HI
        tasks = []
        for i in range(n):
            u_base_scaled = base_utils[i] * scale
            T_i = task_params[i]["T"]
            D_i = T_i  # implicit deadline

            if task_params[i]["crit"] == "HC":
                R = task_params[i]["R"]
                C_LO = max(1, round(u_base_scaled * T_i))
                C_HI = max(C_LO, round(R * u_base_scaled * T_i))

                u_LO = C_LO / T_i
                u_HI = C_HI / T_i

                task = Task(
                    task_id=i, criticality="HC",
                    period=T_i, deadline=D_i,
                    C_LO=C_LO, C_HI=C_HI,
                    u_LO=u_LO, u_HI=u_HI,
                    R=R, lam=None
                )
            else:
                lam = task_params[i]["lam"]
                C_LO = max(1, round(u_base_scaled * T_i))
                C_HI = max(0, round(lam * u_base_scaled * T_i))

                u_LO = C_LO / T_i
                u_HI = C_HI / T_i

                task = Task(
                    task_id=i, criticality="LC",
                    period=T_i, deadline=D_i,
                    C_LO=C_LO, C_HI=C_HI,
                    u_LO=u_LO, u_HI=u_HI,
                    R=None, lam=lam
                )

            tasks.append(task)

        # Step 6: Create TaskSet and compute utilizations
        ts = TaskSet(
            taskset_id=taskset_id,
            num_processors=self.num_processors,
            util_cap=util_cap,
            pCriticality=self.pCriticality,
            tasks=tasks
        )
        ts.compute_utilizations()

        return ts

    def generate_workloads(
        self, util_cap: float, n_workloads: int = 1000, start_id: int = 0
    ) -> List[TaskSet]:
        """Generate multiple workloads for a given utilization cap."""
        workloads = []
        for i in range(n_workloads):
            ts = self.generate_single_taskset(start_id + i, util_cap)
            workloads.append(ts)
        return workloads

    def generate_all(
        self, n_workloads: int = 1000,
        util_cap_lo_ratio: float = 0.5,
        util_cap_hi_ratio: float = 1.0
    ) -> dict:
        """
        Generate workloads for all utilization caps.

        util_cap: from util_cap_lo_ratio * m  to  util_cap_hi_ratio * m
        step:     0.1 * m

        Example for m=4:
          caps = [2.0, 2.4, 2.8, 3.2, 3.6, 4.0]   (step = 0.4)

        Args:
            n_workloads:       number of workloads per utilization cap
            util_cap_lo_ratio: lower ratio (default 0.5)
            util_cap_hi_ratio: upper ratio (default 1.0)

        Returns:
            dict mapping util_cap -> list of TaskSet
        """
        m = self.num_processors
        step = 0.1 * m
        cap_lo = util_cap_lo_ratio * m
        cap_hi = util_cap_hi_ratio * m

        # Generate util_cap values
        util_caps = np.arange(cap_lo, cap_hi + step * 0.01, step)
        util_caps = np.round(util_caps, 2)

        all_workloads = {}
        total_id = 0

        n_min = self.n_tasks_per_proc_range[0] * m
        n_max = self.n_tasks_per_proc_range[1] * m

        print(f"Generating workloads for m={m} processors")
        print(f"Utilization caps: {util_caps.tolist()}")
        print(f"Step: {step:.1f} (= 0.1 × {m})")
        print(f"Workloads per cap: {n_workloads}")
        print(f"pCriticality: {self.pCriticality}")
        print(f"R range: {self.R_range}, lambda range: {self.lambda_range}")
        print(f"Period range: {self.period_range}")
        print(f"n_tasks range: [{n_min}, {n_max}] "
              f"(base {self.n_tasks_per_proc_range} × m={m})")
        print("-" * 60)

        for cap in util_caps:
            cap = float(cap)
            t_start = time.time()
            workloads = self.generate_workloads(cap, n_workloads, total_id)
            t_elapsed = time.time() - t_start

            # Compute statistics
            actual_caps = [ts.actual_util_cap for ts in workloads]
            u_los = [ts.U_LO for ts in workloads]
            u_his = [ts.U_HI for ts in workloads]
            n_tasks_list = [len(ts.tasks) for ts in workloads]

            print(
                f"  util_cap={cap:.2f} | "
                f"actual_cap: mean={np.mean(actual_caps):.3f}, "
                f"std={np.std(actual_caps):.3f} | "
                f"U_LO: {np.mean(u_los):.3f} | "
                f"U_HI: {np.mean(u_his):.3f} | "
                f"avg_tasks: {np.mean(n_tasks_list):.1f} | "
                f"time: {t_elapsed:.2f}s"
            )

            all_workloads[cap] = workloads
            total_id += n_workloads

        return all_workloads


# ============================================================
# Export Functions
# ============================================================

def export_to_json(all_workloads: dict, output_dir: str, prefix: str = "imc"):
    """Export workloads to JSON files, one per utilization cap."""
    os.makedirs(output_dir, exist_ok=True)

    for cap, workloads in all_workloads.items():
        m = workloads[0].num_processors
        filename = f"{prefix}_m{m}_cap{cap:.2f}.json"
        filepath = os.path.join(output_dir, filename)

        data = {
            "num_processors": m,
            "util_cap": cap,
            "pCriticality": workloads[0].pCriticality,
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
                "actual_util_cap": round(ts.actual_util_cap, 6),
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
                        "R": round(t.R, 4) if t.R is not None else None,
                        "lam": round(t.lam, 4) if t.lam is not None else None
                    }
                    for t in ts.tasks
                ]
            }
            data["workloads"].append(ts_data)

        with open(filepath, 'w') as f:
            json.dump(data, f)

        print(f"  Saved: {filepath} ({len(workloads)} workloads)")


def export_summary_csv(all_workloads: dict, output_dir: str, prefix: str = "imc"):
    """Export a summary CSV with aggregate statistics per utilization cap."""
    os.makedirs(output_dir, exist_ok=True)

    m = list(all_workloads.values())[0][0].num_processors
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
            "avg_n_HC", "avg_n_LC"
        ])

        for cap in sorted(all_workloads.keys()):
            workloads = all_workloads[cap]
            n = len(workloads)

            n_tasks = [len(ts.tasks) for ts in workloads]
            u_los = [ts.U_LO for ts in workloads]
            u_his = [ts.U_HI for ts in workloads]
            actual_caps = [ts.actual_util_cap for ts in workloads]
            u_lc_a = [ts.U_LC_A for ts in workloads]
            u_hc_l = [ts.U_HC_L for ts in workloads]
            u_lc_d = [ts.U_LC_D for ts in workloads]
            u_hc_h = [ts.U_HC_H for ts in workloads]
            n_hc = [sum(1 for t in ts.tasks if t.criticality == "HC")
                    for ts in workloads]
            n_lc = [sum(1 for t in ts.tasks if t.criticality == "LC")
                    for ts in workloads]

            writer.writerow([
                f"{cap:.2f}", n, m,
                f"{np.mean(n_tasks):.2f}",
                f"{np.mean(u_los):.4f}", f"{np.std(u_los):.4f}",
                f"{np.mean(u_his):.4f}", f"{np.std(u_his):.4f}",
                f"{np.mean(actual_caps):.4f}", f"{np.std(actual_caps):.4f}",
                f"{np.mean(u_lc_a):.4f}", f"{np.mean(u_hc_l):.4f}",
                f"{np.mean(u_lc_d):.4f}", f"{np.mean(u_hc_h):.4f}",
                f"{np.mean(n_hc):.2f}", f"{np.mean(n_lc):.2f}"
            ])

    print(f"  Summary saved: {filepath}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("IMC Task Set Generator")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Output format:    {OUTPUT_FORMAT}")
    print(f"Processors:       {PROCESSORS}")
    print(f"Workloads/cap:    {N_WORKLOADS}")
    print(f"Seed:             {SEED}")
    print(f"pCriticality:     {P_CRITICALITY}")
    print(f"R range:          [{R_MIN}, {R_MAX}]")
    print(f"Lambda range:     [{LAMBDA_MIN}, {LAMBDA_MAX}]")
    print(f"Period range:     [{PERIOD_MIN}, {PERIOD_MAX}]")
    print(f"Tasks/proc range: [{N_TASKS_PER_PROC_MIN}, {N_TASKS_PER_PROC_MAX}]")
    print(f"Util cap ratio:   [{UTIL_CAP_LO_RATIO}, {UTIL_CAP_HI_RATIO}]")
    print(f"Util cap step:    0.1 × m")

    for m in PROCESSORS:
        print(f"\n{'=' * 60}")
        print(f"Processing m = {m} processors")
        print(f"{'=' * 60}")

        generator = IMCTaskSetGenerator(
            num_processors=m,
            pCriticality=P_CRITICALITY,
            R_range=(R_MIN, R_MAX),
            lambda_range=(LAMBDA_MIN, LAMBDA_MAX),
            period_range=(PERIOD_MIN, PERIOD_MAX),
            n_tasks_per_proc_range=(N_TASKS_PER_PROC_MIN, N_TASKS_PER_PROC_MAX),
            seed=SEED + m   # different seed per processor count
        )

        all_workloads = generator.generate_all(
            n_workloads=N_WORKLOADS,
            util_cap_lo_ratio=UTIL_CAP_LO_RATIO,
            util_cap_hi_ratio=UTIL_CAP_HI_RATIO
        )

        # Export
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