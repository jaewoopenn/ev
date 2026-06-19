"""
Runtime survivability simulator (paper Section 4.3, Figures 5, 6 and 7).

Time-driven, job-level simulation of partitioned IMC task sets. Compares
three migration configurations (ablation study):

    off        - Migration OFF   : no migration; LC tasks degrade immediately
    mig_rec    - Migration Rec   : migration + home-processor recovery
                                   (full IMC-PALM protocol, Algorithms 2 & 3)
    mig_norec  - Migration NoRec : migration without recovery (tasks stay on
                                   the temporary host)

Common rules:
  - Each task migrates at most once per mode-switch event; if the target
    processor switches to HI, the task degrades immediately.
  - Under mig_rec, a successful recovery to the home processor resets the
    one-time migration limit.

Outputs:
    RESULT_DIR/imc_simulation_recovery_results.csv  (Fig 5: vary utilization)
    RESULT_DIR/imc_prob_recovery_results.csv        (Fig 6: vary P^MS)
    RESULT_DIR/imc_overhead_recovery_results.csv    (Fig 7: vary overhead a)
"""

import os
import json
import csv
import random
import copy

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from code_rev.config import (DATA_DIR, RESULT_DIR, M_VALUES, TARGETS,
                     MAX_SIM_SETS, SIM_TICKS, DEFAULT_SWITCH_PROB)
DATA_DIR = "/Users/jaewoo/data/com/data"
RESULT_DIR = "/Users/jaewoo/data/com"

def compute_x_max(U_LC_A: float, U_LC_D: float, U_HC_H: float):
    denom = U_LC_A - U_LC_D
    if denom <= 0.0:
        if U_HC_H + U_LC_D <= 1.0: return 1.0
        return None
    numer = 1.0 - U_HC_H - U_LC_D
    if numer <= 0.0: return None
    x_max = numer / denom
    if x_max > 1.0: x_max = 1.0
    return x_max

def is_schedulable_new(U_LC_A, U_HC_L, U_LC_D, U_HC_H, hc_tasks, lc_tasks):
    x = compute_x_max(U_LC_A, U_LC_D, U_HC_H)
    if x is None: return False
    lo_sum = U_LC_A
    for (u_l, u_h) in hc_tasks:
        if u_l / x >= u_h: lo_sum += u_h
        else: lo_sum += u_l / x
    return lo_sum <= 1.0

def is_schedulable_fixed_x(x_star, U_LC_A, U_LC_D, U_HC_H, hc_tasks):
    """Schedulability test at a fixed offline coefficient x_star.

    Used by the runtime migration admission check; does NOT recompute x_max.
    """
    if x_star is None or x_star <= 0.0:
        return False
    if x_star * U_LC_A + (1.0 - x_star) * U_LC_D + U_HC_H > 1.0:
        return False
    lo_sum = U_LC_A
    for (u_l, u_h) in hc_tasks:
        if u_l / x_star >= u_h:
            lo_sum += u_h
        else:
            lo_sum += u_l / x_star
    return lo_sum <= 1.0

class Processor:
    def __init__(self, proc_id, sched_func):
        self.id = proc_id
        self._sched_func = sched_func
        self.U_LC_A = 0.0
        self.U_HC_L = 0.0
        self.U_LC_D = 0.0
        self.U_HC_H = 0.0
        self.hc_tasks = []
        self.lc_tasks = []
        self.tasks = []
        self.mode = "LO"
        self.ready_queue = []
        self.running_job = None
        self.x_star = None

    def try_add(self, task: dict) -> bool:
        if task["crit"] == "HC":
            new_U_LC_A, new_U_LC_D = self.U_LC_A, self.U_LC_D
            new_U_HC_L = self.U_HC_L + task["u_LO"]
            new_U_HC_H = self.U_HC_H + task["u_HI"]
            new_hc = self.hc_tasks + [(task["u_LO"], task["u_HI"])]
            new_lc = self.lc_tasks
        else:
            new_U_LC_A = self.U_LC_A + task["u_LO"]
            new_U_LC_D = self.U_LC_D + task["u_HI"]
            new_U_HC_L, new_U_HC_H = self.U_HC_L, self.U_HC_H
            new_hc = self.hc_tasks
            new_lc = self.lc_tasks + [(task["u_LO"], task["u_HI"])]
        return self._sched_func(new_U_LC_A, new_U_HC_L, new_U_LC_D, new_U_HC_H, new_hc, new_lc)

    def try_add_fixed_x(self, task: dict) -> bool:
        """Runtime migration admission test at this processor's offline-frozen x_star."""
        if self.x_star is None:
            return False
        if task["crit"] == "HC":
            new_U_LC_A, new_U_LC_D = self.U_LC_A, self.U_LC_D
            new_U_HC_H = self.U_HC_H + task["u_HI"]
            new_hc = self.hc_tasks + [(task["u_LO"], task["u_HI"])]
        else:
            new_U_LC_A = self.U_LC_A + task["u_LO"]
            new_U_LC_D = self.U_LC_D + task["u_HI"]
            new_U_HC_H = self.U_HC_H
            new_hc = self.hc_tasks
        return is_schedulable_fixed_x(
            self.x_star, new_U_LC_A, new_U_LC_D, new_U_HC_H, new_hc
        )

    def add(self, task: dict):
        if task["crit"] == "HC":
            self.U_HC_L += task["u_LO"]
            self.U_HC_H += task["u_HI"]
            self.hc_tasks.append((task["u_LO"], task["u_HI"]))
        else:
            self.U_LC_A += task["u_LO"]
            self.U_LC_D += task["u_HI"]
            self.lc_tasks.append((task["u_LO"], task["u_HI"]))
        self.tasks.append(task)

    def remove(self, task: dict):
        if task in self.tasks:
            self.tasks.remove(task)
            saved_tasks = list(self.tasks)
            self.U_LC_A = self.U_HC_L = self.U_LC_D = self.U_HC_H = 0.0
            self.hc_tasks = []
            self.lc_tasks = []
            self.tasks = []
            for t in saved_tasks:
                self.add(t)

def partition_ffd_new(tasks, m):
    sorted_tasks = sorted(tasks, key=lambda t: max(t["u_LO"], t["u_HI"]), reverse=True)
    procs = [Processor(i, is_schedulable_new) for i in range(m)]
    for task in sorted_tasks:
        placed = False
        for p in procs:
            if p.try_add(task):
                p.add(task)
                placed = True
                break
        if not placed:
            return None
            
    for p in procs:
        p.x_star = compute_x_max(p.U_LC_A, p.U_LC_D, p.U_HC_H)
    return procs

def make_inflated_view(task, alpha):
    v = dict(task)
    base_u = task.get("orig_u_LO", task["u_LO"])
    base_c = task.get("orig_c_LO", task["c_LO"])
    v["u_LO"] = base_u * (1.0 + alpha)
    v["c_LO"] = base_c * (1.0 + alpha)
    return v

def _degrade(p, task, deg_ref):
    for j in p.ready_queue:
        if j["task"]["id"] == task["id"]:
            j["degraded"] = True
    if p.running_job and p.running_job["task"]["id"] == task["id"]:
        p.running_job["degraded"] = True

def run_simulation(base_tasks, m, sim_ticks, mig_mode, switch_prob=0.20, mig_alpha=0.0):
    runtime_tasks = copy.deepcopy(base_tasks)
    procs = partition_ffd_new(runtime_tasks, m)
    if procs is None:
        return 0, 0, 0, 0, 0.0, 0.0
    for rt in runtime_tasks:
        home = next(p for p in procs if any(t is rt for t in p.tasks))
        rt["home_proc"] = home
        rt["current_proc"] = home
        rt["orig_u_LO"] = rt["u_LO"]
        rt["orig_c_LO"] = rt["c_LO"]
        rt["migrated_once"] = False

    total_jobs = 0
    deg_ref = [0]
    lc_total = [0]
    mig_count = [0]    # 성공한 마이그레이션 수
    rec_count = [0]    # home 복구 수
    opt_total = [0.0]  # 전체 LC 잡의 C_O 합
    opt_lost  = [0.0]  # degrade된 LC 잡의 C_O 합
    
    for tick in range(sim_ticks):
        for t in runtime_tasks:
            if tick % t["period"] == 0:
                total_jobs += 1
                ms = (random.random() < switch_prob) if t["crit"] == "HC" else False
                job = {"task": t, "id": total_jobs, "deadline": tick + t["period"],
                       "rem_LO": t["c_LO"], "rem_HI": t["c_HI"], "started": False,
                       "do_switch": ms}
                if t["crit"] == "LC":
                    lc_total[0] += 1
                    opt_total[0] += (t["orig_c_LO"] - t["c_HI"])   # C_O = C_L - C_M
                    if t["current_proc"].mode == "HI":
                        job["degraded"] = True
                t["current_proc"].ready_queue.append(job)

        for p in procs:
            if p.running_job:
                jb = p.running_job
                done = (jb["rem_HI"] <= 0) if jb.get("degraded", False) else \
                       (jb["rem_LO"] <= 0 if p.mode == "LO" else jb["rem_HI"] <= 0)
                if done:
                    if jb["task"]["crit"] == "LC" and jb.get("degraded", False):
                        deg_ref[0] += 1
                        opt_lost[0] += (jb["task"]["orig_c_LO"] - jb["task"]["c_HI"])
                        jb["_counted"] = True
                    p.running_job = None

            if p.running_job is None and len(p.ready_queue) == 0 and p.mode == "HI":
                p.mode = "LO"

                if mig_mode == "mig_norec":
                    for t in runtime_tasks:
                        if t["crit"] == "LC" and t["current_proc"] == p \
                           and t["migrated_once"]:
                            t["migrated_once"] = False

                if mig_mode == "mig_rec":
                    for t in runtime_tasks:
                        if t["home_proc"] == p and t["current_proc"] != p:
                            old_p = t["current_proc"]
                            old_p.remove(t)
                            t["u_LO"] = t["orig_u_LO"]
                            t["c_LO"] = t["orig_c_LO"]
                            t["migrated_once"] = False
                            p.add(t)
                            t["current_proc"] = p
                            rec_count[0] += 1

                            mv = [j for j in old_p.ready_queue if j["task"]["id"] == t["id"]]
                            old_p.ready_queue = [j for j in old_p.ready_queue if j["task"]["id"] != t["id"]]
                            p.ready_queue.extend(mv)

                            if old_p.running_job and old_p.running_job["task"]["id"] == t["id"]:
                                p.ready_queue.append(old_p.running_job)
                                old_p.running_job = None

            if p.running_job is None and p.ready_queue:
                p.ready_queue.sort(key=lambda j: j["deadline"])
                p.running_job = p.ready_queue.pop(0)

                if not p.running_job["started"]:
                    p.running_job["started"] = True
                    if p.running_job["task"]["crit"] == "HC" and p.mode == "LO":
                        if p.running_job.get("do_switch", False):
                            p.mode = "HI"

                            candidates = sorted(
                                [t for t in runtime_tasks if t["crit"] == "LC" and t["current_proc"] == p],
                                key=lambda t: t["u_LO"]
                            )

                            for lc in candidates:
                                if mig_mode == "off" or lc["migrated_once"]:
                                    _degrade(p, lc, deg_ref)
                                    continue

                                migrated = False
                                check = make_inflated_view(lc, mig_alpha) if mig_alpha > 0.0 else lc

                                for tp in procs:
                                    # 이 부분이 수정되었습니다: try_add가 아닌 try_add_fixed_x 사용
                                    if tp != p and tp.mode == "LO" and tp.try_add_fixed_x(check):
                                        p.remove(lc)

                                        lc["u_LO"] = lc["orig_u_LO"]
                                        lc["c_LO"] = lc["orig_c_LO"]

                                        tp.add(lc)
                                        lc["current_proc"] = tp
                                        lc["migrated_once"] = True
                                        mig_count[0] += 1
                                        mv = [j for j in p.ready_queue if j["task"]["id"] == lc["id"]]
                                        p.ready_queue = [j for j in p.ready_queue if j["task"]["id"] != lc["id"]]

                                        if mig_alpha > 0.0:
                                            for j in mv:
                                                if j["rem_LO"] > 0:
                                                    j["rem_LO"] += j["rem_LO"] * mig_alpha

                                        tp.ready_queue.extend(mv)
                                        migrated = True
                                        break

                                if not migrated:
                                    _degrade(p, lc, deg_ref)

            if p.running_job:
                if p.running_job.get("degraded", False):
                    p.running_job["rem_HI"] -= 1
                elif p.mode == "LO":
                    p.running_job["rem_LO"] -= 1
                else:
                    p.running_job["rem_HI"] -= 1

    for p in procs:
        for j in p.ready_queue:
            if j["task"]["crit"] == "LC" and j.get("degraded", False) \
               and not j.get("_counted", False):
                deg_ref[0] += 1
                opt_lost[0] += (j["task"]["orig_c_LO"] - j["task"]["c_HI"])
                j["_counted"] = True
        if p.running_job and p.running_job["task"]["crit"] == "LC" \
           and p.running_job.get("degraded", False) \
           and not p.running_job.get("_counted", False):
            deg_ref[0] += 1
            opt_lost[0] += (p.running_job["task"]["orig_c_LO"] - p.running_job["task"]["c_HI"])   
    return lc_total[0], deg_ref[0], mig_count[0], rec_count[0], opt_total[0], opt_lost[0]

def prepare_sets(all_tasks, m, max_sets):
    prepared = []
    for task_set in all_tasks:
        if len(prepared) >= max_sets: break
        for i, t in enumerate(task_set):
            if "id" not in t: t["id"] = i
            if "c_LO" not in t: t["c_LO"] = max(1, int(t["u_LO"] * t["period"]))
            if "c_HI" not in t: t["c_HI"] = max(1, int(t["u_HI"] * t["period"]))
        if partition_ffd_new(copy.deepcopy(task_set), m) is None: continue
        prepared.append(copy.deepcopy(task_set))
    return prepared

def eval_point(prepared, m, sim_ticks, switch_prob, alpha):
    import statistics
    modes = ["off", "mig_rec", "mig_norec"]
    acc = {mode: {"total": 0, "degrade": 0} for mode in modes}
    opt = {mode: {"total": 0.0, "lost": 0.0} for mode in modes}
    per_set = {mode: [] for mode in modes}     # 태스크셋별 DJR(%)
    mig_counts, rec_counts = [], []            # mig_rec 기준 시뮬당 마이그레이션/복구 횟수
    for task_set in prepared:
        for mode in modes:
            random.seed(42)
            t_total, d_total, n_mig, n_rec, o_tot, o_lost = run_simulation(
                task_set, m, sim_ticks,
                mig_mode=mode, switch_prob=switch_prob, mig_alpha=alpha
            )
            acc[mode]["total"]   += t_total
            acc[mode]["degrade"] += d_total
            opt[mode]["total"]   += o_tot
            opt[mode]["lost"]    += o_lost
            if t_total > 0:
                per_set[mode].append(100 * d_total / t_total)
            if mode == "mig_rec":
                mig_counts.append(n_mig)
                rec_counts.append(n_rec)
    if acc["off"]["total"] == 0:
        return None

    def pooled(mode):  return 100 * acc[mode]["degrade"] / acc[mode]["total"]
    def stdev(mode):
        v = per_set[mode]; return statistics.stdev(v) if len(v) > 1 else 0.0
    def lostratio(mode):
        o = opt[mode]; return 100 * o["lost"] / o["total"] if o["total"] > 0 else 0.0

    return (acc["off"]["total"],                                    # r0  Total_Jobs
            pooled("off"), pooled("mig_rec"), pooled("mig_norec"),  # r1-3  pooled DJR(%)  ← 기존과 동일
            stdev("off"),  stdev("mig_rec"),  stdev("mig_norec"),   # r4-6  태스크셋 간 std
            lostratio("off"), lostratio("mig_rec"), lostratio("mig_norec"),  # r7-9 lost-optional(%)
            (sum(mig_counts)/len(mig_counts) if mig_counts else 0), # r10 평균 마이그레이션/시뮬
            (max(mig_counts) if mig_counts else 0),                 # r11 최대 마이그레이션
            (sum(rec_counts)/len(rec_counts) if rec_counts else 0), # r12 평균 복구/시뮬
            (max(rec_counts) if rec_counts else 0))                 # r13 최대 복구
    
def main():
    m_values = M_VALUES
    data_dir = DATA_DIR
    result_dir = RESULT_DIR
    max_sets = MAX_SIM_SETS
    sim_ticks = SIM_TICKS
    pms = DEFAULT_SWITCH_PROB

    # 세 sweep 공통 16컬럼 스키마 (2번째 컬럼 라벨만 다름)
    def header(varlabel):
        return ["m", varlabel, "Total_Jobs",
                "Degrade_OFF", "Degrade_Mig_Rec", "Degrade_Mig_NoRec",   # r1-3
                "Std_OFF", "Std_Mig_Rec", "Std_Mig_NoRec",               # r4-6
                "Lost_OFF", "Lost_Mig_Rec", "Lost_Mig_NoRec",            # r7-9
                "Avg_Mig", "Max_Mig", "Avg_Rec", "Max_Rec"]              # r10-13

    def row(m, var, r):
        return [m, var] + list(r)   # r0=Total_Jobs … r13=Max_Rec → 총 16컬럼

    # ---- Fig5: vary target utilization ----
    targets = TARGETS
    with open(os.path.join(result_dir, "imc_simulation_recovery_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header("Target"))
        for m in m_values:
            for tg in targets:
                fp = os.path.join(data_dir, f"stasks_m_{m}_target_{tg:.2f}.json")
                if not os.path.exists(fp): continue
                with open(fp) as jf: allt = json.load(jf)
                prepared = prepare_sets(allt, m, max_sets)
                r = eval_point(prepared, m, sim_ticks, switch_prob=pms, alpha=0.0)
                if r:
                    w.writerow(row(m, tg, r))
                    print(f"[Fig5] m={m} Ut={tg:.2f} -> OFF={r[1]:.2f} REC={r[2]:.2f} NOREC={r[3]:.2f} "
                          f"| std(Rec)={r[5]:.2f} lost(Rec)={r[8]:.1f}% avgMig={r[10]:.1f} avgRec={r[12]:.1f}")

    # ---- Fig6: vary mode-switch prob ----
    probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    with open(os.path.join(result_dir, "imc_prob_recovery_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header("Prob"))
        for m in m_values:
            fp = os.path.join(data_dir, f"stasks_m_{m}_target_0.70.json")
            if not os.path.exists(fp): continue
            with open(fp) as jf: allt = json.load(jf)
            prepared = prepare_sets(allt, m, max_sets)
            for pr in probs:
                r = eval_point(prepared, m, sim_ticks, switch_prob=pr, alpha=0.0)
                if r:
                    w.writerow(row(m, pr, r))
                    print(f"[Fig6] m={m} PMS={pr:.1f} -> OFF={r[1]:.2f} REC={r[2]:.2f} NOREC={r[3]:.2f} "
                          f"| std(Rec)={r[5]:.2f} lost(Rec)={r[8]:.1f}% avgMig={r[10]:.1f} avgRec={r[12]:.1f}")

    # ---- Fig7: vary overhead alpha ----
    alphas = [0.0, 0.01, 0.03, 0.05, 0.10]
    with open(os.path.join(result_dir, "imc_overhead_recovery_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header("Alpha"))
        for m in m_values:
            fp = os.path.join(data_dir, f"stasks_m_{m}_target_0.70.json")
            if not os.path.exists(fp): continue
            with open(fp) as jf: allt = json.load(jf)
            prepared = prepare_sets(allt, m, max_sets)
            for a in alphas:
                r = eval_point(prepared, m, sim_ticks, switch_prob=pms, alpha=a)
                if r:
                    w.writerow(row(m, a, r))
                    print(f"[Fig7] m={m} a={a:.2f} -> OFF={r[1]:.2f} REC={r[2]:.2f} NOREC={r[3]:.2f} "
                          f"| std(Rec)={r[5]:.2f} lost(Rec)={r[8]:.1f}% avgMig={r[10]:.1f} avgRec={r[12]:.1f}")

if __name__ == "__main__":
    main()
