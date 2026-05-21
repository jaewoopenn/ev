import os
import json
import csv
import random
import copy

# ============================================================
#  통합 시뮬레이터 (Ablation Study 반영)
#
#  비교 모드 (mig_mode)
#  ---------
#  1. off       : Migration 없음 (발생 즉시 degrade)
#  2. mig_rec   : Migration 허용 + 홈 코어가 LO로 복귀 시 태스크 회수 (Recovery O)
#  3. mig_norec : Migration 허용 + 홈 코어가 LO로 복귀해도 타겟 코어에 잔류 (Recovery X)
#
#  공통 규칙
#  ---------
#  - 한 task는 1회만 migration 가능. 타겟 코어가 HI로 바뀌면 즉시 degrade.
#  - 단, mig_rec 모드에서 홈으로 복귀(Recovery) 성공 시 이주 제한(1회) 리셋.
# ============================================================

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

def is_schedulable_new(U_LC_A, U_HC_L, U_LC_D, U_HC_H, hc_tasks, lc_tasks, fixed_x=None):
    if fixed_x is None:
        # 오프라인 파티셔닝 단계: x를 동적으로 계산
        x = compute_x_max(U_LC_A, U_LC_D, U_HC_H)
        if x is None: return False
    else:
        # 런타임 Migration 단계: 고정된 x를 사용하여 Eq(7) 및 Eq(11) 만족 여부 검사
        x = fixed_x
        # HI-mode 조건 확인 (Eq. 7) -> 부동소수점 오차 방지를 위해 1e-9 여유분 허용
        if (x * U_LC_A) + ((1.0 - x) * U_LC_D) + U_HC_H > 1.0 + 1e-9:
            return False

    lo_sum = U_LC_A
    for (u_l, u_h) in hc_tasks:
        if u_l / x >= u_h: lo_sum += u_h
        else: lo_sum += u_l / x
    
    # 부동소수점 오차 방지를 위해 1e-9 여유분 허용
    return lo_sum <= 1.0 + 1e-9

class Processor:
    def __init__(self, proc_id, sched_func):
        self.id = proc_id
        self._sched_func = sched_func
        self.fixed_x = None  # 오프라인 파티셔닝 완료 후 고정될 x 계수
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
        
        # 런타임 중이라면 고정된 self.fixed_x를 넘겨 재계산을 방지함
        return self._sched_func(new_U_LC_A, new_U_HC_L, new_U_LC_D, new_U_HC_H, new_hc, new_lc, self.fixed_x)

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
            
    # 파티셔닝이 모두 완료된 후, 논문 명세에 따라 각 코어의 x를 고정함
    for p in procs:
        p.fixed_x = compute_x_max(p.U_LC_A, p.U_LC_D, p.U_HC_H)
        if p.fixed_x is None:
            p.fixed_x = 1.0  # degenerate edge case 대응
            
    return procs

def make_inflated_view(task, alpha):
    v = dict(task)
    base_u = task.get("orig_u_LO", task["u_LO"])
    base_c = task.get("orig_c_LO", task["c_LO"])
    v["u_LO"] = base_u * (1.0 + alpha)
    v["c_LO"] = base_c * (1.0 + alpha)
    return v

def _degrade(p, task, deg_ref):
    # Mark all currently-pending/running jobs of this task as degraded.
    # Counting is done once-per-job at job exit (see run loop), NOT here.
    for j in p.ready_queue:
        if j["task"]["id"] == task["id"]:
            j["degraded"] = True
    if p.running_job and p.running_job["task"]["id"] == task["id"]:
        p.running_job["degraded"] = True

def run_simulation(base_tasks, m, sim_ticks, mig_mode, switch_prob=0.20, mig_alpha=0.0):
    runtime_tasks = copy.deepcopy(base_tasks)
    procs = partition_ffd_new(runtime_tasks, m)
    if procs is None:
        return 0, 0

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

    for tick in range(sim_ticks):
        # 1. Job Release
        for t in runtime_tasks:
            if tick % t["period"] == 0:
                total_jobs += 1
                ms = (random.random() < switch_prob) if t["crit"] == "HC" else False
                job = {"task": t, "id": total_jobs, "deadline": tick + t["period"],
                       "rem_LO": t["c_LO"], "rem_HI": t["c_HI"], "started": False,
                       "do_switch": ms}
                if t["crit"] == "LC":
                    lc_total[0] += 1
                    if t["current_proc"].mode == "HI":
                        job["degraded"] = True
                t["current_proc"].ready_queue.append(job)

        # 2. 스케줄링
        for p in procs:
            if p.running_job:
                jb = p.running_job
                done = (jb["rem_HI"] <= 0) if jb.get("degraded", False) else \
                       (jb["rem_LO"] <= 0 if p.mode == "LO" else jb["rem_HI"] <= 0)
                if done:
                    if jb["task"]["crit"] == "LC" and jb.get("degraded", False):
                        deg_ref[0] += 1
                        jb["_counted"] = True
                    p.running_job = None

            # Recovery: idle -> LO
            if p.running_job is None and len(p.ready_queue) == 0 and p.mode == "HI":
                p.mode = "LO"

                # NoRec: task는 잔류하되, 호스트 코어가 idle->LO 복귀하면
                # 그 코어에 머무는 migrated task의 1회 제한을 해제하여
                # 이후 mode switch 때 다시 migrate 가능하게 한다.
                if mig_mode == "mig_norec":
                    for t in runtime_tasks:
                        if t["crit"] == "LC" and t["current_proc"] == p \
                           and t["migrated_once"]:
                            t["migrated_once"] = False

                # mig_rec 모드일 때만 원래 코어로 복귀시킴
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
                                # Admission Test: 마이그레이션 직후의 일회성 오버헤드를 감당할 수 있는지 보수적으로 검사
                                check = make_inflated_view(lc, mig_alpha) if mig_alpha > 0.0 else lc

                                for tp in procs:
                                    # 고정된 target의 x값을 바탕으로 검증이 이루어지게 됨 (try_add 내에서 처리)
                                    if tp != p and tp.mode == "LO" and tp.try_add(check):
                                        p.remove(lc)
                                        
                                        # [중대한 수학적 무결성 수정]
                                        # EDF 이용률 기반 Schedulability Test의 무결성을 지키기 위해,
                                        # Admission Test를 통과한 부풀려진 밀도를 타겟 코어에서도 그대로 유지해야 합니다.
                                        # 회수(Recovery) 전까지 해당 페널티는 타겟 코어의 여유 용량(U_LC_A)을 영구적으로 점유합니다.
                                        lc["u_LO"] = check["u_LO"]
                                        lc["c_LO"] = check["c_LO"]
                                        
                                        tp.add(lc)
                                        lc["current_proc"] = tp
                                        lc["migrated_once"] = True 

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
                j["_counted"] = True
        if p.running_job and p.running_job["task"]["crit"] == "LC" \
           and p.running_job.get("degraded", False) \
           and not p.running_job.get("_counted", False):
            deg_ref[0] += 1
    return lc_total[0], deg_ref[0]

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
    acc = {mode: {"total": 0, "degrade": 0} for mode in ["off", "mig_rec", "mig_norec"]}
    for task_set in prepared:
        for mode in ["off", "mig_rec", "mig_norec"]:
            random.seed(42)
            t_total, d_total = run_simulation(
                task_set, m, sim_ticks,
                mig_mode=mode, switch_prob=switch_prob, mig_alpha=alpha
            )
            acc[mode]["total"] += t_total
            acc[mode]["degrade"] += d_total
    if acc["off"]["total"] == 0: return None
    return (acc["off"]["total"],
            100 * acc["off"]["degrade"] / acc["off"]["total"],
            100 * acc["mig_rec"]["degrade"] / acc["mig_rec"]["total"],
            100 * acc["mig_norec"]["degrade"] / acc["mig_norec"]["total"])

def main():
    m_values = [2, 4, 8]
    data_dir = "/Users/jaewoo/data/com/data"
    result_dir = "/Users/jaewoo/data/com"
    max_sets = 1000
    # max_sets = 100
    sim_ticks = 10000
    
    # ---- Fig5: vary target utilization ----
    targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    with open(os.path.join(result_dir, "imc_simulation_recovery_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "Target", "Total_Jobs", "Degrade_OFF", "Degrade_Mig_Rec", "Degrade_Mig_NoRec"])
        for m in m_values:
            for tg in targets:
                fp = os.path.join(data_dir, f"stasks_m_{m}_target_{tg:.2f}.json")
                if not os.path.exists(fp): continue
                with open(fp) as jf: allt = json.load(jf)
                prepared = prepare_sets(allt, m, max_sets)
                r = eval_point(prepared, m, sim_ticks, switch_prob=0.20, alpha=0.0)
                if r:
                    w.writerow([m, tg, r[0], r[1], r[2], r[3]])
                    print(f"[Fig5] m={m} Ut={tg:.2f} -> OFF={r[1]:.2f} REC={r[2]:.2f} NOREC={r[3]:.2f}")
    
    # ---- Fig6: vary mode-switch prob ----
    probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    with open(os.path.join(result_dir, "imc_prob_recovery_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "Prob", "Total_Jobs", "Degrade_OFF", "Degrade_Mig_Rec", "Degrade_Mig_NoRec"])
        for m in m_values:
            fp = os.path.join(data_dir, f"stasks_m_{m}_target_0.70.json")
            if not os.path.exists(fp): continue
            with open(fp) as jf: allt = json.load(jf)
            prepared = prepare_sets(allt, m, max_sets)
            for pr in probs:
                r = eval_point(prepared, m, sim_ticks, switch_prob=pr, alpha=0.0)
                if r:
                    w.writerow([m, pr, r[0], r[1], r[2], r[3]])
                    print(f"[Fig6] m={m} PMS={pr:.1f} -> OFF={r[1]:.2f} REC={r[2]:.2f} NOREC={r[3]:.2f}")

    # ---- Fig7: vary overhead alpha ----
    alphas = [0.0, 0.01, 0.03, 0.05, 0.10]
    with open(os.path.join(result_dir, "imc_overhead_recovery_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "Alpha", "Total_Jobs", "Degrade_OFF", "Degrade_Mig_Rec", "Degrade_Mig_NoRec"])
        for m in m_values:
            fp = os.path.join(data_dir, f"stasks_m_{m}_target_0.70.json")
            if not os.path.exists(fp): continue
            with open(fp) as jf: allt = json.load(jf)
            prepared = prepare_sets(allt, m, max_sets)
            for a in alphas:
                r = eval_point(prepared, m, sim_ticks, switch_prob=0.20, alpha=a)
                if r:
                    w.writerow([m, a, r[0], r[1], r[2], r[3]])
                    print(f"[Fig7] m={m} a={a:.2f} -> OFF={r[1]:.2f} REC={r[2]:.2f} NOREC={r[3]:.2f}")

if __name__ == "__main__":
    main()