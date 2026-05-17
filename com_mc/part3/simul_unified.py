import os
import json
import csv
import random
import copy

# ============================================================
#  통합 시뮬레이터 (Fig5 / Fig6 / Fig7 공용)
#
#  핵심 정의
#  ---------
#  single (B):  한 task 는 하나의 mode-switch 사이클 내에서
#               최대 1회만 migration 가능. 한 번 migrate 된
#               task 는 그 host 가 이후 HI 로 바뀌어도
#               재migration 하지 않고 즉시 degrade 한다
#               (chain 처럼 여러 host 를 따라가지 않음).
#               단, recovery 로 home 에 복귀하면 시스템이
#               원래 partition 상태로 복원되므로 1회 제한도
#               리셋되어, 다음 mode-switch 때 다시 1회 migrate
#               할 자격을 회복한다.
#  chain     :  migration 이력에 무관하게 current_proc 기준으로
#               계속 따라가며 재migration 가능.
#
#  버그 수정
#  ---------
#  - run_simulation 내부에서 runtime_tasks 로 직접 partition 하여
#    procs.tasks 와 runtime_tasks 가 *동일 객체* 를 공유한다.
#    (원본 simul.py 는 procs=deepcopy A, runtime=deepcopy B 로
#     분리되어 single 분기가 시뮬 본체에 반영되지 않는 결함이 있었음.)
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

def is_schedulable_new(U_LC_A, U_HC_L, U_LC_D, U_HC_H, hc_tasks, lc_tasks):
    x = compute_x_max(U_LC_A, U_LC_D, U_HC_H)
    if x is None: return False
    lo_sum = U_LC_A
    for (u_l, u_h) in hc_tasks:
        if u_l / x >= u_h: lo_sum += u_h
        else: lo_sum += u_l / x
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
    return procs

def make_inflated_view(task, alpha):
    v = dict(task)
    v["u_LO"] = task["u_LO"] * (1.0 + alpha)
    v["c_LO"] = task["c_LO"] * (1.0 + alpha)
    return v

def _degrade(p, task, deg_ref):
    n = 0
    jobs = [j for j in p.ready_queue if j["task"]["id"] == task["id"]]
    p.ready_queue[:] = [j for j in p.ready_queue if j["task"]["id"] != task["id"]]
    n += len(jobs)
    if p.running_job and p.running_job["task"]["id"] == task["id"]:
        n += 1
        p.running_job = None
    deg_ref[0] += n

def run_simulation(base_tasks, m, sim_ticks, mig_mode,
                   switch_prob=0.20, mig_alpha=0.0):
    runtime_tasks = copy.deepcopy(base_tasks)

    # [FIX] runtime_tasks 로 직접 partition -> procs.tasks 와
    #       runtime_tasks 가 동일 객체를 공유.
    procs = partition_ffd_new(runtime_tasks, m)
    if procs is None:
        return 0, 0

    for rt in runtime_tasks:
        home = next(p for p in procs if any(t is rt for t in p.tasks))
        rt["home_proc"] = home
        rt["current_proc"] = home
        rt["orig_u_LO"] = rt["u_LO"]
        rt["orig_c_LO"] = rt["c_LO"]
        rt["migrated_once"] = False        # single=(B): 사이클당 1회 migration 플래그

    allow_migration = (mig_mode != "off")
    use_chain = (mig_mode == "chain")

    total_jobs = 0
    deg_ref = [0]

    for tick in range(sim_ticks):
        # 1. Job Release
        for t in runtime_tasks:
            if tick % t["period"] == 0:
                total_jobs += 1
                job = {"task": t, "id": total_jobs, "deadline": tick + t["period"],
                       "rem_LO": t["c_LO"], "rem_HI": t["c_HI"], "started": False}
                if t["crit"] == "LC" and t["current_proc"].mode == "HI":
                    deg_ref[0] += 1
                else:
                    t["current_proc"].ready_queue.append(job)

        # 2. 스케줄링
        for p in procs:
            if p.running_job:
                if (p.mode == "LO" and p.running_job["rem_LO"] <= 0) or \
                   (p.mode == "HI" and p.running_job["rem_HI"] <= 0):
                    p.running_job = None
            # 기존 코드의 Recovery 부분 (run_simulation 내)
            if p.running_job is None and len(p.ready_queue) == 0 and p.mode == "HI":
                p.mode = "LO"
                if allow_migration:
                    for t in runtime_tasks:
                        if t["home_proc"] == p and t["current_proc"] != p:
                            old_p = t["current_proc"]
                            old_p.remove(t)
                            t["u_LO"] = t["orig_u_LO"]
                            t["c_LO"] = t["orig_c_LO"]
                            t["migrated_once"] = False
                            p.add(t)
                            t["current_proc"] = p
                            
                            # [수정] old_p에 남아있는 Job들을 원래 홈 코어의 큐로 반드시 회수해야 합니다.
                            mv = [j for j in old_p.ready_queue if j["task"]["id"] == t["id"]]
                            old_p.ready_queue = [j for j in old_p.ready_queue if j["task"]["id"] != t["id"]]
                            p.ready_queue.extend(mv)
                            
                            # [수정] old_p에서 방금 실행 중이던 Job도 회수
                            if old_p.running_job and old_p.running_job["task"]["id"] == t["id"]:
                                p.ready_queue.append(old_p.running_job)
                                old_p.running_job = None

            if p.running_job is None and p.ready_queue:
                p.ready_queue.sort(key=lambda j: j["deadline"])
                p.running_job = p.ready_queue.pop(0)

                if not p.running_job["started"]:
                    p.running_job["started"] = True
                    if p.running_job["task"]["crit"] == "HC" and p.mode == "LO":
                        if random.random() < switch_prob:
                            p.mode = "HI"

                            # 수집 기준은 current_proc 로 통일
                            # (procs.tasks 와 runtime_tasks 가 동일 객체이므로
                            #  p.tasks 로 모아도 결과는 같지만, 명시적으로
                            #  current_proc 기준을 사용한다.)
                            candidates = sorted(
                                [t for t in runtime_tasks
                                 if t["crit"] == "LC" and t["current_proc"] == p],
                                key=lambda t: t["u_LO"]
                            )

                            if use_chain:
                                lc_tasks = candidates
                            else:
                                # single=(B): 아직 한 번도 migrate 안 한 task 만
                                #             migration 시도. 이미 1회 migrate
                                #             이력이 있으면 즉시 degrade.
                                lc_tasks = []
                                for t in candidates:
                                    if t["migrated_once"]:
                                        _degrade(p, t, deg_ref)
                                    else:
                                        lc_tasks.append(t)

                            for lc in lc_tasks:
                                if not allow_migration:
                                    _degrade(p, lc, deg_ref)
                                    continue

                                migrated = False
                                check = make_inflated_view(lc, mig_alpha) if mig_alpha > 0.0 else lc

                                for tp in procs:
                                    if tp != p and tp.mode == "LO" and tp.try_add(check):
                                        p.remove(lc)
                                        if mig_alpha > 0.0:
                                            lc["u_LO"] = lc["orig_u_LO"] * (1.0 + mig_alpha)
                                            lc["c_LO"] = lc["orig_c_LO"] * (1.0 + mig_alpha)
                                        tp.add(lc)
                                        lc["current_proc"] = tp
                                        lc["migrated_once"] = True   # 평생 1회 기록

                                        mv = [j for j in p.ready_queue if j["task"]["id"] == lc["id"]]
                                        p.ready_queue = [j for j in p.ready_queue if j["task"]["id"] != lc["id"]]
                                        if mig_alpha > 0.0:
                                            for j in mv:
                                                j["rem_LO"] += lc["orig_c_LO"] * mig_alpha
                                        tp.ready_queue.extend(mv)
                                        migrated = True
                                        break
                                if not migrated:
                                    _degrade(p, lc, deg_ref)

            if p.running_job:
                if p.mode == "LO": p.running_job["rem_LO"] -= 1
                else: p.running_job["rem_HI"] -= 1

    return total_jobs, deg_ref[0]

# ============================================================
#  워크로드 준비
#  - period / c_LO / c_HI 는 task-set 파일에 있는 값을 그대로 사용한다.
#    (덮어쓰지 않음. 파일이 이미 이 값들을 포함한다.)
#  - c_LO / c_HI 가 누락된 경우에 한해 u * period 로 보정한다.
# ============================================================
def prepare_sets(all_tasks, m, max_sets):
    prepared = []
    for task_set in all_tasks:
        if len(prepared) >= max_sets:
            break
        for i, t in enumerate(task_set):
            if "id" not in t:
                t["id"] = i
            # period 는 파일 값 사용 (필수)
            if "c_LO" not in t:
                t["c_LO"] = max(1, int(t["u_LO"] * t["period"]))
            if "c_HI" not in t:
                t["c_HI"] = max(1, int(t["u_HI"] * t["period"]))
        if partition_ffd_new(copy.deepcopy(task_set), m) is None:
            continue
        prepared.append(copy.deepcopy(task_set))
    return prepared

def eval_point(prepared, m, sim_ticks, switch_prob, alpha):
    acc = {mode: {"total": 0, "degrade": 0} for mode in ["off", "single", "chain"]}
    for task_set in prepared:
        for mode in ["off", "single", "chain"]:
            random.seed(42)
            t_total, d_total = run_simulation(
                task_set, m, sim_ticks,
                mig_mode=mode, switch_prob=switch_prob, mig_alpha=alpha
            )
            acc[mode]["total"] += t_total
            acc[mode]["degrade"] += d_total
    if acc["off"]["total"] == 0:
        return None
    return (acc["off"]["total"],
            100 * acc["off"]["degrade"] / acc["off"]["total"],
            100 * acc["single"]["degrade"] / acc["single"]["total"],
            100 * acc["chain"]["degrade"] / acc["chain"]["total"])

# ============================================================
#  메인: 세 실험을 동일 로직으로 생성
# ============================================================
def main():
    m_values = [2,4, 8]
    data_dir = "/Users/jaewoo/data/com/data"
    result_dir = "/Users/jaewoo/data/com"
    # max_sets = 1000
    # max_sets = 100
    max_sets = 50
    sim_ticks = 10000

    # ---- Fig5: vary target utilization, PMS=0.2 ----
    targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    with open(os.path.join(result_dir, "imc_simulation_3way_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "Target", "Total_Jobs", "Degrade_OFF", "Degrade_Single", "Degrade_Chain"])
        for m in m_values:
            for tg in targets:
                fp = os.path.join(data_dir, f"stasks_m_{m}_target_{tg:.2f}.json")
                if not os.path.exists(fp):
                    continue
                with open(fp) as jf:
                    allt = json.load(jf)
                prepared = prepare_sets(allt, m, max_sets)
                r = eval_point(prepared, m, sim_ticks, switch_prob=0.20, alpha=0.0)
                if r:
                    w.writerow([m, tg, r[0], r[1], r[2], r[3]])
                    print(f"[Fig5] m={m} Ut={tg:.2f} -> OFF={r[1]:.2f} S={r[2]:.2f} C={r[3]:.2f}")

    # ---- Fig6: vary mode-switch prob, Ut=0.70 ----
    probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    with open(os.path.join(result_dir, "imc_prob_3way_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "Prob", "Total_Jobs", "Degrade_OFF", "Degrade_Single", "Degrade_Chain"])
        for m in m_values:
            fp = os.path.join(data_dir, f"stasks_m_{m}_target_0.70.json")
            if not os.path.exists(fp):
                continue
            with open(fp) as jf:
                allt = json.load(jf)
            prepared = prepare_sets(allt, m, max_sets)
            for pr in probs:
                r = eval_point(prepared, m, sim_ticks, switch_prob=pr, alpha=0.0)
                if r:
                    w.writerow([m, pr, r[0], r[1], r[2], r[3]])
                    print(f"[Fig6] m={m} PMS={pr:.1f} -> OFF={r[1]:.2f} S={r[2]:.2f} C={r[3]:.2f}")

    # ---- Fig7: vary overhead alpha, PMS=0.2, Ut=0.70 ----
    alphas = [0.0, 0.01, 0.03, 0.05, 0.10]
    with open(os.path.join(result_dir, "imc_overhead_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "Alpha", "Total_Jobs", "Degrade_OFF", "Degrade_Single", "Degrade_Chain"])
        for m in m_values:
            fp = os.path.join(data_dir, f"stasks_m_{m}_target_0.70.json")
            if not os.path.exists(fp):
                continue
            with open(fp) as jf:
                allt = json.load(jf)
            prepared = prepare_sets(allt, m, max_sets)
            for a in alphas:
                r = eval_point(prepared, m, sim_ticks, switch_prob=0.20, alpha=a)
                if r:
                    w.writerow([m, a, r[0], r[1], r[2], r[3]])
                    print(f"[Fig7] m={m} a={a:.2f} -> OFF={r[1]:.2f} S={r[2]:.2f} C={r[3]:.2f}")

if __name__ == "__main__":
    main()