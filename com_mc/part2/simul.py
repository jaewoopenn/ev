import os
import json
import csv
import random
import copy

# ============================================================
# 1. 스케줄러빌리티 수식 및 프로세서 클래스
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
        if not placed: return None
    return procs

# ============================================================
# 2. 동적 시뮬레이터 — Selective Recovery
# ============================================================
def run_simulation(base_tasks, procs, sim_ticks, allow_migration):
    runtime_tasks = copy.deepcopy(base_tasks)
    
    for rt_task in runtime_tasks:
        original_proc_id = next(p.id for p in procs if any(t["id"] == rt_task["id"] for t in p.tasks))
        proc = next(p for p in procs if p.id == original_proc_id)
        rt_task["home_proc"] = proc
        rt_task["current_proc"] = proc

    total_jobs_spawned = 0
    degraded_jobs_count = 0
    
    for tick in range(sim_ticks):
        # 1. Job Release
        for t in runtime_tasks:
            if tick % t["period"] == 0:
                total_jobs_spawned += 1
                new_job = {"task": t, "id": total_jobs_spawned, "deadline": tick + t["period"],
                           "rem_LO": t["c_LO"], "rem_HI": t["c_HI"], "started": False}
                
                if t["crit"] == "LC" and t["current_proc"].mode == "HI":
                    degraded_jobs_count += 1
                else:
                    t["current_proc"].ready_queue.append(new_job)

        # 2. 스케줄링
        for p in procs:
            if p.running_job:
                if (p.mode == "LO" and p.running_job["rem_LO"] <= 0) or \
                   (p.mode == "HI" and p.running_job["rem_HI"] <= 0):
                    p.running_job = None

            # Selective Recovery: idle → LO mode, try to recover tasks
            if p.running_job is None and len(p.ready_queue) == 0 and p.mode == "HI":
                p.mode = "LO"
                if allow_migration:
                    for t in runtime_tasks:
                        if t["home_proc"] == p and t["current_proc"] != p:
                            # Try to recover — only if p can accommodate
                            # if p.try_add(t):
                                t["current_proc"].remove(t)
                                p.add(t)
                                t["current_proc"] = p
                            # else: keep on current processor (permanent)

                            # # Try to recover — only if p can accommodate
                            # if p.try_add(t):
                            #     t["current_proc"].remove(t)
                            #     p.add(t)
                            #     t["current_proc"] = p
                            # # else: keep on current processor (permanent)

            if p.running_job is None and p.ready_queue:
                p.ready_queue.sort(key=lambda j: j["deadline"])
                p.running_job = p.ready_queue.pop(0)

                if not p.running_job["started"]:
                    p.running_job["started"] = True
                    if p.running_job["task"]["crit"] == "HC" and p.mode == "LO":
                        if random.random() < 0.20:
                            p.mode = "HI"
                            
                            # LC tasks sorted by util ascending
                            lc_tasks = sorted(
                                [t for t in runtime_tasks if t["crit"] == "LC" and t["current_proc"] == p],
                                key=lambda t: t["u_LO"]
                            )
                            
                            for lc_task in lc_tasks:
                                if allow_migration:
                                    migrated = False
                                    for target_p in procs:
                                        if target_p != p and target_p.mode == "LO" and target_p.try_add(lc_task):
                                            p.remove(lc_task)
                                            target_p.add(lc_task)
                                            lc_task["current_proc"] = target_p
                                            
                                            jobs_to_move = [j for j in p.ready_queue if j["task"]["id"] == lc_task["id"]]
                                            p.ready_queue = [j for j in p.ready_queue if j["task"]["id"] != lc_task["id"]]
                                            target_p.ready_queue.extend(jobs_to_move)
                                            migrated = True
                                            break
                                    if not migrated:
                                        jobs_to_degrade = [j for j in p.ready_queue if j["task"]["id"] == lc_task["id"]]
                                        p.ready_queue = [j for j in p.ready_queue if j["task"]["id"] != lc_task["id"]]
                                        degraded_jobs_count += len(jobs_to_degrade)
                                        if p.running_job and p.running_job["task"]["id"] == lc_task["id"]:
                                            degraded_jobs_count += 1
                                            p.running_job = None
                                else:
                                    jobs_to_degrade = [j for j in p.ready_queue if j["task"]["id"] == lc_task["id"]]
                                    p.ready_queue = [j for j in p.ready_queue if j["task"]["id"] != lc_task["id"]]
                                    degraded_jobs_count += len(jobs_to_degrade)
                                    if p.running_job and p.running_job["task"]["id"] == lc_task["id"]:
                                        degraded_jobs_count += 1
                                        p.running_job = None

            if p.running_job:
                if p.mode == "LO": p.running_job["rem_LO"] -= 1
                else: p.running_job["rem_HI"] -= 1

    return total_jobs_spawned, degraded_jobs_count

# ============================================================
# 3. 메인 실행부
# ============================================================
def main():
    m_values = [2, 4, 8]
    targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    data_dir = "/Users/jaewoo/data/com/data"
    result_dir = "/Users/jaewoo/data/com"
    csv_file_path = os.path.join(result_dir, "imc_simulation_results.csv")
    
    max_sim_tests = 50
    sim_ticks = 10000 
    periods = [20, 50, 100, 200]

    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["m", "Target", "Total_Jobs", "Degrade_ON_Ratio", "Degrade_OFF_Ratio"])
        
        for m in m_values:
            for target in targets:
                file_path = os.path.join(data_dir, f"stasks_m_{m}_target_{target:.2f}.json")
                if not os.path.exists(file_path): continue
                    
                with open(file_path, 'r') as jf:
                    all_tasks = json.load(jf)
                
                sim_count = 0
                acc_total_jobs_on = 0
                acc_degrade_jobs_on = 0
                acc_total_jobs_off = 0
                acc_degrade_jobs_off = 0
                
                print(f"Running simulation for m={m}, Target={target:.2f}...")
                
                for task_set in all_tasks:
                    if sim_count >= max_sim_tests: break
                    
                    for i, t in enumerate(task_set):
                        t["id"] = i
                        t["period"] = random.choice(periods)
                        t["c_LO"] = max(1, int(t["u_LO"] * t["period"]))
                        t["c_HI"] = max(1, int(t["u_HI"] * t["period"]))
                    
                    procs_init = partition_ffd_new(copy.deepcopy(task_set), m)
                    if procs_init is None: continue
                    
                    # Migration ON (selective recovery)
                    random.seed(42)
                    procs_on = partition_ffd_new(copy.deepcopy(task_set), m)
                    t_on, d_on = run_simulation(task_set, procs_on, sim_ticks, allow_migration=True)
                    
                    # Migration OFF
                    random.seed(42)
                    procs_off = partition_ffd_new(copy.deepcopy(task_set), m)
                    t_off, d_off = run_simulation(task_set, procs_off, sim_ticks, allow_migration=False)
                    
                    acc_total_jobs_on += t_on
                    acc_degrade_jobs_on += d_on
                    acc_total_jobs_off += t_off
                    acc_degrade_jobs_off += d_off
                    
                    sim_count += 1
                
                if acc_total_jobs_on > 0:
                    ratio_on = (acc_degrade_jobs_on / acc_total_jobs_on) * 100
                    ratio_off = (acc_degrade_jobs_off / acc_total_jobs_off) * 100
                    writer.writerow([m, target, acc_total_jobs_on, ratio_on, ratio_off])
                    print(f" -> Result: ON={ratio_on:.2f}%, OFF={ratio_off:.2f}%")
                else:
                    print(" -> No schedulable sets found.")

if __name__ == "__main__":
    main()