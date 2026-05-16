import os
import json
import csv

# ============================================================
# 스케줄러빌리티 수식 및 프로세서 클래스
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

def is_schedulable_original(U_LC_A, U_HC_L, U_LC_D, U_HC_H, hc_tasks, lc_tasks):
    x = compute_x_max(U_LC_A, U_LC_D, U_HC_H)
    if x is None: return False
    lo_util = (U_HC_L / x) + U_LC_A
    return lo_util <= 1.0

def is_schedulable_new(U_LC_A, U_HC_L, U_LC_D, U_HC_H, hc_tasks, lc_tasks):
    x = compute_x_max(U_LC_A, U_LC_D, U_HC_H)
    if x is None: return False
    lo_sum = U_LC_A
    for (u_l, u_h) in hc_tasks:
        if u_l / x >= u_h:
            lo_sum += u_h 
        else:
            lo_sum += u_l / x
    return lo_sum <= 1.0

class Processor:
    def __init__(self, sched_func):
        self.U_LC_A = 0.0    
        self.U_HC_L = 0.0    
        self.U_LC_D = 0.0    
        self.U_HC_H = 0.0    
        self.hc_tasks = []   
        self.lc_tasks = []   
        self.tasks = []
        self._sched_func = sched_func

    @property
    def U_LO(self) -> float: return self.U_LC_A + self.U_HC_L

    @property
    def U_HI(self) -> float: return self.U_LC_D + self.U_HC_H

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

# ============================================================
# 파티셔닝 알고리즘
# ============================================================
def partition_ffd_original(tasks, m):
    sorted_tasks = sorted(tasks, key=lambda t: max(t["u_LO"], t["u_HI"]), reverse=True)
    procs = [Processor(is_schedulable_original) for _ in range(m)]
    for task in sorted_tasks:
        if not any(p.try_add(task) and not p.add(task) for p in procs): return False
    return True

def partition_ffd_new(tasks, m):
    sorted_tasks = sorted(tasks, key=lambda t: max(t["u_LO"], t["u_HI"]), reverse=True)
    procs = [Processor(is_schedulable_new) for _ in range(m)]
    for task in sorted_tasks:
        if not any(p.try_add(task) and not p.add(task) for p in procs): return False
    return True

def partition_mb_new(tasks, m):
    sorted_tasks = sorted(tasks, key=lambda t: max(t["u_LO"], t["u_HI"]), reverse=True)
    procs = [Processor(is_schedulable_new) for _ in range(m)]
    for task in sorted_tasks:
        sorted_procs = sorted(procs, key=lambda p: p.U_HI - p.U_LO if task["crit"] == "HC" else p.U_LO - p.U_HI)
        if not any(p.try_add(task) and not p.add(task) for p in sorted_procs): return False
    return True

def partition_cu_udp_original(tasks, m):
    sorted_tasks = sorted(tasks, key=lambda t: t["u_HI"] if t["crit"] == "HC" else t["u_LO"], reverse=True)
    procs = [Processor(is_schedulable_original) for _ in range(m)]
    for task in sorted_tasks:
        sorted_procs = sorted(procs, key=lambda p: p.U_HC_H - p.U_HC_L) if task["crit"] == "HC" else procs
        if not any(p.try_add(task) and not p.add(task) for p in sorted_procs): return False
    return True

# ============================================================
# 단일 데이터셋 평가 함수
# ============================================================
def evaluate_dataset(all_tasks, m):
    num_tests = len(all_tasks)
    acc_orig = sum(1 for tasks in all_tasks if partition_ffd_original(tasks, m))
    acc_ffd_new = sum(1 for tasks in all_tasks if partition_ffd_new(tasks, m))
    acc_mb_new = sum(1 for tasks in all_tasks if partition_mb_new(tasks, m))
    acc_cu_udp_orig = sum(1 for tasks in all_tasks if partition_cu_udp_original(tasks, m))
    
    r_orig = (acc_orig / num_tests) * 100
    r_ffd_new = (acc_ffd_new / num_tests) * 100
    r_mb_new = (acc_mb_new / num_tests) * 100
    r_cu_udp_orig = (acc_cu_udp_orig / num_tests) * 100
    
    return r_orig, r_ffd_new, r_mb_new, r_cu_udp_orig

# ============================================================
# 메인: 평가 및 파일 저장
# ============================================================
def main():
    m_values = [2, 4, 8]
    targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    p_h_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    data_dir = "/Users/jaewoo/data/com/data"
    result_dir = "/Users/jaewoo/data/com"
    os.makedirs(result_dir, exist_ok=True)
    
    # ============================================================
    # 1. 기존: Utilization variation (P_H = 0.5 고정)
    # ============================================================
    csv_file_path = os.path.join(result_dir, "simulation_results.csv")
    
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["m", "Target", "FFD_Orig_Rate", "FFD_New_Rate", "MB_New_Rate", "CU_UDP_Orig_Rate"])
        
        for m in m_values:
            print(f"\n>>> [Util variation] Evaluating for m={m}")
            for target in targets:
                file_path = os.path.join(data_dir, f"tasks_m_{m}_target_{target:.2f}.json")
                if not os.path.exists(file_path):
                    print(f"  Data missing: {file_path}")
                    continue
                    
                with open(file_path, 'r') as jf:
                    all_tasks = json.load(jf)
                
                r_orig, r_ffd_new, r_mb_new, r_cu_udp_orig = evaluate_dataset(all_tasks, m)
                writer.writerow([m, target, r_orig, r_ffd_new, r_mb_new, r_cu_udp_orig])
                print(f"  Target {target:.2f} done.")
    
    print(f"\nUtil variation saved to: {csv_file_path}")
    
    # ============================================================
    # 2. 추가: P_H variation (Target util = 0.80 고정)
    # ============================================================
    csv_ph_path = os.path.join(result_dir, "simulation_ph_results.csv")
    
    with open(csv_ph_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["m", "P_H", "FFD_Orig_Rate", "FFD_New_Rate", "MB_New_Rate", "CU_UDP_Orig_Rate"])
        
        for m in m_values:
            print(f"\n>>> [P_H variation] Evaluating for m={m}")
            for p_h in p_h_values:
                file_path = os.path.join(data_dir, f"tasks_m_{m}_ph_{p_h:.1f}.json")
                if not os.path.exists(file_path):
                    print(f"  Data missing: {file_path}")
                    continue
                    
                with open(file_path, 'r') as jf:
                    all_tasks = json.load(jf)
                
                r_orig, r_ffd_new, r_mb_new, r_cu_udp_orig = evaluate_dataset(all_tasks, m)
                writer.writerow([m, p_h, r_orig, r_ffd_new, r_mb_new, r_cu_udp_orig])
                print(f"  P_H {p_h:.1f} done.")
    
    print(f"\nP_H variation saved to: {csv_ph_path}")
    print("\nAll evaluations complete.")

if __name__ == "__main__":
    main()