import random
import matplotlib.pyplot as plt

# ============================================================
# 1. 스케줄러빌리티 수식 (버그 수정 반영됨)
# ============================================================
def compute_x_max(U_LC_A: float, U_LC_D: float, U_HC_H: float):
    denom = U_LC_A - U_LC_D
    if denom <= 0.0:
        if U_HC_H + U_LC_D <= 1.0: return 1.0
        return None
    numer = 1.0 - U_HC_H - U_LC_D
    if numer <= 0.0:
        return None
    x_max = numer / denom
    if x_max > 1.0:
        x_max = 1.0
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
            lo_sum += u_h # HI-only
        else:
            lo_sum += u_l / x
    return lo_sum <= 1.0


# ============================================================
# 2. 프로세서 (코어) 클래스
# ============================================================
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
    def U_LO(self) -> float:
        return self.U_LC_A + self.U_HC_L

    @property
    def U_HI(self) -> float:
        return self.U_LC_D + self.U_HC_H

    def try_add(self, task: dict) -> bool:
        if task["crit"] == "HC":
            new_U_LC_A = self.U_LC_A
            new_U_HC_L = self.U_HC_L + task["u_LO"]
            new_U_LC_D = self.U_LC_D
            new_U_HC_H = self.U_HC_H + task["u_HI"]
            new_hc = self.hc_tasks + [(task["u_LO"], task["u_HI"])]
            new_lc = self.lc_tasks
        else:
            new_U_LC_A = self.U_LC_A + task["u_LO"]
            new_U_HC_L = self.U_HC_L
            new_U_LC_D = self.U_LC_D + task["u_HI"]
            new_U_HC_H = self.U_HC_H
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
# 3. 파티셔닝 알고리즘 비교군
# ============================================================

def partition_ffd_original(tasks, m):
    # 1. 기존 방식 (FFD + Original Formula)
    sorted_tasks = sorted(tasks, key=lambda t: max(t["u_LO"], t["u_HI"]), reverse=True)
    procs = [Processor(is_schedulable_original) for _ in range(m)]
    for task in sorted_tasks:
        placed = False
        for p in procs:
            if p.try_add(task):
                p.add(task)
                placed = True
                break
        if not placed: return False
    return True

def partition_ffd_new(tasks, m):
    # 2. 실패했던 방식 (FFD + New Formula) - 여전히 쏠림 현상 발생
    sorted_tasks = sorted(tasks, key=lambda t: max(t["u_LO"], t["u_HI"]), reverse=True)
    procs = [Processor(is_schedulable_new) for _ in range(m)]
    for task in sorted_tasks:
        placed = False
        for p in procs:
            if p.try_add(task):
                p.add(task)
                placed = True
                break
        if not placed: return False
    return True

def partition_mb_new(tasks, m):
    # 3. Mode-Balanced Fit (MB-FFD + New Formula)
    sorted_tasks = sorted(tasks, key=lambda t: max(t["u_LO"], t["u_HI"]), reverse=True)
    procs = [Processor(is_schedulable_new) for _ in range(m)]
    
    for task in sorted_tasks:
        placed = False
        if task["crit"] == "HC":
            sorted_procs = sorted(procs, key=lambda p: p.U_HI - p.U_LO)
        else:
            sorted_procs = sorted(procs, key=lambda p: p.U_LO - p.U_HI)
            
        for p in sorted_procs:
            if p.try_add(task):
                p.add(task)
                placed = True
                break
        if not placed: return False
    return True

def partition_cu_udp_original(tasks, m):
    # 4. CU-UDP (Criticality-Unaware Utilization Difference Based Partitioning + Original Formula)
    # 논문에 따라 HC는 u_HI, LC는 u_LO를 기준으로 전체 태스크를 정렬합니다.
    sorted_tasks = sorted(tasks, key=lambda t: t["u_HI"] if t["crit"] == "HC" else t["u_LO"], reverse=True)
    
    # 여기서 기존 스케줄식인 is_schedulable_original 사용
    procs = [Processor(is_schedulable_original) for _ in range(m)]
    
    for task in sorted_tasks:
        placed = False
        
        if task["crit"] == "HC":
            # HC 태스크: U_HC_H - U_HC_L 차이가 가장 작은 프로세서부터 탐색 (Worst-fit based on util diff)
            sorted_procs = sorted(procs, key=lambda p: p.U_HC_H - p.U_HC_L)
        else:
            # LC 태스크: 순서대로 탐색 (First-fit)
            sorted_procs = procs
            
        for p in sorted_procs:
            if p.try_add(task):
                p.add(task)
                placed = True
                break
        if not placed: return False
    return True


# ============================================================
# 4. 멀티프로세서 워크로드 자동 생성 및 시뮬레이션
# ============================================================
def generate_multiprocessor_workload(m, target_util_per_core):
    target_util = m * target_util_per_core
    tasks = []
    U_LC_A, U_HC_L, U_LC_D, U_HC_H = 0.0, 0.0, 0.0, 0.0
    
    while True:
        u_lo = random.uniform(0.02, 0.15)
        is_hc = random.random() < 0.5
        
        if is_hc:
            u_hi = random.uniform(u_lo, u_lo * 3.0)
            cur_lo = U_LC_A + U_HC_L + u_lo
            cur_hi = U_LC_D + U_HC_H + u_hi
        else:
            # LC 태스크의 u_hi(디그레이드 후 utilization)를 0이 아닌 u_lo의 절반 이하의 랜덤 값으로 설정
            u_hi = random.uniform(0.001, u_lo / 2.0)
            cur_lo = U_LC_A + U_HC_L + u_lo
            cur_hi = U_LC_D + U_HC_H + u_hi
            
        if max(cur_lo, cur_hi) > target_util: break
            
        if is_hc:
            tasks.append({"crit": "HC", "u_LO": u_lo, "u_HI": u_hi})
            U_HC_L += u_lo
            U_HC_H += u_hi
        else:
            tasks.append({"crit": "LC", "u_LO": u_lo, "u_HI": u_hi})
            U_LC_A += u_lo
            U_LC_D += u_hi
            
    return tasks

def run_multiprocessor_comparison():
    m = 4 # 4 코어 시스템 가정
    targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    rates_orig = []
    rates_ffd_new = []
    rates_mb_new = []
    rates_cu_udp_orig = []
    
    print("=" * 110)
    print(f"Multiprocessor (m={m}) Partitioning Comparison")
    print("=" * 110)
    print(f"{'Target/Core':<12} | {'1. FFD(Orig)':<14} | {'2. FFD(New)':<14} | {'3. MB-FFD(New)':<16} | {'4. CU-UDP(Orig)':<16}")
    print("-" * 110)
    
    for target in targets:
        num_tests = 5000
        acc_orig = 0
        acc_ffd_new = 0
        acc_mb_new = 0
        acc_cu_udp_orig = 0
        
        for _ in range(num_tests):
            tasks = generate_multiprocessor_workload(m, target)
            
            if partition_ffd_original(tasks, m): acc_orig += 1
            if partition_ffd_new(tasks, m): acc_ffd_new += 1
            if partition_mb_new(tasks, m): acc_mb_new += 1
            if partition_cu_udp_original(tasks, m): acc_cu_udp_orig += 1
                
        r_orig = (acc_orig / num_tests) * 100
        r_ffd_new = (acc_ffd_new / num_tests) * 100
        r_mb_new = (acc_mb_new / num_tests) * 100
        r_cu_udp_orig = (acc_cu_udp_orig / num_tests) * 100
        
        rates_orig.append(r_orig)
        rates_ffd_new.append(r_ffd_new)
        rates_mb_new.append(r_mb_new)
        rates_cu_udp_orig.append(r_cu_udp_orig)
        
        print(f"{target:<12.2f} | {acc_orig:<5} ({r_orig:>5.2f}%) | {acc_ffd_new:<5} ({r_ffd_new:>5.2f}%) | {acc_mb_new:<5} ({r_mb_new:>5.2f}%)    | {acc_cu_udp_orig:<5} ({r_cu_udp_orig:>5.2f}%)")

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(targets, rates_orig, marker='o', linestyle='-', color='gray', label='1. FFD (Original Math)')
    plt.plot(targets, rates_ffd_new, marker='x', linestyle=':', color='blue', label='2. FFD (New Math)')
    plt.plot(targets, rates_mb_new, marker='s', linestyle='--', color='red', linewidth=2, label='3. MB-FFD (New Math)')
    plt.plot(targets, rates_cu_udp_orig, marker='^', linestyle='-', color='green', linewidth=2, label='4. CU-UDP (Original Math)')

    plt.title(f'Multiprocessor (m={m}) Acceptance Ratio Simulation', fontsize=14)
    plt.xlabel('Target Utilization / Core', fontsize=12)
    plt.ylabel('Acceptance Ratio (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(targets)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_multiprocessor_comparison()