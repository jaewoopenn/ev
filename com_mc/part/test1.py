import random
import matplotlib.pyplot as plt

# Fixed the bug where it incorrectly accepted 100% by removing the fast-path
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
    # Removed fast-path: if U_LO <= 1.0 and U_HI <= 1.0: return True (THIS WAS THE BUG!)
    x = compute_x_max(U_LC_A, U_LC_D, U_HC_H)
    if x is None: return False

    lo_util = (U_HC_L / x) + U_LC_A
    return lo_util <= 1.0

def is_schedulable_new(U_LC_A, U_HC_L, U_LC_D, U_HC_H, hc_tasks, lc_tasks):
    # Removed fast-path: if U_LO <= 1.0 and U_HI <= 1.0: return True
    x = compute_x_max(U_LC_A, U_LC_D, U_HC_H)
    if x is None: return False

    lo_sum = U_LC_A
    for (u_l, u_h) in hc_tasks:
        # Check for HI-only condition
        if u_l / x >= u_h:
            lo_sum += u_h
        else:
            lo_sum += u_l / x
            
    return lo_sum <= 1.0

# Generate random uniprocessor workload
def generate_random_uniprocessor_workload(target_util):
    hc_tasks = []
    lc_tasks = []
    U_LC_A = 0.0
    U_HC_L = 0.0
    U_LC_D = 0.0
    U_HC_H = 0.0
    
    while True:
        u_lo = random.uniform(0.02, 0.15)
        is_hc = random.random() < 0.5
        
        if is_hc:
            u_hi = random.uniform(u_lo, u_lo * 3.0)
            cur_lo = U_LC_A + U_HC_L + u_lo
            cur_hi = U_LC_D + U_HC_H + u_hi
        else:
            u_hi = random.uniform(0, u_lo * 0.5)
            cur_lo = U_LC_A + U_HC_L + u_lo
            cur_hi = U_LC_D + U_HC_H + u_hi
            
        if max(cur_lo, cur_hi) > target_util:
            break
            
        if is_hc:
            hc_tasks.append((u_lo, u_hi))
            U_HC_L += u_lo
            U_HC_H += u_hi
        else:
            lc_tasks.append((u_lo, u_hi))
            U_LC_A += u_lo
            U_LC_D += u_hi
            
    return U_LC_A, U_HC_L, U_LC_D, U_HC_H, hc_tasks, lc_tasks

def run_comparison():
    print("=" * 75)
    print("단일 프로세서 스케줄러빌리티 비교 (Original vs New)")
    print("=" * 75)
    print(f"{'Target Util':<14} | {'Original Accept':<16} | {'New Accept':<16} | {'New Only (Gap)':<12}")
    print("-" * 75)
    
    targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    orig_rates = []
    new_rates = []
    
    for target in targets:
        num_tests = 50000
        
        accept_orig = 0
        accept_new = 0
        gap_count = 0 
        
        for _ in range(num_tests):
            U_LC_A, U_HC_L, U_LC_D, U_HC_H, hc_tasks, lc_tasks = generate_random_uniprocessor_workload(target)
            
            res_orig = is_schedulable_original(U_LC_A, U_HC_L, U_LC_D, U_HC_H, hc_tasks, lc_tasks)
            res_new = is_schedulable_new(U_LC_A, U_HC_L, U_LC_D, U_HC_H, hc_tasks, lc_tasks)
            
            if res_orig: accept_orig += 1
            if res_new: accept_new += 1
                
            if res_new and not res_orig:
                gap_count += 1
                
        rate_orig = (accept_orig / num_tests) * 100
        rate_new = (accept_new / num_tests) * 100
        
        orig_rates.append(rate_orig)
        new_rates.append(rate_new)
        
        print(f"{target:<14.2f} | {accept_orig:<7} ({rate_orig:>5.2f}%) | {accept_new:<7} ({rate_new:>5.2f}%) | +{gap_count} 케이스")

    # 그래프 생성 및 시각화 로직
    plt.figure(figsize=(10, 6))
    plt.plot(targets, orig_rates, marker='o', linestyle='-', color='blue', label='Original (EDF-VD)')
    plt.plot(targets, new_rates, marker='s', linestyle='--', color='red', label='New (HI-only)')

    plt.title('Uniprocessor Schedulability', fontsize=14)
    plt.xlabel('Target Utilization Bound', fontsize=12)
    plt.ylabel('Acceptance Ratio %', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(targets)
    plt.ylim(0, 105)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()