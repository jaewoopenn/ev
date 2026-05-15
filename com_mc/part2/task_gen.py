import os
import json
import random
import math

# ============================================================
# 멀티프로세서 워크로드 자동 생성
# ============================================================
def generate_multiprocessor_workload(m, target_util_per_core):
    target_util = m * target_util_per_core
    
    while True:
        n = random.randint(4 * m, 6 * m)
        tasks = []
        u_lo_sum = 0.0
        u_hi_sum = 0.0
        
        for _ in range(n):
            is_hc = random.random() < 0.5
            u_base = random.uniform(0.05, 0.3) 
            
            if is_hc:
                u_hi_raw = u_base
                u_lo_raw = u_hi_raw / random.uniform(1.0, 3.0)
            else:
                u_lo_raw = u_base
                u_hi_raw = random.uniform(0.001, u_lo_raw / 2.0)
                
            # [10, 500] 구간에서 로그 균등 분포(log-uniform distribution)로 주기 생성
            period =round(math.exp(random.uniform(math.log(10), math.log(500))))
                
            tasks.append({
                "crit": "HC" if is_hc else "LC", 
                "u_LO": u_lo_raw, 
                "u_HI": u_hi_raw,
                "period": period
            })
            u_lo_sum += u_lo_raw
            u_hi_sum += u_hi_raw
        
        sys_max = max(u_lo_sum, u_hi_sum)
        scale_factor = target_util / sys_max
        
        valid = True
        for t in tasks:
            t["u_LO"] *= scale_factor
            t["u_HI"] *= scale_factor
            
            if t["u_LO"] > 0.85 or t["u_HI"] > 0.85:
                valid = False
                break
                
        if valid:
            return tasks

def main():
    m_values = [2, 4, 8]
    targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    num_tests = 5000
    save_dir = "/Users/jaewoo/data/com/data"
    
    os.makedirs(save_dir, exist_ok=True)
    
    for m in m_values:
        print(f"\n>>> Task generation started for m={m} (tests={num_tests} per target)")
        for target in targets:
            tasks_for_target = []
            for _ in range(num_tests):
                tasks_for_target.append(generate_multiprocessor_workload(m, target))
                
            # 파일명에 m값 명시
            file_path = os.path.join(save_dir, f"tasks_m_{m}_target_{target:.2f}.json")
            with open(file_path, 'w') as f:
                json.dump(tasks_for_target, f)
                
            print(f"Saved: {file_path}")
            
    print("\nAll generation complete.")

if __name__ == "__main__":
    main()