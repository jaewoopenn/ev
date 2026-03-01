'''
Created on 2026. 3. 1.

@author: jaewoo
'''
import random
import csv
import os

DATA_SAVE_PATH = "/Users/jaewoo/data/com/data/"

def uunifast(n, u_target):
    """UUnifast 알고리즘: n개의 태스크에 대한 총합이 u_target인 utilization 리스트 생성"""
    sum_u = u_target
    u_list = []
    for i in range(1, n):
        next_sum_u = sum_u * (random.random() ** (1.0 / (n - i)))
        u_list.append(sum_u - next_sum_u)
        sum_u = next_sum_u
    u_list.append(sum_u)
    return u_list

def generate_mc_task_sets(num_sets_per_u, num_tasks, u_start, u_end, u_step, filename="mc_task_sets.csv"):
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)
    file_path = os.path.join(DATA_SAVE_PATH, filename)
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target_u', 'task_set_id', 'task_id', 'crit', 'period', 'u_lo', 'u_hi', 'c_lo', 'c_hi'])
        
        task_set_id = 0
        current_u = u_start
        
        while current_u <= u_end + 0.0001:  # 부동소수점 오차 방지
            target_u_str = f"{current_u:.2f}"
            
            for _ in range(num_sets_per_u):
                task_set_id += 1
                
                # 1. 임시로 Utilization 1.0을 기준으로 태스크 생성 (비율만 확보)
                u_values = uunifast(num_tasks, 1.0)
                
                temp_task_set = []
                prelim_u_lo_sum = 0.0
                prelim_u_hi_sum = 0.0
                
                for task_id, u in enumerate(u_values, 1):
                    crit = 'HI' if random.random() < 0.5 else 'LO'
                    period = random.randint(10, 1000)
                    
                    if crit == 'LO':
                        prelim_u_lo = u
                        prelim_u_hi = 0.0
                    else:
                        prelim_u_hi = u
                        prelim_u_lo = prelim_u_hi * random.uniform(0.2, 0.8)
                        
                    temp_task_set.append({
                        'task_id': task_id, 'crit': crit, 'period': period,
                        'prelim_u_lo': prelim_u_lo, 'prelim_u_hi': prelim_u_hi
                    })
                    prelim_u_lo_sum += prelim_u_lo
                    prelim_u_hi_sum += prelim_u_hi
                
                # 2. 생성된 태스크 셋의 최대 Utilization 확인 (LO 모드 합 vs HI 모드 합)
                max_prelim_u = max(prelim_u_lo_sum, prelim_u_hi_sum)
                
                # 3. 목표 구간 (target_u - u_step, target_u] 내에서 무작위 목표치 설정
                desired_max_u = random.uniform(current_u - u_step + 0.0001, current_u)
                
                # 4. 스케일링 비율(Scale Factor)을 적용하여 제약조건 완벽 충족
                scale_factor = desired_max_u / max_prelim_u
                
                for t in temp_task_set:
                    final_u_lo = t['prelim_u_lo'] * scale_factor
                    final_u_hi = t['prelim_u_hi'] * scale_factor
                    final_c_lo = final_u_lo * t['period']
                    final_c_hi = final_u_hi * t['period']
                    
                    writer.writerow([
                        target_u_str, task_set_id, t['task_id'], t['crit'], t['period'], 
                        final_u_lo, final_u_hi, final_c_lo, final_c_hi
                    ])
                    
            current_u += u_step

    print(f"Task sets successfully generated and scaled. Saved to {file_path}")

if __name__ == "__main__":
    generate_mc_task_sets(num_sets_per_u=1000, num_tasks=20, u_start=0.50, u_end=0.96, u_step=0.04)