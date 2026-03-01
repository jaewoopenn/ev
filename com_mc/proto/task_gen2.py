import random
import csv
import os
DATA_SAVE_PATH = "/Users/jaewoo/data/com/data/"

def load_config(filename="config.txt"):
    """텍스트 파일에서 설정값을 읽어오는 함수"""
    config = {}
    try:
        with open(DATA_SAVE_PATH+filename, 'r', encoding='utf-8') as f:
            for line in f:
                # 주석 및 공백 제거
                line = line.split('#')[0].strip()
                if not line or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Error: {filename} 파일을 찾을 수 없습니다. 기본값을 사용하거나 파일을 생성해주세요.")
        exit(1)
        
    return config

def uunifast(n, u_target):
    """UUnifast 알고리즘"""
    sum_u = u_target
    u_list = []
    for i in range(1, n):
        next_sum_u = sum_u * (random.random() ** (1.0 / (n - i)))
        u_list.append(sum_u - next_sum_u)
        sum_u = next_sum_u
    u_list.append(sum_u)
    return u_list

def generate_mc_task_sets(config):
    # 설정값 형변환 (Casting)
    data_save_path = config.get("DATA_SAVE_PATH", "./")
    filename = config.get("OUTPUT_FILENAME", "mc_task_sets.csv")
    num_sets_per_u = int(config.get("NUM_SETS_PER_U", 1000))
    num_tasks = int(config.get("NUM_TASKS", 20))
    u_start = float(config.get("U_START", 0.60))
    u_end = float(config.get("U_END", 0.96))
    u_step = float(config.get("U_STEP", 0.04))
    period_min = int(config.get("PERIOD_MIN", 10))
    period_max = int(config.get("PERIOD_MAX", 1000))
    prob_hi = float(config.get("PROB_HI", 0.5))
    u_lo_ratio_min = float(config.get("U_LO_RATIO_MIN", 0.2))
    u_lo_ratio_max = float(config.get("U_LO_RATIO_MAX", 0.8))

    os.makedirs(data_save_path, exist_ok=True)
    file_path = os.path.join(data_save_path, filename)
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target_u', 'task_set_id', 'task_id', 'crit', 'period', 'u_lo', 'u_hi', 'c_lo', 'c_hi'])
        
        task_set_id = 0
        current_u = u_start
        
        while current_u <= u_end + 0.0001:
            target_u_str = f"{current_u:.2f}"
            
            for _ in range(num_sets_per_u):
                task_set_id += 1
                
                # 1. 비율 생성을 위해 U=1.0 기준으로 임시 태스크 셋 생성
                u_values = uunifast(num_tasks, 1.0)
                temp_task_set = []
                prelim_u_lo_sum = 0.0
                prelim_u_hi_sum = 0.0
                
                for task_id, u in enumerate(u_values, 1):
                    # config에서 읽어온 확률(prob_hi)을 적용
                    crit = 'HI' if random.random() < prob_hi else 'LO'
                    # config에서 읽어온 주기 범위를 적용
                    period = random.randint(period_min, period_max)
                    
                    if crit == 'LO':
                        prelim_u_lo = u
                        prelim_u_hi = 0.0
                    else:
                        prelim_u_hi = u
                        # config에서 읽어온 비율(u_lo_ratio_min, max)을 적용
                        prelim_u_lo = prelim_u_hi * random.uniform(u_lo_ratio_min, u_lo_ratio_max)
                        
                    temp_task_set.append({
                        'task_id': task_id, 'crit': crit, 'period': period,
                        'prelim_u_lo': prelim_u_lo, 'prelim_u_hi': prelim_u_hi
                    })
                    prelim_u_lo_sum += prelim_u_lo
                    prelim_u_hi_sum += prelim_u_hi
                
                # 2. 가장 높은 모드의 Utilization 합 계산
                max_prelim_u = max(prelim_u_lo_sum, prelim_u_hi_sum)
                
                # 3. 목표 구간 내 랜덤 타겟 U 설정 및 스케일링
                desired_max_u = random.uniform(current_u - u_step + 0.0001, current_u)
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

    print(f"[{filename}] Task sets successfully generated using config.txt settings.")

if __name__ == "__main__":
    # 실행 시 config.txt 파일을 자동으로 읽어와 생성
    configuration = load_config("config.txt")
    generate_mc_task_sets(configuration)