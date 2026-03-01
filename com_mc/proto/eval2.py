import csv
import math
from collections import defaultdict
import os

DATA_SAVE_PATH = "/Users/jaewoo/data/com/data/"

def is_edf_schedulable(task_set):
    """일반 EDF 스케줄 가능성 (Pessimistic)"""
    u_lo_lo = sum(t['u_lo'] for t in task_set if t['crit'] == 'LO')
    u_hi_hi = sum(t['u_hi'] for t in task_set if t['crit'] == 'HI')
    return (u_lo_lo + u_hi_hi) <= 1.0

def is_edf_vd_schedulable(task_set):
    """EDF-VD 스케줄 가능성 (Baruah et al.)"""
    u_lo_lo = sum(t['u_lo'] for t in task_set if t['crit'] == 'LO')
    u_hi_lo = sum(t['u_lo'] for t in task_set if t['crit'] == 'HI')
    u_hi_hi = sum(t['u_hi'] for t in task_set if t['crit'] == 'HI')

    if u_lo_lo + u_hi_lo > 1.0 or u_hi_hi > 1.0: return False
    if u_lo_lo + u_hi_hi <= 1.0: return True
    if u_lo_lo >= 1.0: return False
        
    x = u_hi_lo / (1.0 - u_lo_lo)
    if (x * u_lo_lo) + u_hi_hi <= 1.0: return True
    return False

def is_amc_max_schedulable(task_set):
    """
    AMC-max 스케줄 가능성 분석
    우선순위: Rate Monotonic (RM) 적용 (주기가 짧을수록 높은 우선순위)
    """
    # 1. 주기(period) 기준으로 오름차순 정렬 (인덱스가 작을수록 고우선순위)
    tasks = sorted(task_set, key=lambda x: x['period'])
    
    # 2. LO-mode RTA (Response Time Analysis)
    for i, task in enumerate(tasks):
        r = task['c_lo']
        while True:
            # 부동소수점 오차 방지를 위해 round 적용
            interference = sum(math.ceil(round(r / tasks[j]['period'], 7)) * tasks[j]['c_lo'] for j in range(i))
            new_r = task['c_lo'] + interference  # ✨ 수정된 부분: 기준값을 항상 c_lo로 고정
            
            if new_r > task['period']:
                return False # LO 모드에서 데드라인 위반
            if new_r == r:
                break
            r = new_r  # ✨ 수정된 부분: 갱신된 값을 r에 할당
            
        # HI 모드 분석을 위해 LO 모드에서의 최악 응답시간 저장
        task['r_lo'] = r 

    # 3. HI-mode RTA (AMC-max) - HI 태스크만 검사
    for i, task in enumerate(tasks):
        if task['crit'] != 'HI':
            continue
            
        # 모드 전환 후보 시점(s) 집합 S 구성
        S = set([0])
        for j in range(i):
            if tasks[j]['crit'] == 'LO':
                t = 0
                while t < task['r_lo']:
                    S.add(t)
                    t += tasks[j]['period']
        
        # 각 후보 시점 s에 대해 스케줄 가능성 검사
        for s in S:
            # 시점 s까지 발생한 고우선순위 LO 태스크의 간섭량
            lo_interference = sum(math.ceil(round(s / tasks[j]['period'], 7)) * tasks[j]['c_lo'] for j in range(i) if tasks[j]['crit'] == 'LO')
            
            r = task['c_hi'] + lo_interference
            s_valid = True
            
            while True:
                hi_interference = sum(math.ceil(round(r / tasks[j]['period'], 7)) * tasks[j]['c_hi'] for j in range(i) if tasks[j]['crit'] == 'HI')
                # ✨ 수정된 부분: c_hi + lo_간섭량 + hi_간섭량의 합
                new_r = task['c_hi'] + lo_interference + hi_interference  
                
                if new_r > task['period']:
                    s_valid = False # 이 시점 s에서는 데드라인 위반
                    break
                if new_r == r:
                    break
                r = new_r  # ✨ 수정된 부분
                
            if not s_valid:
                return False # 단 하나의 s라도 데드라인을 놓치면 실패로 간주

    return True
def evaluate_and_save(input_filename="mc_task_sets.csv", output_filename="schedulability_results.csv"):
    input_path = os.path.join(DATA_SAVE_PATH, input_filename)
    output_path = os.path.join(DATA_SAVE_PATH, output_filename)
    
    task_sets = defaultdict(list)
    set_to_u = {}
    
    try:
        with open(input_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_set_id = int(row['task_set_id'])
                target_u = row['target_u']
                
                task_sets[task_set_id].append({
                    'crit': row['crit'],
                    'period': float(row['period']),
                    'u_lo': float(row['u_lo']),
                    'u_hi': float(row['u_hi']),
                    'c_lo': float(row['c_lo']),
                    'c_hi': float(row['c_hi'])
                })
                set_to_u[task_set_id] = target_u
    except FileNotFoundError:
        print(f"Error: {input_path} 파일을 찾을 수 없습니다.")
        return

    u_results = defaultdict(lambda: {'total': 0, 'edf': 0, 'edf_vd': 0, 'amc_max': 0})
    
    print("평가 중입니다. (AMC-max는 RTA 연산이 있어 시간이 조금 걸릴 수 있습니다...)")
    
    for task_set_id, task_set in task_sets.items():
        target_u = set_to_u[task_set_id]
        u_results[target_u]['total'] += 1
        
        if is_edf_schedulable(task_set): u_results[target_u]['edf'] += 1
        if is_edf_vd_schedulable(task_set): u_results[target_u]['edf_vd'] += 1
        if is_amc_max_schedulable(task_set): u_results[target_u]['amc_max'] += 1

    results = []
    print(f"\n{'Util':<6} | {'EDF Ratio':<10} | {'EDF-VD Ratio':<12} | {'AMC-max Ratio':<14}")
    print("-" * 50)
    
    for u_key in sorted(u_results.keys(), key=float):
        total = u_results[u_key]['total']
        
        edf_ratio = u_results[u_key]['edf'] / total if total > 0 else 0
        edf_vd_ratio = u_results[u_key]['edf_vd'] / total if total > 0 else 0
        amc_ratio = u_results[u_key]['amc_max'] / total if total > 0 else 0
        
        results.append({
            "Utilization": u_key, 
            "EDF_Acceptance_Ratio": f"{edf_ratio:.3f}",
            "EDF_VD_Acceptance_Ratio": f"{edf_vd_ratio:.3f}",
            "AMC_Max_Acceptance_Ratio": f"{amc_ratio:.3f}"
        })
        
        print(f"{u_key:<6} | {edf_ratio:<10.3f} | {edf_vd_ratio:<12.3f} | {amc_ratio:<14.3f}")

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Utilization", "EDF_Acceptance_Ratio", "EDF_VD_Acceptance_Ratio", "AMC_Max_Acceptance_Ratio"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n세 가지 알고리즘의 평가 결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    evaluate_and_save()