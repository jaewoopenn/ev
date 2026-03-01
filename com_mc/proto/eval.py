import csv
from collections import defaultdict
import os

DATA_SAVE_PATH = "/Users/jaewoo/data/com/data/"

def is_edf_schedulable(task_set):
    """
    일반 EDF 스케줄 가능성 분석
    모드 전환 없이 모든 태스크가 최대 실행 시간을 가질 때를 가정 (Pessimistic Bound)
    """
    u_lo_lo = sum(t['u_lo'] for t in task_set if t['crit'] == 'LO')
    u_hi_hi = sum(t['u_hi'] for t in task_set if t['crit'] == 'HI')
    
    return (u_lo_lo + u_hi_hi) <= 1.0

def is_edf_vd_schedulable(task_set):
    """
    EDF-VD 스케줄 가능성 분석 (Baruah et al.)
    """
    u_lo_lo = sum(t['u_lo'] for t in task_set if t['crit'] == 'LO')
    u_hi_lo = sum(t['u_lo'] for t in task_set if t['crit'] == 'HI')
    u_hi_hi = sum(t['u_hi'] for t in task_set if t['crit'] == 'HI')

    # 1. 시스템의 기본/최대 Utilization이 1을 넘으면 절대 불가능
    if u_lo_lo + u_hi_lo > 1.0 or u_hi_hi > 1.0:
        return False
        
    # 2. 일반 EDF로도 스케줄 가능하면 당연히 EDF-VD로도 가능
    if u_lo_lo + u_hi_hi <= 1.0:
        return True
        
    # 3. EDF-VD 가상 데드라인 조건 검사
    if u_lo_lo >= 1.0:
        return False
        
    x = u_hi_lo / (1.0 - u_lo_lo)
    
    if (x * u_lo_lo) + u_hi_hi <= 1.0:
        return True

    return False

def evaluate_and_save(input_filename="mc_task_sets.csv", output_filename="schedulability_results.csv"):
    input_path = os.path.join(DATA_SAVE_PATH, input_filename)
    output_path = os.path.join(DATA_SAVE_PATH, output_filename)
    
    # 데이터를 담을 딕셔너리 구조
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
                    'u_lo': float(row['u_lo']),
                    'u_hi': float(row['u_hi'])
                })
                set_to_u[task_set_id] = target_u
    except FileNotFoundError:
        print(f"Error: {input_path} 파일을 찾을 수 없습니다. task_gen.py를 먼저 실행해주세요.")
        return

    # Utilization별 집계 딕셔너리
    u_results = defaultdict(lambda: {'total': 0, 'edf_success': 0, 'edf_vd_success': 0})
    
    for task_set_id, task_set in task_sets.items():
        target_u = set_to_u[task_set_id]
        u_results[target_u]['total'] += 1
        
        # 일반 EDF 검사
        if is_edf_schedulable(task_set):
            u_results[target_u]['edf_success'] += 1
            
        # EDF-VD 검사
        if is_edf_vd_schedulable(task_set):
            u_results[target_u]['edf_vd_success'] += 1

    # 최종 결과를 CSV로 작성
    results = []
    print(f"{'Utilization':<12} | {'EDF Ratio':<10} | {'EDF-VD Ratio':<12}")
    print("-" * 40)
    
    for u_key in sorted(u_results.keys(), key=float):
        total = u_results[u_key]['total']
        
        edf_ratio = u_results[u_key]['edf_success'] / total if total > 0 else 0
        edf_vd_ratio = u_results[u_key]['edf_vd_success'] / total if total > 0 else 0
        
        results.append({
            "Utilization": u_key, 
            "EDF_Acceptance_Ratio": f"{edf_ratio:.3f}",
            "EDF_VD_Acceptance_Ratio": f"{edf_vd_ratio:.3f}"
        })
        
        print(f"{u_key:<12} | {edf_ratio:<10.3f} | {edf_vd_ratio:<12.3f}")

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Utilization", "EDF_Acceptance_Ratio", "EDF_VD_Acceptance_Ratio"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n평가 결과가 {output_path}에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    evaluate_and_save()