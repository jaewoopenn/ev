import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
base_dir = "/users/jaewoo/data/ev/fluid/data/"

# 1. 기본 시스템 파라미터 (파일 내용 기반)
P_CAPACITY = 100       # [cite: 2] Grid Capacity
R_MAX = 20             # [cite: 3] Maximum Charging Rate
TIME_STEP = 0.5       # 시뮬레이션 시간 단위

def simulate_charging(df, algo_type='Fluid-LLF'):
    """
    각 알고리즘별로 Job의 에너지 요구량을 충족하는지 시뮬레이션
    """
    jobs = df.copy()
    jobs['remaining_energy'] = jobs['energy_requirement']
    jobs['completed'] = False
    
    # 시간축 설정
    start_time = jobs['arrival_time'].min()
    end_time = jobs['deadline'].max() + 10
    
    for t in np.arange(start_time, end_time, TIME_STEP):
        # 현재 시점에 도착해 있고, 아직 완료되지 않은 작업들(Active Jobs)
        active_mask = (jobs['arrival_time'] <= t) & (jobs['deadline'] > t) & (jobs['remaining_energy'] > 0)
        active_jobs = jobs[active_mask].copy()
        
        if active_jobs.empty:
            continue
            
        # 알고리즘별 우선순위 정렬
        if algo_type == 'EDF':
            # 마감 기한(deadline)이 짧은 순서대로 정렬 
            active_jobs = active_jobs.sort_values(by='deadline')
        elif algo_type in ['LLF', 'Fluid-LLF']:
            # Laxity 계산: (남은 시간) - (필요한 최소 충전 시간) 
            active_jobs['laxity'] = (active_jobs['deadline'] - t) - (active_jobs['remaining_energy'] / R_MAX)
            active_jobs = active_jobs.sort_values(by='laxity')
            
        # 전력 할당 (Greedy Allocation) 
        allocated_power = 0
        for idx in active_jobs.index:
            if allocated_power >= P_CAPACITY:
                break
            
            # 해당 차량이 받을 수 있는 전력 계산 (R_max와 남은 용량 중 최소값) 
            can_take = min(R_MAX, P_CAPACITY - allocated_power)
            
            # 실제로 필요한 전력량만큼만 소모
            actual_power = min(can_take, jobs.loc[idx, 'remaining_energy'] / TIME_STEP)
            
            jobs.loc[idx, 'remaining_energy'] -= actual_power * TIME_STEP
            allocated_power += actual_power
            
            # 충전 완료 체크
            if jobs.loc[idx, 'remaining_energy'] <= 1e-5:
                jobs.loc[idx, 'completed'] = True

    # 성공적으로 마감기한 내 충전한 비율 반환
    return jobs['completed'].mean()


def run_simulation():
    scenarios = sorted(os.listdir(base_dir))
    final_results = {algo: [] for algo in ['EDF', 'LLF', 'Fluid-LLF']}
    
    for level in scenarios:
        scenario_path = os.path.join(base_dir, level)
        files = [f for f in os.listdir(scenario_path) if f.endswith('.csv')]
        
        temp_scores = {algo: [] for algo in final_results.keys()}
        
        # 각 시나리오당 30개 파일 샘플링 (정확도를 위해)
        for f in files[:30]:
            df_job = pd.read_csv(os.path.join(scenario_path, f))
            for algo in final_results.keys():
                # 이전 단계에서 정의한 simulate_charging 함수 사용
                score = simulate_charging(df_job, algo) 
                temp_scores[algo].append(score)
        
        for algo in final_results.keys():
            final_results[algo].append(np.mean(temp_scores[algo]))
        print("level "+level)
    # 그래프 그리기
    plt.figure(figsize=(12, 7))
    x_axis = [f"Step {i+1}" for i in range(10)]
    
    plt.plot(x_axis, final_results['EDF'], 's--', label='EDF (Earliest Deadline)', alpha=0.7)
    plt.plot(x_axis, final_results['LLF'], 'o-', label='LLF (Least Laxity)', alpha=0.7)
    plt.plot(x_axis, final_results['Fluid-LLF'], 'd-', label='Fluid-LLF (Proposed)', linewidth=2.5)

    plt.axvline(x=6, color='red', linestyle=':', label='System Saturation Point')
    plt.title('Performance Stress Test: Acceptance Ratio by Scenario Density', fontsize=14)
    plt.xlabel('Congestion Level (Step 1: Loose → Step 10: Extreme)', fontsize=12)
    plt.ylabel('Acceptance Ratio (Success Rate)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()

# 시뮬레이션 실행 (앞서 정의된 simulate_charging 함수가 선언되어 있어야 함)
run_simulation()