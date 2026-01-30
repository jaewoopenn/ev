'''
Created on 2026. 1. 6. 

@author: jaewoo
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ---------------------------------------------------------
# 1. Synthetic Stress Data Generation
# ---------------------------------------------------------
def generate_stress_data(num_tasks=200, time_horizon=100, max_rate=5.0, seed=42):
    np.random.seed(seed)
    # 랜덤 도착 (Poisson 유사)
    arrivals = np.random.randint(0, time_horizon - 20, size=num_tasks)
    arrivals.sort()
    
    jobs = []
    for i in range(num_tasks):
        arrival = arrivals[i]
        energy = np.random.uniform(5.0, 30.0) 
        
        # 데드라인을 '빡빡하게' 설정 (Min Time + Random Slack)
        min_duration = energy / max_rate
        slack = np.random.uniform(0, 10.0)
        departure = arrival + int(np.ceil(min_duration + slack))
        
        jobs.append({
            'ID': i+1, 'Arrival': arrival, 'Energy': energy, 'Departure': departure
        })
    return pd.DataFrame(jobs)

# ---------------------------------------------------------
# 2. Optimal Offline Capacity (C_opt) via LP
# ---------------------------------------------------------
def calculate_c_opt(df, max_rate):
    # (LP 코드는 이전과 동일하므로 핵심만 요약)
    # 목적: 모든 Demand를 만족하면서 피크 용량 C를 최소화
    min_time, max_time = df['Arrival'].min(), df['Departure'].max()
    time_steps = range(min_time, max_time)
    n_vars = 1 # C
    
    # 변수 매핑 및 LP 매트릭스 구성 생략 (위의 실행 코드 참조)
    # ... (LP Solving Logic) ...
    # 여기서는 실행된 결과값을 바로 사용한다고 가정하거나
    # 전체 코드가 필요하면 이전 단계의 함수를 그대로 사용합니다.
    return 45.79877  # 계산된 값 (예시)

# ---------------------------------------------------------
# 3. User's 2-Phase Algorithm (Refactored)
# ---------------------------------------------------------
def calculate_scheduling_priorities(current_time, active_evs, grid_capacity, max_rate):
    """
    2-Phase Scheduling Algorithm:
    1. Mandatory Allocation: Deadline 임박 차량에 필수 전력 우선 배정
    2. Minimax Allocation: 남은 전력을 공평하게 배분 (부하 분산)
    3. Greedy Allocation: 자투리 전력을 급한 차량부터 소진
    """
    if not active_evs:
        return [], 0.0

    # Constants
    TOLERANCE = 1e-6
    MAX_ITER = 30

    # ---------------------------------------------------------
    # 0. Data Preparation (Vectorization)
    # ---------------------------------------------------------
    ids = [ev['id'] for ev in active_evs]
    remains = np.array([ev['remaining_energy'] for ev in active_evs], dtype=float)
    departures = np.array([ev['departure'] for ev in active_evs], dtype=float)
    
    # 남은 시간(Duration) 계산 (최소 1.0초 보장)
    durations = np.maximum(1.0, departures - current_time)
    
    # 각 EV가 현재 턴에 물리적으로 받을 수 있는 최대량 (Energy or Rate cap)
    phys_limits = np.minimum(remains, max_rate)

    # ---------------------------------------------------------
    # Step 1: Mandatory Allocation (Zero Laxity)
    # ---------------------------------------------------------
    # 미래의 모든 턴을 풀로 충전해도 부족한 양은 '지금' 받아야 함
    max_future_charge = (durations - 1.0) * max_rate
    mandatory_needs = np.maximum(0.0, remains - max_future_charge)
    
    # 필수 할당량 확정 (물리적 한계 내에서)
    allocation = np.minimum(mandatory_needs, phys_limits)
    
    current_load = np.sum(allocation)
    remaining_grid = grid_capacity - current_load

    # ---------------------------------------------------------
    # Step 2: Minimax Optimization (Distribute Surplus)
    # ---------------------------------------------------------
    # 추가로 더 받을 수 있는 여력(Room)이 있고, 그리드도 남았을 때 수행
    residual_needs = remains - allocation
    residual_capacity = phys_limits - allocation # 현재 턴에 더 받을 수 있는 속도 여유
    
    if remaining_grid > TOLERANCE and np.sum(residual_needs) > TOLERANCE:
        
        # Binary Search for optimal 'stress level' (mid)
        low, high = -1000.0, 1000.0
        best_extra = np.zeros_like(remains)
        
        for _ in range(MAX_ITER):
            stress_level = (low + high) / 2.0
            
            # Minimax Metric: 잔여 필요량 - (스트레스 * 시간)
            target_charge = residual_needs - (stress_level * durations)
            
            # 물리적 한계(residual_capacity) 내에서 할당
            proposed_extra = np.clip(target_charge, 0, residual_capacity)
            
            if np.sum(proposed_extra) > remaining_grid:
                low = stress_level  # 너무 많이 줌 -> 스트레스를 높여서 배분량을 줄임
            else:
                high = stress_level
                best_extra = proposed_extra # 가능한 해 저장

        # 최적화 결과 반영
        allocation += best_extra
        remaining_grid -= np.sum(best_extra)

    # ---------------------------------------------------------
    # Step 3: Greedy Allocation (Fill the gaps)
    # ---------------------------------------------------------
    # 아주 미세한 자투리 전력이 남았을 경우, 가장 급한(Urgency) 차량부터 채움
    if remaining_grid > TOLERANCE:
        # Urgency = Energy Density (남은 에너지 / 남은 시간)
        urgency_scores = remains / durations
        sorted_indices = np.argsort(urgency_scores)[::-1] # 내림차순
        
        for idx in sorted_indices:
            if remaining_grid <= TOLERANCE:
                break
            
            # 이 차량이 더 받을 수 있는 공간 확인
            room_to_charge = phys_limits[idx] - allocation[idx]
            
            if room_to_charge > TOLERANCE:
                top_up = min(remaining_grid, room_to_charge)
                allocation[idx] += top_up
                remaining_grid -= top_up

    # ---------------------------------------------------------
    # 4. Result Formatting
    # ---------------------------------------------------------
    results = []
    total_assigned = 0.0
    
    for i, charge_amount in enumerate(allocation):
        if charge_amount > TOLERANCE:
            results.append({
                'id': ids[i],
                'charge': float(charge_amount),
                'ev_obj': active_evs[i]
            })
            total_assigned += charge_amount
            
    return results, total_assigned

# ---------------------------------------------------------
# 4. Simulation Wrapper (Strict Capacity Enforcement)
# ---------------------------------------------------------

def run_stress_test_sim_strict(df, capacity, max_rate):
    jobs = [{'id': r['ID'], 'arrival': r['Arrival'], 'departure': r['Departure'], 
             'remaining_energy': r['Energy']} for _, r in df.iterrows()]
    
    max_time = int(df['Departure'].max())
    missed_count = 0
    
    for t in range(int(df['Arrival'].min()), max_time):
        # 활성 태스크 필터링
        active_evs = [j for j in jobs if j['arrival'] <= t < j['departure'] and j['remaining_energy'] > 1e-6]
        
        # 알고리즘 실행
        decisions, total_requested = calculate_scheduling_priorities(t, active_evs, capacity, max_rate)
        
        # [중요] 물리적 용량 강제 (Physical Enforcement)
        # 알고리즘이 용량을 초과해 요청하면 강제로 깎음 (실제 그리드 동작)
        if total_requested > capacity + 1e-9:
            scale_factor = capacity / total_requested
            for d in decisions:
                d['charge'] *= scale_factor
        
        # 에너지 충전 적용
        decision_map = {d['id']: d['charge'] for d in decisions}
        for j in active_evs:
            j['remaining_energy'] -= decision_map.get(j['id'], 0.0)
            
    # 결과 집계
    for j in jobs:
        if j['remaining_energy'] > 1e-3: missed_count += 1
            
    return missed_count

# ---------------------------------------------------------
# 5. Execute Sweep & Plot
# ---------------------------------------------------------
MAX_RATE = 5.0
df_stress = generate_stress_data(num_tasks=200)

# LP로 구한 C_opt (실제 실행값 대입)
c_opt_val = 45.80 

alphas = np.linspace(0.5, 1.5, 21)
miss_results = []

print("Running Sweep...")
for alpha in alphas:
    cap = c_opt_val * alpha
    missed = run_stress_test_sim_strict(df_stress, cap, MAX_RATE)
    miss_results.append(missed)

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(alphas, miss_results, 'o-', color='darkblue', linewidth=2)
plt.axvline(x=1.0, color='red', linestyle='--', label='Offline Optimal Limit')
plt.xlabel(r'Augmentation Factor ($\alpha$)')
plt.ylabel('Deadline Miss Count')
plt.title('Breakdown Point Analysis (Stress Test N=200)')
plt.grid(True)
plt.legend()
plt.show()