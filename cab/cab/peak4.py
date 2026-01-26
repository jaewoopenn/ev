import csv
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np  # Required for the new algorithm

# --- 설정 및 상수 ---
PATH = '/users/jaewoo/data/ev/fluid/'
CSV_FILE_NAME = PATH + 'ev_jobs.csv'
SAVE_FILE_NAME = PATH + 'results.png'

NUM_EVS = 10     # 생성할 EV 수
MAX_RATE = 5     # EV당 최대 충전 속도 (kW)


class EV:
    def __init__(self, id1, arrival, energy, departure):
        self.id = id1
        self.arrival = arrival
        self.energy = energy
        self.departure = departure

def generate_random_evs(num_evs):
    evs = []
    print(f"--- 랜덤 EV {num_evs}대 생성 중 (Wide Distribution) ---")
    for i in range(num_evs):
        # [수정 1] 도착 시간을 0~20으로 넓게 퍼뜨림 (Online 특성 강화)
        a = random.randint(0, 20)
        a=0
        
        # [수정 2] 주차 시간을 짧은 것과 긴 것을 섞음
        duration = random.randint(2, 25)
        departure = a + duration
        
        max_feasible_energy = duration * MAX_RATE
        
        # [수정 3] 에너지 요구량을 Max에 가깝게 설정 (Tight Constraints)
        min_req = max_feasible_energy * 0.5 
        energy = random.uniform(min_req, max_feasible_energy)
        
        evs.append(EV(i+1, a, energy, departure))
    return evs


def calculate_offline_optimal_capacity(jobs, max_rate_per_ev):
    print(">>> Offline Optimal Grid Capacity 계산 중 (Max Flow)...")
    
    if not jobs:
        return 0.0

    min_time = min(job['arrival'] for job in jobs)
    max_time = max(job['departure'] for job in jobs)
    time_points = list(range(min_time, max_time + 1))
    total_required_energy = sum(job['required_energy'] for job in jobs)

    low = 0.0
    high = len(jobs) * max_rate_per_ev
    optimal_capacity = high

    def is_feasible(capacity_limit):
        G = nx.DiGraph()
        source = 'S'
        sink = 'T'
        
        for i, job in enumerate(jobs):
            job_node = f"J_{i}"
            G.add_edge(source, job_node, capacity=job['required_energy'])
            
            for t in range(job['arrival'], job['departure']):
                time_node = f"T_{t}"
                G.add_edge(job_node, time_node, capacity=max_rate_per_ev)
        
        for t in time_points:
            time_node = f"T_{t}"
            G.add_edge(time_node, sink, capacity=capacity_limit)
            
        flow_value = nx.maximum_flow_value(G, source, sink)
        return flow_value >= total_required_energy - 1e-5

    for _ in range(20): 
        mid = (low + high) / 2
        if is_feasible(mid):
            optimal_capacity = mid
            high = mid
        else:
            low = mid

    return round(optimal_capacity, 4) 

# --- New Scheduling Algorithm ---
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


def simulate_and_plot(ev_data):
    # 1. 데이터 변환
    raw_jobs = []
    for ev in ev_data:
        raw_jobs.append({
            'id': ev.id,
            'arrival': ev.arrival,
            'departure': ev.departure,
            'required_energy': ev.energy,
            'remaining_energy': ev.energy,
            'charged_history': [],
            'status': 'pending'
        })
    
    # 2. Offline Optimal Capacity 계산
    opt_capacity = calculate_offline_optimal_capacity(raw_jobs, MAX_RATE)
    print(f"\n>>> [Optimal Capacity Found] : {opt_capacity:.4f} kW")
    
    grid_capacity_limit = opt_capacity 
    
    # 종료 시간: 가장 늦은 출발 시간 + 여유분
    max_departure = max(job['departure'] for job in raw_jobs)
    max_time_horizon = max_departure + 2
    current_time = 0
    
    grid_usage_history = []  
    charge_log = []

    print("=== 2-Phase Priority Scheduling 시뮬레이션 시작 ===")

    total_energy_needed = sum(job['required_energy'] for job in raw_jobs)
    total_charged_accum = 0

    # 3. 시간별 시뮬레이션 루프
    while current_time <= max_time_horizon:
        active_evs = [
            job for job in raw_jobs 
            if job['arrival'] <= current_time < job['departure'] and job['remaining_energy'] > 0.001
        ]
        
        # --- NEW SCHEDULING CALL ---
        decisions, step_load = calculate_scheduling_priorities(current_time, active_evs, grid_capacity_limit, MAX_RATE)
        
        actual_step_load = 0
        for action in decisions:
            ev = action['ev_obj']
            requested_charge = action['charge']
            amount = min(requested_charge, ev['remaining_energy'])
            
            ev['remaining_energy'] -= amount
            ev['charged_history'].append((current_time, amount))
            
            actual_step_load += amount
            total_charged_accum += amount
            
            charge_log.append({
                'Time': current_time,
                'EV_ID': ev['id'],
                'Charged_Amount': amount,
                'Remaining_Requirement': max(0, ev['remaining_energy'])
            })
            
        wasted = max(0, grid_capacity_limit - actual_step_load)
        grid_usage_history.append((current_time, actual_step_load, wasted))
        
        # 모든 차량 충전 완료 시 조기 종료
        max_arrival = max(job['arrival'] for job in raw_jobs)
        if total_charged_accum >= total_energy_needed - 0.1 and current_time > max_arrival:
            print(f"모든 차량 충전 완료. 조기 종료 @ t={current_time}")
            break
             
        current_time += 1
        
    print(f"Total Charged: {total_charged_accum:.1f} / Target: {total_energy_needed:.1f}")
    print("=== 시뮬레이션 완료 ===")
    
    # 4. 그래프 그리기
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # 상단: 간트 차트
    sorted_jobs = sorted(raw_jobs, key=lambda x: x['departure'], reverse=True)
    ids = [j['id'] for j in sorted_jobs]
    y_positions = range(len(ids))
    
    for i, job in enumerate(sorted_jobs):
        arrival = job['arrival']
        duration = job['departure'] - job['arrival']
        
        window_color = 'lightgray'
        edge_color = 'grey'
        if job['remaining_energy'] > 0.01:
            window_color = '#ffcccc'
            edge_color = 'red'

        ax1.broken_barh([(arrival, duration)], (i - 0.3, 0.6), 
                       facecolors=window_color, edgecolors=edge_color, alpha=0.5)
        
        charge_segments = [(t, 1) for t, amt in job['charged_history']]
        if charge_segments:
            ax1.broken_barh(charge_segments, (i - 0.3, 0.6), 
                           facecolors='dodgerblue', edgecolors='white')
            for t, amt in job['charged_history']:
                if amt > 0.1: 
                    ax1.text(t + 0.5, i, f"{amt:.1f}", 
                            ha='center', va='center', color='white', fontsize=6, fontweight='bold')
        
        if job['remaining_energy'] > 0.01:
            ax1.text(job['departure'], i, f" Fail: {job['remaining_energy']:.1f}", 
                    va='center', color='red', fontweight='bold', fontsize=8)

    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([f"ID {i}" for i in ids])
    title_str = f'EV Charging Schedule (2-Phase Priority) @ Optimal Cap: {grid_capacity_limit:.2f}kW'
    ax1.set_title(title_str)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    ax1.legend(handles=[
        mpatches.Patch(color='lightgray', label='Window'),
        mpatches.Patch(color='dodgerblue', label='Charging'),
        mpatches.Patch(color='#ffcccc', label='Missed')
    ], loc='upper right')

    # 하단: Grid Usage
    times = [t for t, u, w in grid_usage_history]
    used = [u for t, u, w in grid_usage_history]
    wasted = [w for t, u, w in grid_usage_history]
    
    ax2.bar(times, used, width=1.0, label='Used', color='dodgerblue', edgecolor='black', alpha=0.8, align='edge')
    ax2.bar(times, wasted, bottom=used, width=1.0, label='Available', color='lightgreen', edgecolor='black', alpha=0.3, align='edge', hatch='//')
    
    ax2.set_xlabel('Time (Hours)')
    ax2.set_ylabel('Power (kW)')
    ax2.set_title(f'Grid Usage (Limit: {grid_capacity_limit:.2f} kW)')
    ax2.axhline(y=grid_capacity_limit, color='red', linestyle='--', linewidth=2, label='Limit')
    ax2.legend(loc='lower right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# --- [Part 3] 메인 실행부 ---
if __name__ == "__main__":
    # 1. EV 데이터 생성
    ev_list = generate_random_evs(NUM_EVS)
    
    # 2. 시뮬레이션 및 플로팅 실행
    simulate_and_plot(ev_list)