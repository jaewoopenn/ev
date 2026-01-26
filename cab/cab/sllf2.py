import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx  # Max Flow 계산을 위해 추가

# --- 상수 설정 ---
MAX_RATE = 5            # EV당 최대 충전 속도 (kW)
# GRID_CAPACITY는 이제 상수가 아니라 계산된 값을 사용합니다.
PATH = '/users/jaewoo/data/ev/fluid/'
CSV_FILE_NAME = PATH + 'ev_jobs.csv'
SAVE_FILE_NAME = PATH + 'results.png'

def calculate_offline_optimal_capacity(jobs, max_rate_per_ev):
    """
    Max Flow와 이진 탐색을 사용하여 모든 Job을 처리할 수 있는
    최소한의 Grid Capacity(Offline Optimal)를 계산합니다.
    """
    print(">>> Offline Optimal Grid Capacity 계산 중 (Max Flow)...")
    
    # 시간 범위 산정
    min_time = min(job['arrival'] for job in jobs)
    max_time = max(job['departure'] for job in jobs)
    time_points = list(range(min_time, max_time + 1))
    total_required_energy = sum(job['required_energy'] for job in jobs)

    # 이진 탐색 범위 설정
    # 최소: 0, 최대: (차량 수 * 개별 최대 속도) - 넉넉하게 잡음
    low = 0.0
    high = len(jobs) * max_rate_per_ev
    optimal_capacity = high

    # Max Flow 그래프 생성 함수 (내부 함수)
    def is_feasible(capacity_limit):
        G = nx.DiGraph()
        
        # 노드 정의
        source = 'S'
        sink = 'T'
        
        # 1. Source -> Job 노드 (용량: 필요 에너지)
        for i, job in enumerate(jobs):
            job_node = f"J_{i}"
            G.add_edge(source, job_node, capacity=job['required_energy'])
            
            # 2. Job -> Time 노드 (용량: 시간당 최대 충전 속도)
            # 해당 Job이 머무르는 시간대에만 연결
            for t in range(job['arrival'], job['departure']):
                time_node = f"T_{t}"
                G.add_edge(job_node, time_node, capacity=max_rate_per_ev)
        
        # 3. Time 노드 -> Sink (용량: Grid Capacity Limit - 우리가 테스트하는 값)
        for t in time_points:
            time_node = f"T_{t}"
            G.add_edge(time_node, sink, capacity=capacity_limit)
            
        # Max Flow 계산
        flow_value = nx.maximum_flow_value(G, source, sink)
        
        # 흐른 유량이 총 필요 에너지와 같으면 Feasible
        return flow_value >= total_required_energy - 1e-5

    # 이진 탐색 실행 (정밀도 0.1kW 까지)
    for _ in range(20): 
        mid = (low + high) / 2
        if is_feasible(mid):
            optimal_capacity = mid
            high = mid # 더 줄여봄
        else:
            low = mid # 용량 부족함

    # 약간의 여유(Margin)를 두어 올림 처리 (부동소수점 오차 방지)
    return round(optimal_capacity, 2) + 0.01


def solve_sLLF_step(current_time, active_evs, P_limit, r_bar_global):
    """
    한 시점(t)에 대한 sLLF 배분 계산
    """
    if not active_evs:
        return [], 0.0

    mapped_evs = []
    
    for i, ev in enumerate(active_evs):
        r_bar = r_bar_global 
        d = ev['departure']
        e_rem = ev['remaining_energy']
        
        # Eq. (4) Laxity Calculation
        if r_bar > 0:
            l_i = (d - current_time) - (e_rem / r_bar)
        else:
            l_i = -9999

        mapped_evs.append({
            'original_obj': ev,
            'idx': i,
            'r_bar': r_bar,
            'e_rem': e_rem,
            'laxity': l_i
        })

    total_demand_instant = sum([min(m['r_bar'], m['e_rem']) for m in mapped_evs])
    target_power = min(P_limit, total_demand_instant)

    # Laxity 범위 계산
    l_vals = [m['laxity'] for m in mapped_evs]
    if not l_vals: return [], 0.0
    
    L_min = min(l_vals) - 5.0
    L_max = max(l_vals) + 5.0
    
    best_L = L_min

    # Binary Search for L (Threshold)
    for _ in range(30):
        L = (L_min + L_max) / 2
        current_sum = 0
        for m in mapped_evs:
            raw_rate = m['r_bar'] * (L - m['laxity'] + 1)
            clamped_rate = max(0, min(raw_rate, min(m['r_bar'], m['e_rem'])))
            current_sum += clamped_rate
        
        if current_sum > target_power:
            L_max = L
        else:
            L_min = L
            best_L = L 

    decisions = []
    total_load = 0.0
    
    for m in mapped_evs:
        raw_rate = m['r_bar'] * (best_L - m['laxity'] + 1)
        charge_amount = max(0, min(raw_rate, min(m['r_bar'], m['e_rem'])))
        
        if charge_amount > 1e-6:
            decisions.append({
                'ev_obj': m['original_obj'],
                'charge': charge_amount
            })
            total_load += charge_amount

    return decisions, total_load


def simulate_and_plot_modified_with_csv(file_path):
    # 1. 데이터 로드
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return

    # 초기 Job 리스트 생성
    raw_jobs = []
    for _, row in df.iterrows():
        raw_jobs.append({
            'id': int(row['ID']),
            'arrival': int(row['Arrival']),
            'departure': int(row['Departure']),
            'required_energy': float(row['Energy']),
            'remaining_energy': float(row['Energy']),
            'charged_history': [],
            'status': 'pending'
        })
    
    # [NEW] 2. Offline Optimal Capacity 계산
    calculated_capacity = calculate_offline_optimal_capacity(raw_jobs, MAX_RATE)
    print(f"\n>>> [Optimal Capacity Found] : {calculated_capacity:.2f} kW")
    print(f">>> 이 값을 사용하여 sLLF 시뮬레이션을 시작합니다.\n")
    
    # 시뮬레이션 변수 설정
    grid_capacity_limit = calculated_capacity 
    max_time_horizon = df['Departure'].max() + 2
    current_time = 0
    grid_usage_history = []  
    charge_log = []

    print("=== Modified Scheduling (Step-by-Step sLLF) 시뮬레이션 시작 ===")

    total_energy_needed = sum(job['required_energy'] for job in raw_jobs)
    total_charged_accum = 0

    # 3. 시간별 시뮬레이션 루프
    while current_time <= max_time_horizon:
        active_evs = [
            job for job in raw_jobs 
            if job['arrival'] <= current_time < job['departure'] and job['remaining_energy'] > 0.001
        ]
        
        # 계산된 grid_capacity_limit 사용
        decisions, step_load = solve_sLLF_step(current_time, active_evs, grid_capacity_limit, MAX_RATE)
        
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
        
        # 종료 조건
        if total_charged_accum >= total_energy_needed - 0.1 and current_time > df['Arrival'].max():
            print(f"모든 차량 충전 완료. 조기 종료 @ t={current_time}")
            break
             
        current_time += 1
        
    print(f"Total Charged: {total_charged_accum:.1f} / Target: {total_energy_needed:.1f}")
    print("=== 시뮬레이션 완료 ===")
    
    # 5. 그래프 그리기
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
    title_str = f'EV Charging Schedule (sLLF) @ Optimal Cap: {grid_capacity_limit:.2f}kW'
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
    plt.savefig(SAVE_FILE_NAME)
    print(f"Graph saved to {SAVE_FILE_NAME}")

# def save_evs_to_csv(evs, filename):
#     """(기록용) EV 리스트를 CSV 파일로 저장"""
#     try:
#         with open(filename, mode='w', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             writer.writerow(['ID', 'Arrival', 'Energy', 'Departure'])
#             for ev in evs:
#                 writer.writerow([ev.id, ev.arrival, ev.energy, ev.departure])
#         print(f"기록 저장 성공: '{filename}'")
#     except IOError as e:
#         print(f"오류: 파일을 저장하는 중 문제가 발생했습니다. {e}")

# 실행
simulate_and_plot_modified_with_csv(CSV_FILE_NAME)