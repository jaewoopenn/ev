import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- 상수 설정 ---
MAX_RATE = 5          # EV당 최대 충전 속도 (kW)
GRID_CAPACITY = 28      # 전체 전력망 용량 (kW)
PATH='/users/jaewoo/data/ev/fluid/'
CSV_FILE_NAME = PATH+'ev_jobs.csv'
SAVE_FILE_NAME=PATH+'results.png'
RESULTS_FN=PATH+'results.csv'
# MAX_RATE = 5          # EV당 최대 충전 속도 (kW)
# GRID_CAPACITY = 10      # 전체 전력망 용량 (kW)

# TODO: 매 time마다 하지 말고, rel/deadline마다 하는걸로 수정하자 (천천히) 
# refactor 했음 
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

def simulate_and_plot_modified_with_csv(file_path, output_csv_name="charge_history.csv"):
    # 1. 데이터 로드
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
        return

    # 2. 시뮬레이션 초기화
    jobs = []
    for _, row in df.iterrows():
        jobs.append({
            'id': int(row['ID']),
            'arrival': int(row['Arrival']),
            'departure': int(row['Departure']),
            'required_energy': row['Energy'],
            'remaining_energy': row['Energy'],
            'charged_history': [],
            'status': 'pending'
        })
    
    max_time_horizon = df['Departure'].max()
    current_time = 0
    grid_usage_history = []  
    
    # [추가됨] 1. CSV 저장을 위한 리스트 초기화
    charge_log = []

    print("=== Modified Scheduling 시뮬레이션 시작 ===")

    # 3. 시간별 시뮬레이션 루프
    total = 0
    tot_amt = 0
    
    # 총 필요 에너지 계산
    # for job in jobs:
        # tot_amt += job['remaining_energy']
        
    while current_time <= max_time_horizon:
        active_evs = [job for job in jobs if job['arrival'] <= current_time < job['departure'] and job['remaining_energy'] > 0.001]
        
        # 분리된 로직 함수 호출 (외부 함수라고 가정)
        decisions, total_load = calculate_scheduling_priorities(current_time, active_evs, GRID_CAPACITY, MAX_RATE)
        total_load=0
        # 결과 반영
        for action in decisions:
            ev = action['ev_obj']
            amount = min(action['charge'],ev['remaining_energy'])
            tot_amt+=amount
            total_load+=amount
            ev['remaining_energy'] -= amount
            ev['charged_history'].append((current_time, amount))
            
            # [추가됨] 2. 충전 기록 수집
            # 시간, 차량ID, 충전량, 충전 후 남은 필요량 등을 기록
            charge_log.append({
                'Time': current_time,
                'EV_ID': ev['id'],
                'Charged_Amount': amount,
                'Remaining_Requirement': max(0, ev['remaining_energy']) # 음수 방지
            })
            
        # 낭비 전력 기록
        wasted = GRID_CAPACITY - total_load
        total += total_load
        grid_usage_history.append((current_time, total_load, wasted))
        
        # (로그 출력이 너무 많으면 주석 처리 가능)
        print(f"Time: {current_time}, Wasted: {wasted:.1f}")
        
        current_time += 1
        
    print(f"Total Charged: {total:.1f} / Target: {tot_amt:.1f}")
    print("=== 시뮬레이션 완료 ===")
    
    # [추가됨] 3. 수집된 기록을 DataFrame으로 변환 후 CSV 저장
    if charge_log:
        log_df = pd.DataFrame(charge_log)
        # 보기 좋게 정렬 (시간 -> ID 순)
        log_df = log_df.sort_values(by=['Time', 'EV_ID'])
        log_df.to_csv(output_csv_name, index=False)
        print(f"=== 충전 기록 저장 완료: {output_csv_name} ===")
    else:
        print("=== 충전 기록이 없습니다 (저장 건너뜀) ===")

    # 4. 그래프 그리기 (기존 코드 유지)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # --- [상단] EV 간트 차트 ---
    sorted_jobs = sorted(jobs, key=lambda x: x['departure'], reverse=True)

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

        # A. 대기 가능 시간 바
        ax1.broken_barh([(arrival, duration)], (i - 0.3, 0.6), 
                       facecolors=window_color, edgecolors=edge_color, alpha=0.5)
        
        # B. 실제 충전 시간 바
        charge_segments = [(t, 1) for t, amt in job['charged_history']]
        if charge_segments:
            ax1.broken_barh(charge_segments, (i - 0.3, 0.6), 
                           facecolors='dodgerblue', edgecolors='white')
            
            for t, amt in job['charged_history']:
                ax1.text(t + 0.5, i, f"{amt:.1f}", 
                        ha='center', va='center', color='white', fontsize=7, fontweight='bold')
        
        # C. 미스 텍스트 표시
        if job['remaining_energy'] > 0.01:
            ax1.text(job['departure'] + 0.2, i, f"Missed: {job['remaining_energy']:.2f}", 
                    va='center', color='red', fontweight='bold', fontsize=9)

    # Y축 설정
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([f"ID {i}" for i in ids])
    
    for tick_label, job in zip(ax1.get_yticklabels(), sorted_jobs):
        if job['remaining_energy'] > 0.01:
            tick_label.set_color('red')

    ax1.set_title('EV Charging Schedule ')
    ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    gray_patch = mpatches.Patch(color='lightgray', label='Available Window')
    blue_patch = mpatches.Patch(color='dodgerblue', label='Charging')
    red_patch = mpatches.Patch(color='#ffcccc', label='Missed Deadline')
    ax1.legend(handles=[gray_patch, blue_patch, red_patch], loc='upper right')

    # --- [하단] Grid Usage (Stacked Bar) ---
    times = [t for t, u, w in grid_usage_history]
    used = [u for t, u, w in grid_usage_history]
    wasted = [w for t, u, w in grid_usage_history]
    
    ax2.bar(times, used, width=1.0, label='Used Energy', color='dodgerblue', edgecolor='black', alpha=0.8, align='edge')
    ax2.bar(times, wasted, bottom=used, width=1.0, label='Available (Wasted)', color='lightgreen', edgecolor='black', alpha=0.4, align='edge', hatch='//')
    
    ax2.set_xlabel('Time (Hours)')
    ax2.set_ylabel('Energy (kWh)')
    ax2.set_title('Grid Energy Usage vs Available Capacity')
    ax2.set_ylim(0, GRID_CAPACITY + 5)
    ax2.axhline(y=GRID_CAPACITY, color='red', linestyle='--', linewidth=2, label='Grid Capacity Limit')
    
    for t, u, w in grid_usage_history:
        if w > 1: 
            ax2.text(t + 0.5, u + w/2, f"{w:.1f}", ha='center', va='center', fontsize=8, color='green')

    ax2.legend(loc='lower right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(SAVE_FILE_NAME)
    print(f"Graph saved to {SAVE_FILE_NAME}")

# 사용 예시:
# simulate_and_plot_modified_with_csv('ev_jobs.csv', 'results.csv')
# 실행
simulate_and_plot_modified_with_csv(CSV_FILE_NAME,RESULTS_FN)