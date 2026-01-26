import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- 상수 설정 ---
MAX_RATE = 5            # EV당 최대 충전 속도 (kW) -> r_bar
GRID_CAPACITY = 21      # 전체 전력망 용량 (kW) -> P_limit
PATH = '/users/jaewoo/data/ev/fluid/'
CSV_FILE_NAME = PATH + 'ev_jobs.csv'
SAVE_FILE_NAME = PATH + 'results.png'
RESULTS_FN = PATH + 'results.csv'

def solve_sLLF_step(current_time, active_evs, P_limit, r_bar_global):
    """
    한 시점(t)에 대한 sLLF 배분 계산
    """
    if not active_evs:
        return [], 0.0

    # 1. 데이터 매핑 및 Laxity 계산
    # 메인 루프의 키값(departure, remaining_energy)을 수식에 맞게 변환
    mapped_evs = []
    laxities = {}
    
    for i, ev in enumerate(active_evs):
        # r_bar는 개별 차량 제한이 없으면 글로벌 상수 사용
        r_bar = r_bar_global 
        d = ev['departure']
        e_rem = ev['remaining_energy']
        
        # Eq. (4) Laxity Calculation
        # Laxity = (Deadline - current_time) - (Remaining Energy / Max Rate)
        if r_bar > 0:
            l_i = (d - current_time) - (e_rem / r_bar)
        else:
            l_i = -9999 # Should not happen

        laxities[i] = l_i
        
        # 계산 편의를 위해 리스트에 저장
        mapped_evs.append({
            'original_obj': ev,
            'idx': i,
            'r_bar': r_bar,
            'e_rem': e_rem,
            'laxity': l_i
        })

    # 2. Threshold L 계산 (Binary Search)
    # 현재 필요한 총 전력량 vs 그리드 용량 비교
    total_demand_instant = sum([min(m['r_bar'], m['e_rem']) for m in mapped_evs])
    target_power = min(P_limit, total_demand_instant)

    # Laxity 기반 탐색 범위 설정
    l_vals = [m['laxity'] for m in mapped_evs]
    L_min = min(l_vals) - 5.0
    L_max = max(l_vals) + 5.0
    
    best_L = L_min

    # Binary Search for L
    for _ in range(30): # 30 iterations is usually enough for single step
        L = (L_min + L_max) / 2
        
        # Eq. (16) Aggregation: Calculate Sum of Rates based on L
        current_sum = 0
        for m in mapped_evs:
            # sLLF 할당 공식
            # rate = r_bar * (L - laxity + 1)
            raw_rate = m['r_bar'] * (L - m['laxity'] + 1)
            # 범위 제한: 0 <= rate <= min(r_bar, e_rem)
            clamped_rate = max(0, min(raw_rate, min(m['r_bar'], m['e_rem'])))
            current_sum += clamped_rate
        
        if current_sum > target_power:
            L_max = L # 할당량이 너무 많으므로 L을 줄여야 함
        else:
            L_min = L
            best_L = L # Feasible 하므로 저장하고 더 큰 L(더 여유로운 배분)을 탐색

    # 3. 최종 결정 생성
    decisions = []
    total_load = 0.0
    
    for m in mapped_evs:
        # 최적의 L로 개별 충전량 계산
        raw_rate = m['r_bar'] * (best_L - m['laxity'] + 1)
        charge_amount = max(0, min(raw_rate, min(m['r_bar'], m['e_rem'])))
        
        if charge_amount > 1e-6:
            decisions.append({
                'ev_obj': m['original_obj'],
                'charge': charge_amount
            })
            total_load += charge_amount

    return decisions, total_load


def simulate_and_plot_modified_with_csv(file_path, output_csv_name="charge_history.csv"):
    # 1. 데이터 로드
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        # 테스트를 위한 더미 데이터 생성 (파일 없을 시)
        # df = pd.DataFrame({
        #     'ID': [1, 2, 3],
        #     'Arrival': [0, 2, 5],
        #     'Departure': [10, 12, 15],
        #     'Energy': [20, 30, 10]
        # })
        return

    # 2. 시뮬레이션 초기화
    jobs = []
    for _, row in df.iterrows():
        jobs.append({
            'id': int(row['ID']),
            'arrival': int(row['Arrival']),
            'departure': int(row['Departure']),
            'required_energy': float(row['Energy']),
            'remaining_energy': float(row['Energy']),
            'charged_history': [],
            'status': 'pending'
        })
    
    max_time_horizon = df['Departure'].max() + 2 # 여유 시간 추가
    current_time = 0
    grid_usage_history = []  
    charge_log = []

    print("=== Modified Scheduling (Step-by-Step sLLF) 시뮬레이션 시작 ===")

    # 총 필요 에너지 계산 (Target 확인용)
    total_energy_needed = sum(job['required_energy'] for job in jobs)
    total_charged_accum = 0

    # 3. 시간별 시뮬레이션 루프
    while current_time <= max_time_horizon:
        # 현재 활성화된(도착했고, 떠나지 않았으며, 충전이 필요한) EV 필터링
        active_evs = [
            job for job in jobs 
            if job['arrival'] <= current_time < job['departure'] and job['remaining_energy'] > 0.001
        ]
        
        # [수정됨] Step-by-Step sLLF 함수 호출
        # solve_sLLF_step(현재시간, 활성차량리스트, 그리드용량, 차량당최대속도)
        decisions, step_load = solve_sLLF_step(current_time, active_evs, GRID_CAPACITY, MAX_RATE)
        
        # 결과 반영
        actual_step_load = 0
        for action in decisions:
            ev = action['ev_obj']
            requested_charge = action['charge']
            
            # 실제 잔여량보다 많이 충전할 수 없음 (Double Check)
            amount = min(requested_charge, ev['remaining_energy'])
            
            ev['remaining_energy'] -= amount
            ev['charged_history'].append((current_time, amount))
            
            actual_step_load += amount
            total_charged_accum += amount
            
            # CSV 기록
            charge_log.append({
                'Time': current_time,
                'EV_ID': ev['id'],
                'Charged_Amount': amount,
                'Remaining_Requirement': max(0, ev['remaining_energy'])
            })
            
        # 낭비 전력 기록 (Capacity - Actual Load)
        wasted = max(0, GRID_CAPACITY - actual_step_load)
        grid_usage_history.append((current_time, actual_step_load, wasted))
        
        # 진행상황 출력 (10틱 단위 혹은 필요시)
        # if current_time % 10 == 0:
        #     print(f"Time: {current_time}, Load: {actual_step_load:.2f}, Active: {len(active_evs)}")
        
        # 모든 차량 충전 완료 시 조기 종료 체크 (옵션)
        if total_charged_accum >= total_energy_needed - 0.1 and current_time > df['Arrival'].max():
             print(f"모든 차량 충전 완료. 조기 종료 @ t={current_time}")
             break
             
        current_time += 1
        
    print(f"Total Charged: {total_charged_accum:.1f} / Target: {total_energy_needed:.1f}")
    print("=== 시뮬레이션 완료 ===")
    
    # 3. CSV 저장
    if charge_log:
        log_df = pd.DataFrame(charge_log)
        log_df = log_df.sort_values(by=['Time', 'EV_ID'])
        log_df.to_csv(output_csv_name, index=False)
        print(f"=== 충전 기록 저장 완료: {output_csv_name} ===")
    else:
        print("=== 충전 기록이 없습니다 (저장 건너뜀) ===")

    # 4. 그래프 그리기
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # --- [상단] EV 간트 차트 ---
    sorted_jobs = sorted(jobs, key=lambda x: x['departure'], reverse=True)
    ids = [j['id'] for j in sorted_jobs]
    y_positions = range(len(ids))
    
    for i, job in enumerate(sorted_jobs):
        arrival = job['arrival']
        duration = job['departure'] - job['arrival']
        
        window_color = 'lightgray'
        edge_color = 'grey'
        if job['remaining_energy'] > 0.01: # 미완료 차량 강조
             window_color = '#ffcccc'
             edge_color = 'red'

        # Window Bar
        ax1.broken_barh([(arrival, duration)], (i - 0.3, 0.6), 
                       facecolors=window_color, edgecolors=edge_color, alpha=0.5)
        
        # Charging Segments
        charge_segments = [(t, 1) for t, amt in job['charged_history']] # 너비는 1시간 고정
        if charge_segments:
            # 막대 그리기
            ax1.broken_barh(charge_segments, (i - 0.3, 0.6), 
                           facecolors='dodgerblue', edgecolors='white')
            
            # 텍스트는 너무 많으면 겹치므로 조건부 출력 (예: 0.1 이상만)
            for t, amt in job['charged_history']:
                if amt > 0.1: 
                    ax1.text(t + 0.5, i, f"{amt:.1f}", 
                            ha='center', va='center', color='white', fontsize=6, fontweight='bold')
        
        # Fail 텍스트
        if job['remaining_energy'] > 0.01:
            ax1.text(job['departure'], i, f" Fail: {job['remaining_energy']:.1f}", 
                    va='center', color='red', fontweight='bold', fontsize=8)

    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([f"ID {i}" for i in ids])
    ax1.set_title('EV Charging Schedule (sLLF)')
    ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # Legend
    gray_patch = mpatches.Patch(color='lightgray', label='Window')
    blue_patch = mpatches.Patch(color='dodgerblue', label='Charging')
    red_patch = mpatches.Patch(color='#ffcccc', label='Missed')
    ax1.legend(handles=[gray_patch, blue_patch, red_patch], loc='upper right')

    # --- [하단] Grid Usage ---
    times = [t for t, u, w in grid_usage_history]
    used = [u for t, u, w in grid_usage_history]
    wasted = [w for t, u, w in grid_usage_history]
    
    ax2.bar(times, used, width=1.0, label='Used', color='dodgerblue', edgecolor='black', alpha=0.8, align='edge')
    # Wasted는 Stacked Bar로 표현
    ax2.bar(times, wasted, bottom=used, width=1.0, label='Available', color='lightgreen', edgecolor='black', alpha=0.3, align='edge', hatch='//')
    
    ax2.set_xlabel('Time (Hours)')
    ax2.set_ylabel('Power (kW)')
    ax2.set_title('Grid Usage')
    ax2.axhline(y=GRID_CAPACITY, color='red', linestyle='--', linewidth=2, label='Limit')
    ax2.legend(loc='lower right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(SAVE_FILE_NAME)
    print(f"Graph saved to {SAVE_FILE_NAME}")

# 실행
simulate_and_plot_modified_with_csv(CSV_FILE_NAME, RESULTS_FN)