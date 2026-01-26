import csv
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

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
        # a=0
        
        # [수정 2] 주차 시간을 짧은 것과 긴 것을 섞음
        # 짧고 급한 차와, 길고 여유로운 차가 섞여야 sLLF가 실수할 확률이 높음
        duration = random.randint(2, 25)
        departure = a + duration
        
        max_feasible_energy = duration * MAX_RATE
        
        # [수정 3] 에너지 요구량을 Max에 가깝게 설정하여 'Laxity'를 0에 가깝게 만듦 (빡빡한 조건)
        # 최소 50% ~ 최대 100% 요구
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
    # return round(optimal_capacity, 2) + 0.01

def solve_sLLF_step(current_time, active_evs, P_limit, r_bar_global):
    """한 시점(t)에 대한 sLLF 배분 계산"""
    if not active_evs:
        return [], 0.0

    mapped_evs = []
    
    for i, ev in enumerate(active_evs):
        r_bar = r_bar_global 
        d = ev['departure']
        e_rem = ev['remaining_energy']
        
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

    l_vals = [m['laxity'] for m in mapped_evs]
    if not l_vals: return [], 0.0
    
    L_min = min(l_vals) - 5.0
    L_max = max(l_vals) + 5.0
    best_L = L_min

    # Binary Search for Threshold L
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
    
    # [중요 검증 포인트]
    # Offline Optimal은 이론상 '완벽한 미래 정보'를 가진 경우의 최소치입니다.
    # sLLF(Online)는 미래를 모르고, Laxity 기반으로 배분하므로 
    # 이론상 Offline Optimal Capacity와 동일한 Cap을 주면 실패하는 경우가 반드시 생겨야 합니다.
    # 하지만 부동소수점 오차로 성공할 수 있으므로, 아주 조금만 Tight하게 잡거나 그대로 둡니다.
    
    grid_capacity_limit = opt_capacity 
    # grid_capacity_limit = opt_capacity * 0.99  # 만약 확실한 실패를 보고 싶다면 주석 해제

    
    # 종료 시간: 가장 늦은 출발 시간 + 여유분
    max_departure = max(job['departure'] for job in raw_jobs)
    max_time_horizon = max_departure + 2
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
        
        # 모든 차량 충전 완료 시 조기 종료
        # (마지막 차량의 도착 시간은 지났어야 함)
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
    plt.show()
    # plt.savefig(SAVE_FILE_NAME)
    # print(f"Graph saved to {SAVE_FILE_NAME}")

# --- [Part 3] 메인 실행부 ---
if __name__ == "__main__":
    # 1. EV 데이터 생성
    ev_list = generate_random_evs(NUM_EVS)
    
    # 2. 기록용 CSV 저장 (선택 사항)
    # save_evs_to_csv(ev_list, CSV_FILE_NAME)
    
    # 3. 시뮬레이션 및 플로팅 실행 (파일 읽기 없이 바로 전달)
    simulate_and_plot(ev_list)