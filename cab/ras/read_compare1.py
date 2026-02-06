import pandas as pd
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. 설정 및 상수 (통합 설정)
# ---------------------------------------------------------
TIME_STEP = 0.1              # 시뮬레이션 시간 단위 (시간)
EPSILON = 1e-6               
TOTAL_STATION_POWER = 35.0   # 전체 충전소 용량 (kW)
MAX_EV_POWER = 6.6           # 개별 EV 최대 충전 속도 (kW)
MIN_CHARGING_RATE = 1.5      # [NEW] 최소 충전 속도 보장 제약 (kW)
# F_PATH='/users/jaewoo/data/acn/acn_data_1week.csv'
F_PATH='/users/jaewoo/data/acn/acn_data_caltech_20191001_20191031.csv'
S_PATH='/users/jaewoo/data/ev/cab/acn.png'

@dataclass
class EVRequest:
    ev_id: int
    arrival: float
    required_energy: float
    deadline: float
    remaining: float

    def __repr__(self):
        return f"EV{self.ev_id}(A={self.arrival:.2f}, E={self.required_energy:.2f}, D={self.deadline:.2f})"

# ---------------------------------------------------------
# 2. 데이터 로딩 (from read_compare2.py)
# ---------------------------------------------------------
def load_ev_data(filename):
    try:
        df_raw = pd.read_csv(filename)
        df_raw['connectionTime'] = pd.to_datetime(df_raw['connectionTime'], utc=True)
        df_raw['disconnectTime'] = pd.to_datetime(df_raw['disconnectTime'], utc=True)
        df_raw = df_raw.sort_values('connectionTime').reset_index(drop=True)
        
        start_time = df_raw['connectionTime'].min()
        df_raw['arrival_h'] = (df_raw['connectionTime'] - start_time).dt.total_seconds() / 3600.0
        df_raw['deadline_h'] = (df_raw['disconnectTime'] - start_time).dt.total_seconds() / 3600.0
        
        df_clean = df_raw[df_raw['deadline_h'] > df_raw['arrival_h']].copy()
        
        requests = []
        for idx, row in df_clean.iterrows():
            requests.append(EVRequest(
                ev_id=idx,
                arrival=row['arrival_h'],
                required_energy=row['kWhDelivered'],
                deadline=row['deadline_h'],
                remaining=row['kWhDelivered']
            ))
        return requests
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

# ---------------------------------------------------------
# 3. 핵심 알고리즘 (from fast_port_def2.py)
# ---------------------------------------------------------
def calculate_sllf_power(current_time, active_evs, grid_capacity, max_ev_power, min_ev_power, time_step):
    count = len(active_evs)
    if count == 0: return []

    current_laxities = []
    min_requirements = []
    max_acceptables = []

    for ev in active_evs:
        l_t = (ev.deadline - current_time) - (ev.remaining / max_ev_power)
        current_laxities.append(l_t)
        phys_limit = min(max_ev_power, ev.remaining / time_step)
        max_acceptables.append(phys_limit)
        min_requirements.append(min(min_ev_power, phys_limit))

    total_min_req = sum(min_requirements)
    if total_min_req > grid_capacity + EPSILON:
        scale = grid_capacity / total_min_req
        return [r * scale for r in min_requirements]

    remaining_capacity = grid_capacity - total_min_req

    def get_extra_power_for_target_L(target_L):
        total_extra = 0.0
        extras = []
        for i in range(count):
            req_p = (max_ev_power / time_step) * (target_L - current_laxities[i] + time_step)
            available_room = max_acceptables[i] - min_requirements[i]
            alloc_extra = max(0.0, min(req_p, available_room)) if available_room > EPSILON else 0.0
            extras.append(alloc_extra)
            total_extra += alloc_extra
        return total_extra, extras

    low_L, high_L = min(current_laxities) - 5.0, max(current_laxities) + 5.0
    best_extras = [0.0] * count
    for _ in range(20): 
        mid_L = (low_L + high_L) / 2.0
        p_sum, p_extras = get_extra_power_for_target_L(mid_L)
        if p_sum > remaining_capacity: high_L = mid_L
        else: low_L = mid_L; best_extras = p_extras
            
    return [min_requirements[i] + best_extras[i] for i in range(count)]

def calculate_optimal_ras_power(current_time, active_evs, grid_capacity, max_ev_power, min_ev_power, time_step):
    if not active_evs: return []
    
    # Backward Analysis for Deadline Must Load
    segments = [{"start": current_time, "end": current_time + time_step, "capacity": grid_capacity * time_step, "is_now": True}]
    deadlines = sorted(list(set(ev.deadline for ev in active_evs)))
    start_t = current_time + time_step
    for d in deadlines:
        if d > start_t:
            segments.append({"start": start_t, "end": d, "capacity": grid_capacity * (d - start_t), "is_now": False})
            start_t = d

    deadline_must_energy = {ev.ev_id: 0.0 for ev in active_evs}
    sorted_evs_backward = sorted(active_evs, key=lambda x: x.deadline, reverse=True)
    temp_segs = copy.deepcopy(segments)
    
    for ev in sorted_evs_backward:
        energy_needed = ev.remaining
        for seg in reversed(temp_segs):
            if energy_needed <= EPSILON: break
            if seg["start"] >= ev.deadline: continue
            
            fill = min(energy_needed, seg["capacity"], max_ev_power * (seg["end"] - seg["start"]))
            if fill > EPSILON:
                seg["capacity"] -= fill
                energy_needed -= fill
                if seg.get("is_now", False):
                    deadline_must_energy[ev.ev_id] += fill

    final_allocations = {}
    total_mandatory = 0.0
    mandatory_map = {}

    for ev in active_evs:
        p_deadline = deadline_must_energy[ev.ev_id] / time_step
        p_min_rate = min(min_ev_power, ev.remaining / time_step)
        must_p = min(max(p_deadline, p_min_rate), max_ev_power)
        mandatory_map[ev.ev_id] = must_p
        total_mandatory += must_p

    surplus = max(0.0, grid_capacity - total_mandatory)
    sorted_evs_edf = sorted(active_evs, key=lambda x: x.deadline)
    
    for ev in sorted_evs_edf:
        must_p = mandatory_map[ev.ev_id]
        room = max(0.0, min(max_ev_power, ev.remaining/time_step) - must_p)
        bonus = min(surplus, room)
        final_allocations[ev.ev_id] = must_p + bonus
        surplus -= bonus
        
    return [final_allocations.get(ev.ev_id, 0.0) for ev in active_evs]

# ---------------------------------------------------------
# 4. 시뮬레이션 엔진 (from fast_port2.py + read_compare2.py)
# ---------------------------------------------------------
def run_simulation(ev_set: List[EVRequest], algorithm: str) -> float:
    evs = copy.deepcopy(ev_set)
    total_evs_count = len(evs)
    
    active_evs = [] 
    waiting_evs = sorted(evs, key=lambda x: x.arrival)
    
    success_count = 0
    failure_count = 0
    current_time = 0.0
    
    # max_concurrent_evs: Admission Control Limit
    max_concurrent_evs = int(TOTAL_STATION_POWER // MIN_CHARGING_RATE)
    
    while success_count + failure_count < total_evs_count:
        # 1. 도착 처리 (Waiting -> Ready)
        while waiting_evs and waiting_evs[0].arrival <= current_time + EPSILON:
            active_evs.append(waiting_evs.pop(0))
            
        # 2. 실패 처리
        for i in range(len(active_evs) - 1, -1, -1):
            if current_time > active_evs[i].deadline + EPSILON:
                failure_count += 1
                active_evs.pop(i)
        
        # 3. Admission Control (Active set management)
        # fast_port2: 정원 초과 시 우선순위 낮은 차는 대기(충전 0)하거나 제외됨.
        # 여기서는 '충전 기회'를 얻는 active_charging_group을 선별합니다.
        
        charging_candidates = []
        if len(active_evs) <= max_concurrent_evs:
            charging_candidates = active_evs
        else:
            # Admission Policy: 우선순위 높은 차만 충전 포트 접속 허용
            # Admission Sort (fast_port2 uses LLF for admission usually)
            if algorithm == 'EDF':
                active_evs.sort(key=lambda x: (x.deadline, x.ev_id))
            elif algorithm == 'FCFS':
                active_evs.sort(key=lambda x: (x.arrival, x.ev_id))
            else: # LLF, sLLF, S-RAS
                active_evs.sort(key=lambda x: ((x.deadline - current_time) - (x.remaining / MAX_EV_POWER), x.ev_id))
            
            charging_candidates = active_evs[:max_concurrent_evs]

        # 4. 전력 할당
        allocated_powers = {ev.ev_id: 0.0 for ev in active_evs}
        
        if not charging_candidates:
             pass # No charging
        
        elif algorithm == 'sLLF':
            powers = calculate_sllf_power(current_time, charging_candidates, TOTAL_STATION_POWER, MAX_EV_POWER, MIN_CHARGING_RATE, TIME_STEP)
            for i, ev in enumerate(charging_candidates): allocated_powers[ev.ev_id] = powers[i]
            
        elif algorithm == 'S-RAS' or algorithm == 'NEW_ALGO':
            powers = calculate_optimal_ras_power(current_time, charging_candidates, TOTAL_STATION_POWER, MAX_EV_POWER, MIN_CHARGING_RATE, TIME_STEP)
            for i, ev in enumerate(charging_candidates): allocated_powers[ev.ev_id] = powers[i]
            
        else:
            # Baseline (EDF, LLF, FCFS) with Min Rate Constraint (from fast_port2 logic)
            used_power = 0.0
            
            # Step A: Min Rate Allocation
            for ev in charging_candidates:
                req = min(MIN_CHARGING_RATE, ev.remaining/TIME_STEP)
                allocated_powers[ev.ev_id] = req
                used_power += req
            
            # Step B: Surplus Greedy Allocation
            surplus = max(0.0, TOTAL_STATION_POWER - used_power)
            
            # 정렬 순서는 이미 Admission 단계에서 되어 있음 (charging_candidates)
            # 다만 Greedy 분배를 위해 다시 정렬이 필요한 경우 (예: Admission은 LLF지만 배분은 EDF인 경우 등)
            # fast_port2에서는 Admission 정렬과 배분 정렬을 일치시킴.
            
            for ev in charging_candidates:
                current_p = allocated_powers[ev.ev_id]
                max_p = min(MAX_EV_POWER, ev.remaining/TIME_STEP)
                room = max_p - current_p
                
                add = min(surplus, room)
                allocated_powers[ev.ev_id] += add
                surplus -= add

        # 5. 충전 업데이트
        for i in range(len(active_evs) - 1, -1, -1):
            ev = active_evs[i]
            p = allocated_powers.get(ev.ev_id, 0.0)
            charged = min(ev.remaining, p * TIME_STEP)
            ev.remaining -= charged
            
            if ev.remaining <= EPSILON:
                success_count += 1
                active_evs.pop(i)
                
        current_time += TIME_STEP
        if not active_evs and not waiting_evs: break
        if current_time > 10000: break # Safety break

    return (success_count / total_evs_count) * 100.0

# ---------------------------------------------------------
# 5. 실행 및 결과 그래프
# ---------------------------------------------------------
# 데이터 로드
filename = F_PATH
requests = load_ev_data(filename)

if not requests:
    print("데이터를 로드할 수 없습니다. 'acn_data_1week.csv' 파일 경로를 확인해주세요.")
else:
    print(f"Loaded {len(requests)} requests.")
    print(f"Simulation Condition: Cap={TOTAL_STATION_POWER}kW, MinRate={MIN_CHARGING_RATE}kW, AdmissionLim={int(TOTAL_STATION_POWER//MIN_CHARGING_RATE)} EVs")

    algorithms = ['EDF', 'LLF', 'sLLF', 'S-RAS']
    results = {}

    for algo in algorithms:
        score = run_simulation(requests, algo)
        results[algo] = score
        print(f"{algo}: {score:.2f}%")

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values(), 
                   color=['gray', 'blue', 'orange', 'green'], alpha=0.8)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1.0, 
                 f"{yval:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title(f'Algorithm Success Rate (w/ Admission Control & MinRate {MIN_CHARGING_RATE}kW)', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    # plt.show()
    plt.savefig(S_PATH)