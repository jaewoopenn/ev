import pandas as pd
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
TIME_STEP = 0.1              
EPSILON = 1e-6               
TOTAL_STATION_POWER = 37.0   
MAX_EV_POWER = 6.6           

# [설정] 0이면 sLLF 포함, 0보다 크면 sLLF 제외 및 MinRate 제약 적용
MIN_CHARGING_RATE = 1.5      
# MIN_CHARGING_RATE = 0

F_PATH='/users/jaewoo/data/acn/acn_data_caltech_20191001_20191031.csv'
S_PATH='/users/jaewoo/data/ev/cab/acn_result.png'

@dataclass
class EVRequest:
    ev_id: int
    arrival: float
    required_energy: float
    deadline: float
    remaining: float

    def __repr__(self):
        return f"EV{self.ev_id}(Rem={self.remaining:.2f}, D={self.deadline:.2f})"

# ---------------------------------------------------------
# 2. 데이터 로딩
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
# 3. 핵심 알고리즘
# ---------------------------------------------------------

def calculate_sllf_power(current_time, active_evs, grid_capacity, max_ev_power, time_step):
    """
    [sLLF] Min Rate가 0일 때만 작동 (Laxity Balancing)
    """
    count = len(active_evs)
    if count == 0: return []

    current_laxities = []
    max_acceptables = []

    for ev in active_evs:
        l_t = (ev.deadline - current_time) - (ev.remaining / max_ev_power)
        current_laxities.append(l_t)
        phys_limit = min(max_ev_power, ev.remaining / time_step)
        max_acceptables.append(phys_limit)

    def get_power_sum(target_L):
        total_p = 0.0
        allocs = []
        for i in range(count):
            req_p = (max_ev_power / time_step) * (target_L - current_laxities[i] + time_step)
            req_p = max(0.0, min(req_p, max_acceptables[i]))
            allocs.append(req_p)
            total_p += req_p
        return total_p, allocs

    min_lax = min(current_laxities)
    max_lax = max(current_laxities)
    low_L, high_L = min_lax - 10.0, max_lax + 10.0
    best_allocs = [0.0] * count
    
    for _ in range(25):
        mid_L = (low_L + high_L) / 2.0
        p_sum, p_allocs = get_power_sum(mid_L)
        if p_sum > grid_capacity: high_L = mid_L 
        else: low_L = mid_L; best_allocs = p_allocs
            
    return best_allocs

def calculate_sras_power(current_time, ready_evs, grid_capacity, max_ev_power, min_ev_power, time_step):
    """
    [S-RAS Implementation]
    Fixed: I_min=0일 때 Tier 2에서 차량들이 활성화되지 않는 버그 수정
    """
    if not ready_evs: return []

    # -----------------------------------------------------
    # 1. Backward Analysis (M_i 계산)
    # -----------------------------------------------------
    segments = []
    first_seg_end = current_time + time_step
    segments.append({"start": current_time, "end": first_seg_end, "capacity": grid_capacity * time_step, "index": 0})
    
    deadlines = sorted(list(set(ev.deadline for ev in ready_evs)))
    deadlines = [d for d in deadlines if d > first_seg_end]
    start_t = first_seg_end
    for d in deadlines:
        segments.append({"start": start_t, "end": d, "capacity": grid_capacity * (d - start_t), "index": 1})
        start_t = d

    must_run_load = {ev.ev_id: 0.0 for ev in ready_evs}
    sorted_evs_backward = sorted(ready_evs, key=lambda x: x.deadline, reverse=True)
    temp_segments = copy.deepcopy(segments)
    
    for ev in sorted_evs_backward:
        energy_needed = ev.remaining
        for seg in reversed(temp_segments):
            if energy_needed <= EPSILON: break
            if seg["start"] >= ev.deadline: continue
            
            fill = min(energy_needed, seg["capacity"], max_ev_power * (seg["end"] - seg["start"]))
            if fill > EPSILON:
                seg["capacity"] -= fill
                energy_needed -= fill
                if seg["index"] == 0: 
                    must_run_load[ev.ev_id] += fill

    # -----------------------------------------------------
    # 2. Execution Phase (Tier 1 + Tier 2)
    # -----------------------------------------------------
    final_allocations = {ev.ev_id: 0.0 for ev in ready_evs}
    
    # [Tier 1 Candidates]
    tier1_requests = []
    for ev in ready_evs:
        m_i = must_run_load[ev.ev_id] / time_step
        req = 0.0
        if m_i > EPSILON:
            req = max(m_i, min_ev_power)
            phys_limit = min(max_ev_power, ev.remaining / time_step)
            req = min(req, phys_limit)
        tier1_requests.append((ev, req))
    
    # [Tier 1 Allocation with Capacity Check]
    tier1_requests.sort(key=lambda x: x[0].deadline)
    current_load_sum = 0.0
    for ev, req in tier1_requests:
        if req <= EPSILON: continue
        if current_load_sum + req <= grid_capacity + EPSILON:
            final_allocations[ev.ev_id] = req
            current_load_sum += req
        else:
            rem = max(0.0, grid_capacity - current_load_sum)
            if rem > EPSILON:
                final_allocations[ev.ev_id] = rem
                current_load_sum += rem
            else:
                final_allocations[ev.ev_id] = 0.0

    # [Tier 2] Surplus Filling (Efficiency)
    surplus = max(0.0, grid_capacity - current_load_sum)
    sorted_evs_edf = sorted(ready_evs, key=lambda x: x.deadline)
    
    # --- [BUG FIX START] ---
    if min_ev_power <= EPSILON:
        # Case A: MinRate가 0인 경우 (Ideal Fluid Model)
        # "Activation" 단계가 필요 없으므로, 모든 차량에 대해 Surplus를 Greedy하게 할당 (EDF 순)
        for ev in sorted_evs_edf:
            if surplus <= EPSILON: break
            
            curr = final_allocations[ev.ev_id]
            phys_limit = min(max_ev_power, ev.remaining / time_step)
            room = phys_limit - curr
            
            if room > EPSILON:
                add = min(surplus, room)
                final_allocations[ev.ev_id] += add
                surplus -= add
    else:
        # Case B: MinRate가 존재하는 경우 (Discrete Constraint)
        # Step 2-1: 신규 가동 (대기 차량에게 I_min 부여)
        for ev in sorted_evs_edf:
            if surplus < min_ev_power - EPSILON: break 
            
            if final_allocations[ev.ev_id] <= EPSILON: # 대기 중인 차량
                phys_limit = min(max_ev_power, ev.remaining / time_step)
                # 물리적 요구량이 I_min보다 작으면 그것만, 아니면 I_min 할당
                req = min(min_ev_power, phys_limit)
                
                # I_min조차 받을 공간이 없는(배터리 거의 꽉참) 차는 패스하거나 남은 만큼만 줌
                # 하지만 S-RAS 정의상 Active되려면 I_min이 원칙.
                # 예외적으로 남은 양이 적으면 그것만 줘도 Active로 취급
                final_allocations[ev.ev_id] = req
                surplus -= req

        # Step 2-2: 가속 (이미 켜진 차량들에게 남은 자원 몰아주기)
        for ev in sorted_evs_edf:
            if surplus <= EPSILON: break
            
            curr = final_allocations[ev.ev_id]
            if curr > EPSILON: # Active 상태인 차량만 가속
                phys_limit = min(max_ev_power, ev.remaining / time_step)
                room = phys_limit - curr
                if room > EPSILON:
                    add = min(surplus, room)
                    final_allocations[ev.ev_id] += add
                    surplus -= add
    # --- [BUG FIX END] ---

    return [final_allocations[ev.ev_id] for ev in ready_evs]

def calculate_greedy_allocation(ready_evs, total_power, min_rate, max_rate, time_step, mode='EDF'):
    """
    Baseline (EDF, LLF) with Min Rate Constraint
    - Explicit Admission Control 없음.
    - 우선순위 높은 차에게 I_min 보장 -> 남으면 가속.
    """
    if not ready_evs: return []
    
    sorted_evs = []
    if mode == 'EDF':
        sorted_evs = sorted(ready_evs, key=lambda x: x.deadline)
    elif mode == 'LLF':
        sorted_evs = sorted(ready_evs, key=lambda x: (x.deadline - x.remaining/max_rate))
    
    allocations = {ev.ev_id: 0.0 for ev in ready_evs}
    used_power = 0.0
    active_list = []

    # Step A: Min Rate Allocation (Activation)
    for ev in sorted_evs:
        phys_req = min(max_rate, ev.remaining / time_step)
        req_to_activate = min(min_rate, phys_req)
        
        if used_power + req_to_activate <= total_power + EPSILON:
            allocations[ev.ev_id] = req_to_activate
            used_power += req_to_activate
            active_list.append(ev)
        else:
            allocations[ev.ev_id] = 0.0
            
    # Step B: Surplus Filling
    surplus = max(0.0, total_power - used_power)
    for ev in active_list:
        if surplus <= EPSILON: break
        curr = allocations[ev.ev_id]
        limit = min(max_rate, ev.remaining / time_step)
        room = limit - curr
        if room > EPSILON:
            add = min(surplus, room)
            allocations[ev.ev_id] += add
            surplus -= add
            
    return [allocations[ev.ev_id] for ev in ready_evs]

# ---------------------------------------------------------
# 4. 시뮬레이션 엔진
# ---------------------------------------------------------
def run_simulation(ev_set: List[EVRequest], algorithm: str) -> float:
    evs = copy.deepcopy(ev_set)
    total_evs_count = len(evs)
    waiting_queue = sorted(evs, key=lambda x: x.arrival) 
    active_evs = [] 
    
    success_count = 0
    failure_count = 0
    current_time = 0.0
    
    while success_count + failure_count < total_evs_count:
        # 1. 도착 처리
        while waiting_queue and waiting_queue[0].arrival <= current_time + EPSILON:
            active_evs.append(waiting_queue.pop(0))
            
        # 2. 데드라인 체크 (실패 처리)
        for i in range(len(active_evs) - 1, -1, -1):
            if current_time >= active_evs[i].deadline - EPSILON:
                failure_count += 1
                active_evs.pop(i)

        if not active_evs and not waiting_queue: break

        # 3. 전력 할당
        allocated_powers = []
        if algorithm == 'sLLF':
            allocated_powers = calculate_sllf_power(current_time, active_evs, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP)
        elif algorithm == 'Q-FAS':
            allocated_powers = calculate_sras_power(current_time, active_evs, TOTAL_STATION_POWER, MAX_EV_POWER, MIN_CHARGING_RATE, TIME_STEP)
        elif algorithm in ['EDF', 'LLF']:
            allocated_powers = calculate_greedy_allocation(active_evs, TOTAL_STATION_POWER, MIN_CHARGING_RATE, MAX_EV_POWER, TIME_STEP, mode=algorithm)
        
        # [Safety Check] 전체 전력 사용량이 Cap을 넘는지 확인 (알고리즘 오류 방지)
        total_requested = sum(allocated_powers)
        if total_requested > TOTAL_STATION_POWER + EPSILON:
            # 강제 스케일링 (Simulate Voltage Drop / Breaker Limit)
            scale = TOTAL_STATION_POWER / total_requested
            allocated_powers = [p * scale for p in allocated_powers]

        alloc_map = {ev.ev_id: allocated_powers[i] for i, ev in enumerate(active_evs)}

        # 4. 충전 업데이트
        for i in range(len(active_evs) - 1, -1, -1):
            ev = active_evs[i]
            p = alloc_map.get(ev.ev_id, 0.0)
            charged = min(ev.remaining, p * TIME_STEP)
            ev.remaining -= charged
            
            if ev.remaining <= EPSILON:
                success_count += 1
                active_evs.pop(i)
                
        current_time += TIME_STEP
        if current_time > 10000: break

    return (success_count / total_evs_count) * 100.0

# ---------------------------------------------------------
# 5. 실행 및 결과
# ---------------------------------------------------------
filename = F_PATH
requests = load_ev_data(filename)

if not requests:
    print("데이터를 로드할 수 없습니다.")
else:
    print(f"Loaded {len(requests)} requests.")
    print(f"Simulation Condition: Cap={TOTAL_STATION_POWER}kW, MinRate={MIN_CHARGING_RATE}kW")

    algorithms = []
    if MIN_CHARGING_RATE <= EPSILON:
        algorithms = ['EDF', 'LLF', 'sLLF', 'Q-FAS']
    else:
        algorithms = ['EDF', 'LLF', 'Q-FAS']
        
    results = {}

    for algo in algorithms:
        print(f"Running {algo}...", end=" ")
        score = run_simulation(requests, algo)
        results[algo] = score
        print(f"-> {score:.2f}%")

    plt.figure(figsize=(10, 6))
    bar_colors = {'EDF':'gray', 'LLF':'blue', 'sLLF':'red', 'Q-FAS':'green'}
    colors = [bar_colors[algo] for algo in algorithms]

    bars = plt.bar(results.keys(), results.values(), color=colors, alpha=0.8)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, 
                 f"{yval:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title(f'Success Rate (MinRate {MIN_CHARGING_RATE}kW)', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.savefig(S_PATH)
    print(f"Result saved to {S_PATH}")