import random
import copy
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
TIME_STEP = 0.1         
EPSILON = 1e-6          

TOTAL_STATION_POWER = 4.2  # 충전소 전체 가용 전력 (kW)
MAX_EV_POWER = 1.0         # 개별 EV의 최대 충전 속도 (kW)

@dataclass
class EVRequest:
    ev_id: int
    arrival: int        
    required_energy: float 
    deadline: int       
    remaining: float    

    def __repr__(self):
        return f"EV{self.ev_id}(A={self.arrival}, E={self.required_energy:.2f}, D={self.deadline})"

# ---------------------------------------------------------
# 2. EV 요청 생성기
# ---------------------------------------------------------
def generate_ev_set(congestion_level):
    max_time = 60
    base_jobs = int(TOTAL_STATION_POWER * 2)
    step = int(TOTAL_STATION_POWER * 1.5)
    num_jobs = int(base_jobs + (congestion_level * step))
    
    ev_requests = []
    for i in range(num_jobs):
        arrival = random.randint(0, int(max_time * 0.6))
        required_energy = round(random.uniform(0.5, 8.0), 2)
        # 최소 필요 시간 계산
        min_required_time = math.ceil(required_energy / MAX_EV_POWER)
        # 여유 시간(Slack) 부여
        slack = random.randint(0, min_required_time * 2) 
        deadline = arrival + min_required_time + slack
        
        ev_requests.append(EVRequest(i, arrival, required_energy, deadline, required_energy))
    return ev_requests

# ---------------------------------------------------------
# 3. New Algorithm Logic (Pure Python 변환 버전)
# ---------------------------------------------------------
def calculate_new_algo_power(current_time, active_evs, grid_capacity, max_ev_power, time_step):
    """
    3-Phase Scheduling Algorithm (Mandatory -> Minimax -> Greedy)
    NumPy 없이 순수 Python으로 구현하여 호환성 유지
    Returns: List of allocated power (kW) for each EV corresponding to active_evs indices
    """
    count = len(active_evs)
    if count == 0:
        return []

    # ---------------------------------------------------------
    # 0. Data Preparation
    # ---------------------------------------------------------
    # 남은 시간 (Duration): 현재 시간부터 마감까지 남은 시간 (최소 1 Time Step 보장)
    durations = []
    remains = []
    for ev in active_evs:
        d = max(time_step, float(ev.deadline) - current_time)
        durations.append(d)
        remains.append(ev.remaining)

    # ---------------------------------------------------------
    # Step 1: Mandatory Allocation (Deadline 방어)
    # ---------------------------------------------------------
    # 미래에 최대로 충전해도 부족할 양은 '지금' 무조건 받아야 함
    allocation = [0.0] * count
    
    for i in range(count):
        # 미래에 활용 가능한 최대 에너지 (현재 스텝 이후부터 마감까지)
        future_time = durations[i] - time_step
        max_future_charge = max(0.0, future_time * max_ev_power)
        
        # 지금 당장 필요한 최소 에너지
        mandatory_energy = max(0.0, remains[i] - max_future_charge)
        
        # 이를 Power(속도)로 변환: 에너지 / 시간 = 전력
        # 단, 물리적 한계(MAX_EV_POWER)와 현재 필요량(remains)을 넘을 순 없음
        mandatory_power = mandatory_energy / time_step
        
        # 상한선: Max Power vs 남은 에너지 전체를 이번 턴에 다 받는 속도
        phys_limit_power = min(max_ev_power, remains[i] / time_step)
        
        allocation[i] = min(mandatory_power, phys_limit_power)

    current_load = sum(allocation)
    remaining_grid = grid_capacity - current_load

    # 만약 필수 할당만으로 전력이 부족하면, 비율대로 줄이거나 그대로 리턴 (Overload)
    # 여기서는 시뮬레이션 특성상 그냥 진행(나중에 Cut)하거나, 비례 축소할 수 있음.
    # 본 로직은 'Mandatory'는 확보하려고 노력하는 구조이므로 일단 진행.

    # ---------------------------------------------------------
    # Step 2: Minimax Optimization (남는 전력 공평 분배)
    # ---------------------------------------------------------
    if remaining_grid > EPSILON:
        # 이 단계에서 추가로 더 받을 수 있는 전력량(Residual Capacity) 계산
        residual_caps = []
        residual_needs = [] # 에너지 기준
        
        for i in range(count):
            phys_limit_power = min(max_ev_power, remains[i] / time_step)
            res_cap = max(0.0, phys_limit_power - allocation[i])
            residual_caps.append(res_cap)
            
            # 이미 할당된 에너지를 제외한 남은 필요 에너지
            allocated_energy = allocation[i] * time_step
            res_need = remains[i] - allocated_energy
            residual_needs.append(res_need)

        if sum(residual_needs) > EPSILON:
            # Binary Search for optimal 'stress level'
            low, high = -1000.0, 1000.0
            best_extra = [0.0] * count
            
            for _ in range(20): # Max Iteration 20
                stress_level = (low + high) / 2.0
                proposed_total = 0.0
                current_proposal = []
                
                for i in range(count):
                    # Minimax Metric: Target Energy = Need - (Stress * Time)
                    # 여유 시간이 길수록(Duration 큼) Stress 영향을 많이 받아 할당량이 줄어듦
                    target_energy = residual_needs[i] - (stress_level * durations[i])
                    target_power = target_energy / time_step
                    
                    # 0 ~ 물리적 한계 사이로 클리핑
                    p_power = max(0.0, min(target_power, residual_caps[i]))
                    current_proposal.append(p_power)
                    proposed_total += p_power
                
                if proposed_total > remaining_grid:
                    low = stress_level # 너무 많이 줌 -> 스트레스 높임
                else:
                    high = stress_level
                    best_extra = current_proposal # 가능한 해 저장
            
            # 최적화 결과 반영
            for i in range(count):
                allocation[i] += best_extra[i]
                remaining_grid -= best_extra[i]

    # ---------------------------------------------------------
    # Step 3: Greedy Allocation (자투리 채우기)
    # ---------------------------------------------------------
    if remaining_grid > EPSILON:
        # Urgency = Energy Density (남은 에너지 / 남은 시간)
        urgency_indices = []
        for i in range(count):
            score = remains[i] / durations[i] if durations[i] > 0 else 9999
            urgency_indices.append((score, i))
        
        # 급한 순서대로 정렬 (점수 높은 순)
        urgency_indices.sort(key=lambda x: x[0], reverse=True)
        
        for score, idx in urgency_indices:
            if remaining_grid <= EPSILON:
                break
            
            phys_limit_power = min(max_ev_power, remains[idx] / time_step)
            room_to_charge = phys_limit_power - allocation[idx]
            
            if room_to_charge > EPSILON:
                top_up = min(remaining_grid, room_to_charge)
                allocation[idx] += top_up
                remaining_grid -= top_up

    return allocation

# ---------------------------------------------------------
# 4. 시뮬레이션 엔진 (수정됨)
# ---------------------------------------------------------
def run_simulation(ev_set: List[EVRequest], algorithm: str) -> bool:
    evs = copy.deepcopy(ev_set)
    max_deadline = max(e.deadline for e in evs) if evs else 0
    
    current_time = 0.0
    finished_cnt = 0
    total_evs = len(evs)

    while finished_cnt < total_evs:
        # 1. Ready Queue Filter
        ready_queue = [
            e for e in evs 
            if e.arrival <= current_time + EPSILON and e.remaining > EPSILON
        ]
        
        # 2. Deadline Miss Check
        for e in ready_queue:
            if current_time > float(e.deadline) + EPSILON:
                return False 

        # -----------------------------------------------------
        # Power Allocation Logic
        # -----------------------------------------------------
        # NEW_ALGO는 할당 로직이 근본적으로 다름 (Sort -> Fill 방식이 아님)
        
        if algorithm == 'NEW_ALGO':
            # 전체 동시 계산
            allocated_powers = calculate_new_algo_power(
                current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP
            )
            
            # 계산된 전력 적용
            for i, ev in enumerate(ready_queue):
                power = allocated_powers[i]
                charged_amount = power * TIME_STEP
                # 실제 잔여량보다 더 충전할 순 없음 (부동소수점 오차 보정)
                actual_charged = min(ev.remaining, charged_amount)
                ev.remaining -= actual_charged
                if ev.remaining <= EPSILON:
                    ev.remaining = 0.0

        else:
            # 기존 알고리즘들 (Sort -> Water-filling)
            
            # 3. Sort
            if algorithm == 'EDF':
                ready_queue.sort(key=lambda x: (x.deadline, x.ev_id))
            elif algorithm == 'LLF':
                # Laxity = (Deadline - Now) - (Remaining / MaxRate)
                ready_queue.sort(key=lambda x: (x.deadline - current_time - (x.remaining / MAX_EV_POWER), x.ev_id))
            elif algorithm == 'FCFS':
                ready_queue.sort(key=lambda x: (x.arrival, x.ev_id))

            # 4. Water-filling Allocation
            current_used_power = 0.0
            for ev in ready_queue:
                available_power = TOTAL_STATION_POWER - current_used_power
                if available_power <= EPSILON:
                    break 
                
                charging_rate = min(MAX_EV_POWER, available_power)
                charged_amount = charging_rate * TIME_STEP
                actual_charged = min(ev.remaining, charged_amount)
                
                ev.remaining -= actual_charged
                current_used_power += charging_rate
                
                if ev.remaining <= EPSILON:
                    ev.remaining = 0.0

        # Update Count
        finished_cnt = sum(1 for e in evs if e.remaining <= EPSILON)
        current_time += TIME_STEP
        
        if current_time > max_deadline + 10.0:
            return False

    return True

# ---------------------------------------------------------
# 5. 메인 실험 루프
# ---------------------------------------------------------
congestion_levels = list(range(1, 11))

edf_ratios = []
llf_ratios = []
fcfs_ratios = []
new_ratios = []  # New Algo 결과 저장용

trials_per_level = 50 # 속도를 위해 100 -> 50으로 임시 조정 (원하면 100으로 변경 가능)

print(f"EV Charging Simulation Start (EDF vs LLF vs FCFS vs NEW_ALGO)...")
print(f"Total Power: {TOTAL_STATION_POWER}kW, Max EV Rate: {MAX_EV_POWER}kW")

for level in congestion_levels:
    edf_wins = 0
    llf_wins = 0
    fcfs_wins = 0
    new_wins = 0
    
    for _ in range(trials_per_level):
        ev_requests = generate_ev_set(level)
        
        if run_simulation(ev_requests, 'EDF'):
            edf_wins += 1
        if run_simulation(ev_requests, 'LLF'):
            llf_wins += 1
        if run_simulation(ev_requests, 'FCFS'):
            fcfs_wins += 1
        if run_simulation(ev_requests, 'NEW_ALGO'):
            new_wins += 1
            
    edf_ratios.append(edf_wins / trials_per_level)
    llf_ratios.append(llf_wins / trials_per_level)
    fcfs_ratios.append(fcfs_wins / trials_per_level)
    new_ratios.append(new_wins / trials_per_level)
    
    print(f"Level {level}: EDF={edf_ratios[-1]:.2f}, LLF={llf_ratios[-1]:.2f}, FCFS={fcfs_ratios[-1]:.2f}, NEW={new_ratios[-1]:.2f}")

# ---------------------------------------------------------
# 6. 결과 그래프
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(congestion_levels, edf_ratios, marker='o', label='EDF', linestyle='-', color='blue', alpha=0.7)
plt.plot(congestion_levels, llf_ratios, marker='s', label='LLF', linestyle='--', color='red', alpha=0.7)
plt.plot(congestion_levels, fcfs_ratios, marker='^', label='FCFS', linestyle='-.', color='green', alpha=0.7)
plt.plot(congestion_levels, new_ratios, marker='*', label='NEW_ALGO', linestyle='-', linewidth=2, color='purple')

plt.title(f'EV Charging Success Ratio Comparison\n(Mandatory+Minimax+Greedy Added)', fontsize=14)
plt.xlabel('Congestion Level (Traffic Intensity)', fontsize=12)
plt.ylabel('Success Ratio (Deadline Met)', fontsize=12)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle=':', alpha=0.7)
plt.xticks(congestion_levels)
plt.legend(fontsize=12)

plt.show()