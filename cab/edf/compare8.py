import random
import copy
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
TRIAL_NUM=200
# TRIAL_NUM=100
TIME_STEP = 0.1         
EPSILON = 1e-6          

TOTAL_STATION_POWER = 6
MAX_EV_POWER = 1.0         

STRESS_START=5
STRESS_NUM=10


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
# 2. EV 요청 생성기 (Super Fine-grained 12 Levels)
# ---------------------------------------------------------
def generate_ev_set(congestion_level):
    max_time = 60
    
    # [설정 변경] 이전 코드의 1~9단계를 1~12단계로 세분화 (Stretch)
    # 목표: 그래프의 기울기를 완만하게 만들어 성능 차이 구간을 길게 확보
    
    # 1. Arrival Rate (도착 빈도)
    # 이전: 0.55 + (L * 0.10) => 9단계일 때 약 1.45
    # 변경: 12단계일 때 1.45가 되려면 => step 약 0.075
    arrival_rate = 0.55 + (congestion_level * 0.075)
    # arrival_rate = 0.55 + (5 * 0.075)

    # 2. Avg Energy (요구량)
    # 이전: 2.5 + (L * 0.12) => 9단계일 때 약 3.58
    # 변경: 12단계일 때 3.58이 되려면 => step 약 0.09
    base_avg_energy = 2.5
    avg_energy = base_avg_energy + (congestion_level * 0.09)
    # avg_energy = base_avg_energy + (5 * 0.09)
    std_energy = 1.3 # 편차를 살짝 키워 변별력 강화

    # 3. Slack Factor (여유 시간)
    # 이전: 1.9 - (L * 0.13) => 9단계일 때 약 0.73
    # 변경: 12단계일 때 0.73이 되려면 => step 약 0.098
    # 여유 시간이 아주 천천히 줄어듭니다.
    slack_mean_factor = max(0.4, 1.9 - (congestion_level * 0.098))
    # slack_mean_factor = max(0.4, 1.9 - (5 * 0.098))

    ev_requests = []
    current_time = 0.0
    ev_id = 0
    
    while True:
        inter_arrival = random.expovariate(arrival_rate)
        current_time += inter_arrival
        
        if current_time > max_time:
            break
            
        arrival_int = int(current_time)
        
        req_energy = random.gauss(avg_energy, std_energy)
        req_energy = max(0.5, min(8.0, req_energy))
        req_energy = round(req_energy, 2)
        
        min_required_time = math.ceil(req_energy / MAX_EV_POWER)
        
        # Slack Variance
        raw_slack = random.expovariate(1.0) * (min_required_time * slack_mean_factor)
        slack = int(raw_slack)
        
        deadline = arrival_int + min_required_time + slack
        
        ev_requests.append(EVRequest(ev_id, arrival_int, req_energy, deadline, req_energy))
        ev_id += 1
        
    return ev_requests

# ---------------------------------------------------------
# 3. New Algorithm Logic (그대로 유지)
# ---------------------------------------------------------
def calculate_new_algo_power(current_time, active_evs, grid_capacity, max_ev_power, time_step):
    count = len(active_evs)
    if count == 0: return []

    durations = []
    remains = []
    for ev in active_evs:
        d = max(time_step, float(ev.deadline) - current_time)
        durations.append(d)
        remains.append(ev.remaining)

    # Step 1: Mandatory
    allocation = [0.0] * count
    for i in range(count):
        future_time = durations[i] - time_step
        max_future_charge = max(0.0, future_time * max_ev_power)
        mandatory_energy = max(0.0, remains[i] - max_future_charge)
        
        mandatory_power = mandatory_energy / time_step
        phys_limit_power = min(max_ev_power, remains[i] / time_step)
        allocation[i] = min(mandatory_power, phys_limit_power)

    current_load = sum(allocation)
    if current_load > grid_capacity + EPSILON:
        scale_factor = grid_capacity / current_load
        for i in range(count):
            allocation[i] *= scale_factor
        remaining_grid = 0.0
    else:
        remaining_grid = grid_capacity - current_load

    # Step 2: Minimax
    if remaining_grid > EPSILON:
        residual_caps, residual_needs = [], []
        for i in range(count):
            phys_limit_power = min(max_ev_power, remains[i] / time_step)
            res_cap = max(0.0, phys_limit_power - allocation[i])
            residual_caps.append(res_cap)
            allocated_energy = allocation[i] * time_step
            res_need = remains[i] - allocated_energy
            residual_needs.append(res_need)

        if sum(residual_needs) > EPSILON:
            low, high = -1000.0, 1000.0
            best_extra = [0.0] * count
            for _ in range(15):
                stress_level = (low + high) / 2.0
                proposed_total = 0.0
                current_proposal = []
                for i in range(count):
                    target_energy = residual_needs[i] - (stress_level * durations[i])
                    target_power = target_energy / time_step
                    p_power = max(0.0, min(target_power, residual_caps[i]))
                    current_proposal.append(p_power)
                    proposed_total += p_power
                
                if proposed_total > remaining_grid:
                    low = stress_level
                else:
                    high = stress_level
                    best_extra = current_proposal
            
            for i in range(count):
                allocation[i] += best_extra[i]
                remaining_grid -= best_extra[i]

    # Step 3: Greedy
    if remaining_grid > EPSILON:
        urgency_indices = []
        for i in range(count):
            phys_limit_power = min(max_ev_power, remains[i] / time_step)
            if phys_limit_power - allocation[i] > EPSILON:
                score = remains[i] / durations[i] if durations[i] > 0 else 9999
                urgency_indices.append((score, i))
        urgency_indices.sort(key=lambda x: x[0], reverse=True)
        
        for score, idx in urgency_indices:
            if remaining_grid <= EPSILON: break
            phys_limit_power = min(max_ev_power, remains[idx] / time_step)
            room_to_charge = phys_limit_power - allocation[idx]
            if room_to_charge > EPSILON:
                top_up = min(remaining_grid, room_to_charge)
                allocation[idx] += top_up
                remaining_grid -= top_up

    return allocation

# ---------------------------------------------------------
# 4. 시뮬레이션 엔진 (유지)
# ---------------------------------------------------------
def run_simulation(ev_set: List[EVRequest], algorithm: str) -> bool:
    evs = copy.deepcopy(ev_set)
    max_deadline = max(e.deadline for e in evs) if evs else 0
    current_time = 0.0
    finished_cnt = 0
    total_evs = len(evs)

    while finished_cnt < total_evs:
        ready_queue = [
            e for e in evs 
            if e.arrival <= current_time + EPSILON and e.remaining > EPSILON
        ]
        
        for e in ready_queue:
            if current_time > float(e.deadline) + EPSILON:
                return False 

        if algorithm == 'NEW_ALGO':
            allocated_powers = calculate_new_algo_power(
                current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP
            )
            if sum(allocated_powers) > TOTAL_STATION_POWER + EPSILON:
                scale = TOTAL_STATION_POWER / sum(allocated_powers)
                allocated_powers = [p * scale for p in allocated_powers]
            
            for i, ev in enumerate(ready_queue):
                charged = min(ev.remaining, allocated_powers[i] * TIME_STEP)
                ev.remaining -= charged
                if ev.remaining <= EPSILON: ev.remaining = 0.0

        else:
            if algorithm == 'EDF':
                ready_queue.sort(key=lambda x: (x.deadline, x.ev_id))
            elif algorithm == 'LLF':
                ready_queue.sort(key=lambda x: (x.deadline - current_time - (x.remaining/MAX_EV_POWER), x.ev_id))
            elif algorithm == 'FCFS':
                ready_queue.sort(key=lambda x: (x.arrival, x.ev_id))

            current_used = 0.0
            for ev in ready_queue:
                available = TOTAL_STATION_POWER - current_used
                if available <= EPSILON: break
                rate = min(MAX_EV_POWER, available)
                charged = min(ev.remaining, rate * TIME_STEP)
                ev.remaining -= charged
                current_used += rate
                if ev.remaining <= EPSILON: ev.remaining = 0.0

        finished_cnt = sum(1 for e in evs if e.remaining <= EPSILON)
        current_time += TIME_STEP
        if current_time > max_deadline + 10.0:
            return False

    return True

# ---------------------------------------------------------
# 5. 메인 실험 (12단계)
# ---------------------------------------------------------
# 12단계로 확장
# congestion_levels = list(range(1, 13))
congestion_levels = list(range(STRESS_START, STRESS_START+STRESS_NUM))
edf_ratios, llf_ratios, fcfs_ratios, new_ratios = [], [], [], []
trials_per_level = TRIAL_NUM

print(f"EV Simulation (Fine-Grained 12 Levels)")
print(f"Mapping old Levels 1-9 to New Levels 1-12...")

for level in congestion_levels:
    wins = {'EDF':0, 'LLF':0, 'FCFS':0, 'NEW_ALGO':0}
    for _ in range(trials_per_level):
        ev_requests = generate_ev_set(level)
        if not ev_requests: continue

        if run_simulation(ev_requests, 'EDF'): wins['EDF'] += 1
        if run_simulation(ev_requests, 'LLF'): wins['LLF'] += 1
        if run_simulation(ev_requests, 'FCFS'): wins['FCFS'] += 1
        if run_simulation(ev_requests, 'NEW_ALGO'): wins['NEW_ALGO'] += 1
            
    edf_ratios.append(wins['EDF'] / trials_per_level)
    llf_ratios.append(wins['LLF'] / trials_per_level)
    fcfs_ratios.append(wins['FCFS'] / trials_per_level)
    new_ratios.append(wins['NEW_ALGO'] / trials_per_level)
    
    print(f"Level {level:2d}: EDF={edf_ratios[-1]:.2f}, LLF={llf_ratios[-1]:.2f}, FCFS={fcfs_ratios[-1]:.2f} | NEW={new_ratios[-1]:.2f}")

# ---------------------------------------------------------
# 6. 결과 그래프
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(congestion_levels, edf_ratios, marker='o', label='EDF', linestyle='-', color='blue', alpha=0.3)
plt.plot(congestion_levels, llf_ratios, marker='s', label='LLF', linestyle='--', color='red', alpha=0.8)
plt.plot(congestion_levels, fcfs_ratios, marker='^', label='FCFS', linestyle=':', color='green', alpha=0.3)
plt.plot(congestion_levels, new_ratios, marker='*', label='NEW_ALGO', linestyle='-', linewidth=3, color='purple')

plt.title(f'Performance Comparison (12-Step Zoomed View)', fontsize=14)
plt.xlabel('Stress Level (Fine-grained)', fontsize=12)
plt.ylabel('Success Ratio', fontsize=12)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle=':', alpha=0.7)
plt.xticks(congestion_levels)
plt.legend(fontsize=12)
plt.show()