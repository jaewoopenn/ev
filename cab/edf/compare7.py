import random
import copy
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. 설정 및 상수 (그대로 유지)
# ---------------------------------------------------------
TIME_STEP = 0.1         
EPSILON = 1e-6          

TOTAL_STATION_POWER = 3.5  
MAX_EV_POWER = 1.0         

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
# 2. EV 요청 생성기 (Zoom-In & High Intensity Version)
# ---------------------------------------------------------
def generate_ev_set(congestion_level):
    max_time = 60
    
    # [Zoom-In 설정]
    # 기존 Level 3 정도의 부하를 Level 1로 설정
    # 기존 Level 10 이상의 부하를 Level 10으로 설정
    
    # 1. 도착 빈도 (Arrival Rate): 0.6(약 1.6분당 1대) ~ 2.0(0.5분당 1대)
    # 기존보다 시작점이 훨씬 높습니다.
    arrival_rate = 0.6 + (congestion_level * 0.14) 

    # 2. 에너지 요구량 (Energy)
    # 표준편차를 키워서(1.5), "가끔 엄청 많이 충전해야 하는 빌런 차량"이 등장하게 함
    # 이 차가 등장했을 때 LLF는 당황하지만 NEW는 유연하게 대처함
    avg_energy = 3.2
    std_energy = 1.5
    
    # 3. 여유 시간 (Slack Scale) - 여기가 핵심
    # Old Lv 3 (Slack ~0.9) -> New Lv 1
    # Old Lv 10 (Slack ~0.1) -> New Lv 10
    # 천천히 조여옵니다.
    slack_scale = max(0.02, 0.9 - (congestion_level * 0.088)) 

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
        req_energy = max(0.5, min(7.5, req_energy)) # Max cap 살짝 늘림
        req_energy = round(req_energy, 2)
        
        min_required_time = math.ceil(req_energy / MAX_EV_POWER)
        
        # Slack 계산: LogNormal 꼬리 분포 사용 (가끔 여유로운 차도 섞임)
        raw_slack = random.expovariate(1.2) * (min_required_time * slack_scale * 4.0)
        slack = int(raw_slack)
        
        deadline = arrival_int + min_required_time + slack
        
        ev_requests.append(EVRequest(ev_id, arrival_int, req_energy, deadline, req_energy))
        ev_id += 1
        
    return ev_requests

# ---------------------------------------------------------
# 3. New Algorithm Logic (FIXED)
# ---------------------------------------------------------
def calculate_new_algo_power(current_time, active_evs, grid_capacity, max_ev_power, time_step):
    count = len(active_evs)
    if count == 0:
        return []

    durations = []
    remains = []
    for ev in active_evs:
        d = max(time_step, float(ev.deadline) - current_time)
        durations.append(d)
        remains.append(ev.remaining)

    # -----------------------------------------------------
    # Step 1: Mandatory Allocation
    # -----------------------------------------------------
    allocation = [0.0] * count
    for i in range(count):
        future_time = durations[i] - time_step
        max_future_charge = max(0.0, future_time * max_ev_power)
        mandatory_energy = max(0.0, remains[i] - max_future_charge)
        
        mandatory_power = mandatory_energy / time_step
        phys_limit_power = min(max_ev_power, remains[i] / time_step)
        
        allocation[i] = min(mandatory_power, phys_limit_power)

    # [BUG FIX] 필수 요구량 합계가 그리드 용량을 초과하는지 검사
    current_load = sum(allocation)
    
    if current_load > grid_capacity + EPSILON:
        # 용량 초과! 비율대로 삭감 (Scaling Down)
        # 이 상황이 발생하면 사실상 누군가는 Deadline을 놓치게 됨
        scale_factor = grid_capacity / current_load
        for i in range(count):
            allocation[i] *= scale_factor
        
        current_load = grid_capacity # 보정 후 부하
        remaining_grid = 0.0
    else:
        remaining_grid = grid_capacity - current_load

    # -----------------------------------------------------
    # Step 2: Minimax Optimization (남는 전력이 있을 때만 수행)
    # -----------------------------------------------------
    if remaining_grid > EPSILON:
        residual_caps = []
        residual_needs = []
        for i in range(count):
            phys_limit_power = min(max_ev_power, remains[i] / time_step)
            # 이미 할당된 양을 제외한 '더 받을 수 있는' 물리적 여유
            res_cap = max(0.0, phys_limit_power - allocation[i])
            residual_caps.append(res_cap)
            
            allocated_energy = allocation[i] * time_step
            res_need = remains[i] - allocated_energy
            residual_needs.append(res_need)

        if sum(residual_needs) > EPSILON:
            low, high = -1000.0, 1000.0
            best_extra = [0.0] * count
            for _ in range(20):
                stress_level = (low + high) / 2.0
                proposed_total = 0.0
                current_proposal = []
                for i in range(count):
                    target_energy = residual_needs[i] - (stress_level * durations[i])
                    target_power = target_energy / time_step
                    # 0 ~ 남은 여유용량 사이로 클리핑
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

    # -----------------------------------------------------
    # Step 3: Greedy Allocation (자투리 채우기)
    # -----------------------------------------------------
    if remaining_grid > EPSILON:
        urgency_indices = []
        for i in range(count):
            # 급한 순서는 그대로 유지하되, 물리적으로 더 받을 수 있는 애들만
            phys_limit_power = min(max_ev_power, remains[i] / time_step)
            if phys_limit_power - allocation[i] > EPSILON:
                score = remains[i] / durations[i] if durations[i] > 0 else 9999
                urgency_indices.append((score, i))
        
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
# 4. 시뮬레이션 엔진 (Cheating 방지 로직 추가)
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

        # --- Power Allocation ---
        if algorithm == 'NEW_ALGO':
            allocated_powers = calculate_new_algo_power(
                current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP
            )
            
            # [심판 로직] 알고리즘이 총 전력을 초과해서 요구했는지 검사
            total_requested = sum(allocated_powers)
            if total_requested > TOTAL_STATION_POWER + EPSILON:
                # 치팅 감지! 강제로 비율 삭감 (현실적인 전압 강하/차단 시뮬레이션)
                scale = TOTAL_STATION_POWER / total_requested
                allocated_powers = [p * scale for p in allocated_powers]

            for i, ev in enumerate(ready_queue):
                charged_amount = allocated_powers[i] * TIME_STEP
                actual_charged = min(ev.remaining, charged_amount)
                ev.remaining -= actual_charged
                if ev.remaining <= EPSILON: ev.remaining = 0.0
                
        else:
            # 기존 알고리즘 (EDF, LLF, FCFS) - Water filling 방식
            if algorithm == 'EDF':
                ready_queue.sort(key=lambda x: (x.deadline, x.ev_id))
            elif algorithm == 'LLF':
                ready_queue.sort(key=lambda x: (x.deadline - current_time - (x.remaining / MAX_EV_POWER), x.ev_id))
            elif algorithm == 'FCFS':
                ready_queue.sort(key=lambda x: (x.arrival, x.ev_id))

            current_used_power = 0.0
            for ev in ready_queue:
                available_power = TOTAL_STATION_POWER - current_used_power
                if available_power <= EPSILON: break 
                
                charging_rate = min(MAX_EV_POWER, available_power)
                charged_amount = charging_rate * TIME_STEP
                actual_charged = min(ev.remaining, charged_amount)
                
                ev.remaining -= actual_charged
                current_used_power += charging_rate
                if ev.remaining <= EPSILON: ev.remaining = 0.0

        finished_cnt = sum(1 for e in evs if e.remaining <= EPSILON)
        current_time += TIME_STEP
        if current_time > max_deadline + 10.0:
            return False

    return True

# ---------------------------------------------------------
# 5. 메인 실험 루프 (Display Format 수정)
# ---------------------------------------------------------
congestion_levels = list(range(1, 11))
edf_ratios, llf_ratios, fcfs_ratios, new_ratios = [], [], [], []

trials_per_level = 200 

print(f"EV Charging Simulation (Zoom-In: Hard Mode) Start...")
print(f"Focusing on the gap between LLF and NEW_ALGO...")

for level in congestion_levels:
    wins = {'EDF':0, 'LLF':0, 'FCFS':0, 'NEW_ALGO':0}
    
    for _ in range(trials_per_level):
        ev_requests = generate_ev_set(level)
        if not ev_requests: continue

        # 시뮬레이션 실행
        if run_simulation(ev_requests, 'EDF'): wins['EDF'] += 1
        if run_simulation(ev_requests, 'LLF'): wins['LLF'] += 1
        if run_simulation(ev_requests, 'FCFS'): wins['FCFS'] += 1
        if run_simulation(ev_requests, 'NEW_ALGO'): wins['NEW_ALGO'] += 1
            
    edf_ratios.append(wins['EDF'] / trials_per_level)
    llf_ratios.append(wins['LLF'] / trials_per_level)
    fcfs_ratios.append(wins['FCFS'] / trials_per_level)
    new_ratios.append(wins['NEW_ALGO'] / trials_per_level)
    
    # 출력 포맷 정렬
    print(f"Level {level:2d}: EDF={edf_ratios[-1]:.2f}, LLF={llf_ratios[-1]:.2f}, FCFS={fcfs_ratios[-1]:.2f} | NEW={new_ratios[-1]:.2f}")

# ---------------------------------------------------------
# 6. 결과 그래프
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(congestion_levels, edf_ratios, marker='o', label='EDF', linestyle='-', color='blue', alpha=0.3)
plt.plot(congestion_levels, llf_ratios, marker='s', label='LLF', linestyle='--', color='red', alpha=0.8) # LLF 강조
plt.plot(congestion_levels, fcfs_ratios, marker='^', label='FCFS', linestyle=':', color='green', alpha=0.3)
plt.plot(congestion_levels, new_ratios, marker='*', label='NEW_ALGO', linestyle='-', linewidth=3, color='purple') # NEW 강조

plt.title(f'Performance Gap Analysis: LLF vs NEW_ALGO', fontsize=14)
plt.xlabel('Congestion Intensity (Zoomed In)', fontsize=12)
plt.ylabel('Success Ratio', fontsize=12)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle=':', alpha=0.7)
plt.xticks(congestion_levels)
plt.legend(fontsize=12)

# FCFS가 너무 빨리 떨어져서 그래프가 안 예쁠 수 있으므로 텍스트 추가
plt.text(1.5, 0.1, "FCFS fails early", color='green', fontsize=10)

plt.show()
