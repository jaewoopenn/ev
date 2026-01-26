import random
import copy
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. 설정 및 상수 (모든 EV 동일 최대 속도 가정)
# ---------------------------------------------------------
TRIAL_NUM = 100         # 실험 횟수
TIME_STEP = 0.1         # 시뮬레이션 시간 단위
EPSILON = 1e-6          

TOTAL_STATION_POWER = 6.0   # 충전소 전체 전력량
MAX_EV_POWER = 1.0          # 단일 EV의 최대 충전 속도 (모두 동일하다고 가정)

STRESS_START = 5
STRESS_NUM = 10

@dataclass
class EVRequest:
    ev_id: int
    arrival: int        
    required_energy: float 
    deadline: int       
    remaining: float    

    def __repr__(self):
        return f"EV{self.ev_id}(A={self.arrival}, Rem={self.remaining:.2f}, D={self.deadline})"

# ---------------------------------------------------------
# 2. EV 데이터 생성기
# ---------------------------------------------------------
def generate_ev_set(congestion_level):
    max_time = 60
    
    # 혼잡도 레벨에 따라 도착률과 에너지 요구량을 증가시켜 부하를 조절
    arrival_rate = 0.55 + (congestion_level * 0.075)
    base_avg_energy = 2.5
    avg_energy = base_avg_energy + (congestion_level * 0.09)
    std_energy = 1.3 
    slack_mean_factor = max(0.4, 1.9 - (congestion_level * 0.098))

    ev_requests = []
    current_time = 0.0
    ev_id = 0
    
    while True:
        inter_arrival = random.expovariate(arrival_rate)
        current_time += inter_arrival
        if current_time > max_time: break
            
        arrival_int = int(current_time)
        req_energy = random.gauss(avg_energy, std_energy)
        req_energy = max(0.5, min(8.0, req_energy))
        req_energy = round(req_energy, 2)
        
        # 최소 필요 시간 계산 (MAX_EV_POWER가 모두 동일하므로 단순 나눗셈)
        min_required_time = math.ceil(req_energy / MAX_EV_POWER)
        
        raw_slack = random.expovariate(1.0) * (min_required_time * slack_mean_factor)
        slack = int(raw_slack)
        deadline = arrival_int + min_required_time + slack
        
        ev_requests.append(EVRequest(ev_id, arrival_int, req_energy, deadline, req_energy))
        ev_id += 1
        
    return ev_requests

# ---------------------------------------------------------
# 3. sLLF 알고리즘 (핵심 구현)
# ---------------------------------------------------------
def calculate_sllf_power(current_time, active_evs, grid_capacity, max_ev_power, time_step):
    """
    [논문 구현] Smoothed Least-Laxity-First (sLLF)
    Bisection Search를 통해 다음 타임스텝의 목표 Laxity(L)을 찾고,
    이에 맞춰 전력을 배분함.
    """
    count = len(active_evs)
    if count == 0: return []

    # 1. 현재 시점의 Laxity 계산 [cite: 115]
    # Laxity = (남은 시간) - (완충까지 필요한 최소 시간)
    current_laxities = []
    phys_limits = []  # 물리적으로 가능한 최대 충전량 (배터리 잔량 or 최대 속도)

    for ev in active_evs:
        # 남은 시간
        remaining_time = ev.deadline - current_time
        # 완충 시간 = 남은 에너지 / 최대 속도
        time_to_charge = ev.remaining / max_ev_power
        
        l_t = remaining_time - time_to_charge
        current_laxities.append(l_t)
        
        # 이번 스텝에 충전 가능한 물리적 한계
        p_limit = min(max_ev_power, ev.remaining / time_step)
        phys_limits.append(p_limit)

    # 모든 차량을 풀파워로 충전해도 그리드 용량이 남는다면 굳이 조절할 필요 없음
    if sum(phys_limits) <= grid_capacity + EPSILON:
        return phys_limits

    # 2. 목표 Laxity(L)에 따른 필요 전력 계산 함수
    # 논문 Eq (16): r_i(t) = max_rate * (L - l_i(t) + 1)  (여기서 1은 단위시간)
    # 변형 식: Power = MaxPower * (Target_L - Current_L + time_step) / time_step
    def get_total_power_for_target_L(target_L):
        total_p = 0.0
        allocs = []
        for i in range(count):
            # 다음 타임스텝의 Laxity가 target_L이 되기 위해 필요한 파워
            # 유도: L(t+1) = L(t) - time_step + (Power * time_step / MaxPower)
            #      target_L = current_L - time_step + (Power * time_step / MaxPower)
            #      Power = (target_L - current_L + time_step) * MaxPower / time_step
            
            req_p = (max_ev_power / time_step) * (target_L - current_laxities[i] + time_step)
            
            # [0, min(MaxPower, Remaining)] 범위로 클램핑 
            req_p = max(0.0, min(req_p, phys_limits[i]))
            
            allocs.append(req_p)
            total_p += req_p
        return total_p, allocs

    # 3. Bisection Search (이분 탐색) 
    # 적절한 L을 찾아서 Total Power == Grid Capacity가 되도록 맞춤
    low_L = -200.0  # 충분히 작은 Laxity
    high_L = 200.0  # 충분히 큰 Laxity
    best_allocations = [0.0] * count
    
    # 20회 반복이면 float 정밀도에서 충분히 수렴
    for _ in range(20):
        mid_L = (low_L + high_L) / 2.0
        p_sum, p_allocs = get_total_power_for_target_L(mid_L)
        
        if p_sum > grid_capacity:
            # 전력이 모자람 -> 전력 소비를 줄여야 함
            # 전력 소비를 줄인다는 건 충전을 덜 한다는 뜻 -> Laxity가 더 빨리 줄어듦(더 작아짐)
            # 즉, 목표 Laxity(Target L)을 낮춰야 함
            high_L = mid_L
        else:
            # 전력이 남음 -> 더 충전해서 Laxity를 높일 수 있음
            low_L = mid_L
            best_allocations = p_allocs # Feasible한 해를 저장

    # 부동소수점 오차로 미세하게 초과하는 경우 보정
    if sum(best_allocations) > grid_capacity + EPSILON:
        scale = grid_capacity / sum(best_allocations)
        best_allocations = [p * scale for p in best_allocations]
        
    return best_allocations

# ---------------------------------------------------------
# 4. New Algo (비교용, 기존 코드 유지)
# ---------------------------------------------------------
def calculate_new_algo_power(current_time, active_evs, grid_capacity, max_ev_power, time_step):
    count = len(active_evs)
    if count == 0: return []
    
    durations = [max(time_step, float(ev.deadline) - current_time) for ev in active_evs]
    remains = [ev.remaining for ev in active_evs]
    allocation = [0.0] * count

    # Step 1: Mandatory
    for i in range(count):
        future_time = durations[i] - time_step
        max_future_charge = max(0.0, future_time * max_ev_power)
        mandatory_energy = max(0.0, remains[i] - max_future_charge)
        allocation[i] = min(mandatory_energy / time_step, min(max_ev_power, remains[i]/time_step))

    current_load = sum(allocation)
    remaining_grid = grid_capacity - current_load
    
    if current_load > grid_capacity: # Overload correction
        scale = grid_capacity / current_load
        return [p * scale for p in allocation]

    # Step 2: Minimax (Stress Equalization)
    if remaining_grid > EPSILON:
        residual_needs = []
        residual_caps = []
        for i in range(count):
             phys_limit = min(max_ev_power, remains[i]/time_step)
             residual_caps.append(max(0.0, phys_limit - allocation[i]))
             residual_needs.append(remains[i] - allocation[i]*time_step)
        
        if sum(residual_needs) > EPSILON:
            low, high = -100.0, 100.0
            best_extra = [0.0]*count
            for _ in range(15):
                stress = (low+high)/2.0
                prop_total = 0.0
                curr_prop = []
                for i in range(count):
                    # Stress based allocation target
                    target_e = residual_needs[i] - (stress * durations[i])
                    p = max(0.0, min(target_e/time_step, residual_caps[i]))
                    curr_prop.append(p)
                    prop_total += p
                if prop_total > remaining_grid: low = stress
                else: high = stress; best_extra = curr_prop
            
            for i in range(count):
                allocation[i] += best_extra[i]
                remaining_grid -= best_extra[i]

    # Step 3: Greedy
    if remaining_grid > EPSILON:
        urgency = []
        for i in range(count):
            phys_limit = min(max_ev_power, remains[i]/time_step)
            if phys_limit - allocation[i] > EPSILON:
                score = remains[i]/durations[i] if durations[i]>0 else 999
                urgency.append((score, i))
        urgency.sort(key=lambda x: x[0], reverse=True)
        for _, idx in urgency:
            if remaining_grid < EPSILON: break
            phys_limit = min(max_ev_power, remains[idx]/time_step)
            add = min(remaining_grid, phys_limit - allocation[idx])
            allocation[idx] += add
            remaining_grid -= add
            
    return allocation

# ---------------------------------------------------------
# 5. 시뮬레이션 엔진
# ---------------------------------------------------------
def run_simulation(ev_set: List[EVRequest], algorithm: str) -> bool:
    evs = copy.deepcopy(ev_set)
    current_time = 0.0
    finished_cnt = 0
    total_evs = len(evs)
    max_deadline = max(e.deadline for e in evs) if evs else 0

    while finished_cnt < total_evs:
        # 현재 충전소에 있고, 아직 충전이 덜 된 차량들
        ready_queue = [
            e for e in evs 
            if e.arrival <= current_time + EPSILON and e.remaining > EPSILON
        ]
        
        # Deadline Check
        for e in ready_queue:
            if current_time > float(e.deadline) + EPSILON:
                return False 

        allocated_powers = []
        
        if algorithm == 'sLLF':
            allocated_powers = calculate_sllf_power(
                current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP
            )
        elif algorithm == 'NEW_ALGO':
            allocated_powers = calculate_new_algo_power(
                current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP
            )
        else:
            # Traditional Heuristics
            if algorithm == 'EDF':
                ready_queue.sort(key=lambda x: (x.deadline, x.ev_id))
            elif algorithm == 'LLF':
                ready_queue.sort(key=lambda x: ((x.deadline - current_time) - (x.remaining/MAX_EV_POWER), x.ev_id))
            elif algorithm == 'FCFS':
                ready_queue.sort(key=lambda x: (x.arrival, x.ev_id))
            
            allocated_powers = [0.0] * len(ready_queue)
            current_used = 0.0
            for i, ev in enumerate(ready_queue):
                available = TOTAL_STATION_POWER - current_used
                rate = min(MAX_EV_POWER, available)
                rate = max(0.0, rate)
                allocated_powers[i] = rate
                current_used += rate

        # Apply Charging
        for i, ev in enumerate(ready_queue):
            charged = min(ev.remaining, allocated_powers[i] * TIME_STEP)
            ev.remaining -= charged
            if ev.remaining <= EPSILON: ev.remaining = 0.0

        finished_cnt = sum(1 for e in evs if e.remaining <= EPSILON)
        current_time += TIME_STEP
        
        if current_time > max_deadline + 20.0: return False

    return True

# ---------------------------------------------------------
# 6. 메인 실행
# ---------------------------------------------------------
congestion_levels = list(range(STRESS_START, STRESS_START+STRESS_NUM))
ratios = {k: [] for k in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']}

print(f"Comparison: sLLF vs NEW_ALGO (Identical Max Rate)")

for level in congestion_levels:
    wins = {k: 0 for k in ratios.keys()}
    
    for _ in range(TRIAL_NUM):
        ev_requests = generate_ev_set(level)
        if not ev_requests: continue

        if run_simulation(ev_requests, 'EDF'): wins['EDF'] += 1
        if run_simulation(ev_requests, 'LLF'): wins['LLF'] += 1
        if run_simulation(ev_requests, 'NEW_ALGO'): wins['NEW_ALGO'] += 1
        if run_simulation(ev_requests, 'sLLF'): wins['sLLF'] += 1
            
    for k in ratios:
        ratios[k].append(wins[k] / TRIAL_NUM)
    
    print(f"Level {level:2d}: EDF={ratios['EDF'][-1]:.2f}, LLF={ratios['LLF'][-1]:.2f}, "
          f"sLLF={ratios['sLLF'][-1]:.2f}, NEW={ratios['NEW_ALGO'][-1]:.2f}")

# ---------------------------------------------------------
# 7. 그래프 출력
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(congestion_levels, ratios['EDF'], marker='o', label='EDF', linestyle=':', color='gray', alpha=0.5)
plt.plot(congestion_levels, ratios['LLF'], marker='s', label='LLF', linestyle='--', color='blue', alpha=0.5)
plt.plot(congestion_levels, ratios['NEW_ALGO'], marker='^', label='NEW_ALGO', linestyle='-', color='green', linewidth=2)
plt.plot(congestion_levels, ratios['sLLF'], marker='*', label='sLLF', linestyle='-', color='red', linewidth=2)

plt.title('Performance Comparison (Approx. sLLF vs NEW_ALGO)', fontsize=14)
plt.xlabel('Congestion Level', fontsize=12)
plt.ylabel('Success Ratio', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(congestion_levels)
plt.show()