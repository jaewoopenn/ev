import random
import copy
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import multiprocessing  # [수정] 병렬 처리를 위한 모듈 임포트

# ---------------------------------------------------------
# 1. Configuration & Constants
# ---------------------------------------------------------
TRIAL_NUM = 200
TIME_STEP = 0.1         
EPSILON = 1e-6          

TOTAL_STATION_POWER = 6
MAX_EV_POWER = 1.0         

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
        return f"EV{self.ev_id}(A={self.arrival}, E={self.required_energy:.2f}, D={self.deadline})"

# ---------------------------------------------------------
# 2. EV Request Generator (12 Levels)
# ---------------------------------------------------------
def generate_ev_set(congestion_level):
    max_time = 60
    
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
        
        if current_time > max_time:
            break
            
        arrival_int = int(current_time)
        
        req_energy = random.gauss(avg_energy, std_energy)
        req_energy = max(0.5, min(8.0, req_energy))
        req_energy = round(req_energy, 2)
        
        min_required_time = math.ceil(req_energy / MAX_EV_POWER)
        
        raw_slack = random.expovariate(1.0) * (min_required_time * slack_mean_factor)
        slack = int(raw_slack)
        
        deadline = arrival_int + min_required_time + slack
        
        ev_requests.append(EVRequest(ev_id, arrival_int, req_energy, deadline, req_energy))
        ev_id += 1
        
    return ev_requests

# ---------------------------------------------------------
# 3. sLLF Algorithm Implementation (From PDF)
# ---------------------------------------------------------
def calculate_sllf_power(current_time, active_evs, grid_capacity, max_ev_power, time_step):
    count = len(active_evs)
    if count == 0: return []

    laxities = []
    max_demands = []
    
    for ev in active_evs:
        l_t = (ev.deadline - current_time) - (ev.remaining / max_ev_power)
        laxities.append(l_t)
        p_max = min(max_ev_power, ev.remaining / time_step)
        max_demands.append(p_max)

    if sum(max_demands) <= grid_capacity + EPSILON:
        return max_demands

    def get_total_power_for_L(L):
        total_p = 0.0
        allocations = []
        for i in range(count):
            req_power_unbounded = (max_ev_power / time_step) * (L - laxities[i] + time_step)
            p_i = max(0.0, min(max_demands[i], req_power_unbounded))
            allocations.append(p_i)
            total_p += p_i
        return total_p, allocations

    low_L = -100.0
    high_L = 100.0
    best_allocations = [0.0] * count
    
    for _ in range(30):
        mid_L = (low_L + high_L) / 2.0
        p_sum, p_allocs = get_total_power_for_L(mid_L)
        
        if p_sum > grid_capacity:
            high_L = mid_L
        else:
            low_L = mid_L
            best_allocations = p_allocs

    if sum(best_allocations) > grid_capacity + EPSILON:
        scale = grid_capacity / sum(best_allocations)
        best_allocations = [p * scale for p in best_allocations]
        
    return best_allocations

# ---------------------------------------------------------
# 4. New Algo Logic (Original)
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
# 5. Simulation Engine
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

        allocated_powers = []
        
        if algorithm == 'sLLF':
            allocated_powers = calculate_sllf_power(
                current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP
            )
            for i, ev in enumerate(ready_queue):
                charged = min(ev.remaining, allocated_powers[i] * TIME_STEP)
                ev.remaining -= charged
                if ev.remaining <= EPSILON: ev.remaining = 0.0
                
        elif algorithm == 'NEW_ALGO':
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
        
        if current_time > max_deadline + 20.0:
            return False

    return True

# [수정] 병렬 처리를 위한 단일 Trial 작업 함수 정의
# 이 함수는 Pool worker 내부에서 실행됩니다.
def run_single_trial(level):
    # 독립적인 EV set 생성 (각 프로세스마다 random 상태가 다름)
    ev_requests = generate_ev_set(level)
    
    # 결과 저장용 딕셔너리
    trial_result = {'EDF':0, 'LLF':0, 'FCFS':0, 'NEW_ALGO':0, 'sLLF':0}
    
    if not ev_requests:
        return trial_result

    # 각 알고리즘별 시뮬레이션 수행
    if run_simulation(ev_requests, 'EDF'): trial_result['EDF'] = 1
    if run_simulation(ev_requests, 'LLF'): trial_result['LLF'] = 1
    if run_simulation(ev_requests, 'FCFS'): trial_result['FCFS'] = 1
    if run_simulation(ev_requests, 'NEW_ALGO'): trial_result['NEW_ALGO'] = 1
    if run_simulation(ev_requests, 'sLLF'): trial_result['sLLF'] = 1
            
    return trial_result

# ---------------------------------------------------------
# 6. Main Experiment Execution
# ---------------------------------------------------------
if __name__ == '__main__':  # [수정] 멀티프로세싱 엔트리 포인트 보호
    congestion_levels = list(range(STRESS_START, STRESS_START+STRESS_NUM))
    edf_ratios, llf_ratios, fcfs_ratios, new_ratios, sllf_ratios = [], [], [], [], []
    trials_per_level = TRIAL_NUM

    print(f"EV Simulation: Comparing Heuristics, NEW_ALGO, and sLLF (Parallel Execution)")
    print(f"Using {multiprocessing.cpu_count()} CPU cores.")

    # [수정] Process Pool 생성
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for level in congestion_levels:
            # 해당 레벨에 대해 TRIAL_NUM 만큼 작업을 리스트로 만듦
            tasks = [level] * trials_per_level
            
            # 병렬 실행 (map)
            # results는 각 trial의 결과 딕셔너리 리스트가 됨
            results = pool.map(run_single_trial, tasks)
            
            # 결과 집계
            wins = {'EDF':0, 'LLF':0, 'FCFS':0, 'NEW_ALGO':0, 'sLLF':0}
            for res in results:
                for key in wins:
                    wins[key] += res[key]
                
            edf_ratios.append(wins['EDF'] / trials_per_level)
            llf_ratios.append(wins['LLF'] / trials_per_level)
            fcfs_ratios.append(wins['FCFS'] / trials_per_level)
            new_ratios.append(wins['NEW_ALGO'] / trials_per_level)
            sllf_ratios.append(wins['sLLF'] / trials_per_level)
            
            print(f"Level {level:2d}: EDF={edf_ratios[-1]:.2f}, LLF={llf_ratios[-1]:.2f}, sLLF={sllf_ratios[-1]:.2f}, NEW={new_ratios[-1]:.2f}")

    # ---------------------------------------------------------
    # 7. Plotting Results
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(congestion_levels, edf_ratios, marker='o', label='EDF', linestyle='-', color='blue', alpha=0.3)
    plt.plot(congestion_levels, llf_ratios, marker='s', label='LLF', linestyle='--', color='red', alpha=0.5)
    plt.plot(congestion_levels, fcfs_ratios, marker='^', label='FCFS', linestyle=':', color='green', alpha=0.3)
    plt.plot(congestion_levels, sllf_ratios, marker='D', label='sLLF', linestyle='-', color='orange', linewidth=2)
    plt.plot(congestion_levels, new_ratios, marker='*', label='NEW_ALGO', linestyle='-', linewidth=3, color='purple')

    plt.title(f'Scheduling Algorithm Performance Comparison', fontsize=14)
    plt.xlabel('Congestion / Stress Level', fontsize=12)
    plt.ylabel('Success Ratio (Feasibility)', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(congestion_levels)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()