import random
import copy
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
from multiprocessing import Pool, cpu_count  # 병렬 처리를 위한 임포트
import time

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
TRIAL_NUM = 200
# TRIAL_NUM=100
TIME_STEP = 0.1
EPSILON = 1e-6

TOTAL_STATION_POWER = 6
MAX_EV_POWER = 1.0

STRESS_START = 5
STRESS_NUM = 12


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
    
    # 1. Arrival Rate (도착 빈도)
    arrival_rate = 0.55 + (congestion_level * 0.075)

    # 2. Avg Energy (요구량)
    base_avg_energy = 2.5
    avg_energy = base_avg_energy + (congestion_level * 0.09)
    std_energy = 1.3 

    # 3. Slack Factor (여유 시간)
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
        
        # Slack Variance
        raw_slack = random.expovariate(1.0) * (min_required_time * slack_mean_factor)
        slack = int(raw_slack)
        
        deadline = arrival_int + min_required_time + slack
        
        ev_requests.append(EVRequest(ev_id, arrival_int, req_energy, deadline, req_energy))
        ev_id += 1
        
    return ev_requests

# ---------------------------------------------------------
# 3. New Algorithm Logic (유지)
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
    # 각 시뮬레이션마다 독립적인 복사본 사용 (사이드 이펙트 방지)
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
# [New] 병렬 처리를 위한 단일 작업 함수
# ---------------------------------------------------------
def run_single_trial(level):
    """
    하나의 trial에 대해 데이터셋을 생성하고,
    4가지 알고리즘을 모두 수행한 뒤 성공 여부 딕셔너리를 반환합니다.
    Worker Process에서 실행됩니다.
    """
    # 랜덤 시드는 각 프로세스마다 자동/독립적으로 관리됨
    ev_requests = generate_ev_set(level)
    
    # 생성된 EV가 없으면 모두 실패 처리 혹은 무시
    if not ev_requests:
        return {'EDF':0, 'LLF':0, 'FCFS':0, 'NEW_ALGO':0}

    result = {}
    result['EDF'] = 1 if run_simulation(ev_requests, 'EDF') else 0
    result['LLF'] = 1 if run_simulation(ev_requests, 'LLF') else 0
    result['FCFS'] = 1 if run_simulation(ev_requests, 'FCFS') else 0
    result['NEW_ALGO'] = 1 if run_simulation(ev_requests, 'NEW_ALGO') else 0
    
    return result

# ---------------------------------------------------------
# 5. 메인 실험 (병렬화 적용)
# ---------------------------------------------------------
if __name__ == '__main__':
    # 멀티프로세싱 사용 시, 반드시 __name__ == '__main__' 블록 내에서 실행해야 합니다.
    
    start_time_total = time.time()
    
    congestion_levels = list(range(STRESS_START, STRESS_START+STRESS_NUM))
    trials_per_level = TRIAL_NUM

    edf_ratios, llf_ratios, fcfs_ratios, new_ratios = [], [], [], []

    # 사용 가능한 CPU 코어 수 확인
    num_cores = cpu_count()
    print(f"EV Simulation (Fine-Grained) on {num_cores} Cores")
    print(f"Stress Levels: {congestion_levels}, Trials per level: {trials_per_level}")
    print("-" * 60)

    # 병렬 처리를 위한 Pool 생성
    # Pool은 한 번 만들어서 계속 재사용하는 것이 오버헤드가 적습니다.
    with Pool(processes=num_cores) as pool:
        
        for level in congestion_levels:
            st = time.time()
            
            # 작업을 병렬로 수행하기 위해 인자 리스트 생성
            # 예: [5, 5, 5, ..., 5] (200개)
            trial_args = [level] * trials_per_level
            
            # map 함수를 통해 병렬 실행 및 결과 수집
            results = pool.map(run_single_trial, trial_args)
            
            # 결과 집계
            wins = {'EDF':0, 'LLF':0, 'FCFS':0, 'NEW_ALGO':0}
            for res in results:
                wins['EDF'] += res['EDF']
                wins['LLF'] += res['LLF']
                wins['FCFS'] += res['FCFS']
                wins['NEW_ALGO'] += res['NEW_ALGO']

            # 비율 계산
            r_edf = wins['EDF'] / trials_per_level
            r_llf = wins['LLF'] / trials_per_level
            r_fcfs = wins['FCFS'] / trials_per_level
            r_new = wins['NEW_ALGO'] / trials_per_level
            
            edf_ratios.append(r_edf)
            llf_ratios.append(r_llf)
            fcfs_ratios.append(r_fcfs)
            new_ratios.append(r_new)
            
            elapsed = time.time() - st
            print(f"Level {level:2d}: EDF={r_edf:.2f}, LLF={r_llf:.2f}, FCFS={r_fcfs:.2f} | NEW={r_new:.2f} (Took {elapsed:.2f}s)")

    print("-" * 60)
    print(f"Total Simulation Time: {time.time() - start_time_total:.2f} seconds")

    # ---------------------------------------------------------
    # 6. 결과 그래프
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(congestion_levels, edf_ratios, marker='o', label='EDF', linestyle='-', color='blue', alpha=0.3)
    plt.plot(congestion_levels, llf_ratios, marker='s', label='LLF', linestyle='--', color='red', alpha=0.8)
    plt.plot(congestion_levels, fcfs_ratios, marker='^', label='FCFS', linestyle=':', color='green', alpha=0.3)
    plt.plot(congestion_levels, new_ratios, marker='*', label='NEW_ALGO', linestyle='-', linewidth=3, color='purple')

    plt.title(f'Performance Comparison (Parallelized)', fontsize=14)
    plt.xlabel('Stress Level', fontsize=12)
    plt.ylabel('Success Ratio', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(congestion_levels)
    plt.legend(fontsize=12)
    plt.show()