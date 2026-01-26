import os
import pickle
import copy
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
from multiprocessing import Pool, cpu_count

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
DATA_SAVE_PATH = "/Users/jaewoo/data/ev/cab/data"

TRIAL_NUM = 100
TIME_STEP = 1
EPSILON = 1e-6

TOTAL_STATION_POWER = 3.0
MAX_EV_POWER = 1.0

STRESS_START = 0
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
# 2. 알고리즘 로직 (Safe Version 적용)
# ---------------------------------------------------------

def calculate_sllf_power(current_time, active_evs, grid_capacity, max_ev_power, time_step):
    count = len(active_evs)
    if count == 0: return []

    current_laxities = []
    phys_limits = []

    # 1. Laxity 및 물리적 제한 계산
    for ev in active_evs:
        remaining_time = ev.deadline - current_time
        time_to_charge = ev.remaining / max_ev_power
        l_t = remaining_time - time_to_charge
        current_laxities.append(l_t)
        p_limit = min(max_ev_power, ev.remaining / time_step)
        phys_limits.append(p_limit)

    if sum(phys_limits) <= grid_capacity + EPSILON:
        return phys_limits

    def get_total_power_for_target_L(target_L):
        total_p = 0.0
        allocs = []
        for i in range(count):
            req_p = (max_ev_power / time_step) * (target_L - current_laxities[i] + time_step)
            req_p = max(0.0, min(req_p, phys_limits[i]))
            allocs.append(req_p)
            total_p += req_p
        return total_p, allocs

    # [수정] 동적 범위 설정 (Dynamic Bounds)
    if not current_laxities:
        min_lax, max_lax = -100, 100
    else:
        min_lax = min(current_laxities)
        max_lax = max(current_laxities)
    
    low_L = min_lax - time_step * 5.0 
    high_L = max_lax + time_step * 5.0

    best_allocations = [0.0] * count
    
    for _ in range(30):
        mid_L = (low_L + high_L) / 2.0
        p_sum, p_allocs = get_total_power_for_target_L(mid_L)
        
        if p_sum > grid_capacity:
            high_L = mid_L
        else:
            low_L = mid_L
            best_allocations = p_allocs

    if sum(best_allocations) > grid_capacity + EPSILON:
        scale = grid_capacity / sum(best_allocations)
        best_allocations = [p * scale for p in best_allocations]
        
    return best_allocations

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
    
    if current_load > grid_capacity:
        scale = grid_capacity / current_load
        return [p * scale for p in allocation]

    # Step 2: Minimax (Safe Version)
    if remaining_grid > EPSILON:
        residual_needs = []
        residual_caps = []
        max_stress_candidates = []
        min_stress_candidates = []

        for i in range(count):
            phys_limit = min(max_ev_power, remains[i]/time_step)
            cap = max(0.0, phys_limit - allocation[i])
            need = remains[i] - allocation[i]*time_step
            residual_caps.append(cap)
            residual_needs.append(need)
            
            # [수정] 동적 범위 계산을 위한 후보 수집
            max_stress_candidates.append(need / durations[i])
            min_stress_candidates.append((need - cap * time_step) / durations[i])
        
        if sum(residual_needs) > EPSILON:
            # [수정] 동적 범위 적용
            low = min(min_stress_candidates) - 50.0 
            high = max(max_stress_candidates) + 50.0 
            
            best_extra = [0.0]*count
            for _ in range(25):
                stress = (low+high)/2.0
                prop_total = 0.0
                curr_prop = []
                for i in range(count):
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
# 3. 시뮬레이션 엔진
# ---------------------------------------------------------
def run_simulation(ev_set: List[EVRequest], algorithm: str) -> bool:
    evs = copy.deepcopy(ev_set)
    current_time = 0.0
    finished_cnt = 0
    total_evs = len(evs)
    max_deadline = max(e.deadline for e in evs) if evs else 0

    while finished_cnt < total_evs:
        ready_queue = [
            e for e in evs 
            if e.arrival <= current_time + EPSILON and e.remaining > EPSILON
        ]
        
        # 데드라인 체크 (하나라도 놓치면 실패)
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
            # Baseline Algorithms
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

        for i, ev in enumerate(ready_queue):
            charged = min(ev.remaining, allocated_powers[i] * TIME_STEP)
            ev.remaining -= charged
            if ev.remaining <= EPSILON: ev.remaining = 0.0

        finished_cnt = sum(1 for e in evs if e.remaining <= EPSILON)
        current_time += TIME_STEP
        
        if current_time > max_deadline + 20.0: return False

    return True

# ---------------------------------------------------------
# 4. 병렬 작업 워커 (수정됨)
# ---------------------------------------------------------
def worker_task_data(args):
    """
    (level, ev_requests) 튜플을 받아 시뮬레이션 실행
    파일 I/O를 여기서 하지 않고, 메인에서 로드된 데이터를 받아서 처리함
    """
    level, ev_requests = args
    result_vector = {}
    
    # 4가지 알고리즘 실행
    for algo in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']:
        success = run_simulation(ev_requests, algo)
        result_vector[algo] = 1 if success else 0
        
    return level, result_vector

# ---------------------------------------------------------
# 5. 메인 실행
# ---------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()
    
    if not os.path.exists(DATA_SAVE_PATH):
        print(f"Error: Data directory '{DATA_SAVE_PATH}' not found.")
        exit(1)

    congestion_levels = list(range(STRESS_START, STRESS_START + STRESS_NUM))
    
    # 작업 리스트 생성
    # [변경점] 메인 프로세스에서 파일(레벨) 단위로 읽어서 메모리에 올린 후 작업 분배
    all_tasks = []
    
    print(f"Loading data from '{DATA_SAVE_PATH}' and preparing tasks...")

    for level in congestion_levels:
        filename = f"ev_level_{level}.pkl"
        full_path = os.path.join(DATA_SAVE_PATH, filename)
        
        try:
            with open(full_path, 'rb') as f:
                # 이 파일 안에는 List[List[EVRequest]] 가 들어있음 (100개의 trial)
                level_dataset = pickle.load(f)
                
                # 데이터셋 유효성 체크
                if not isinstance(level_dataset, list):
                    print(f"Warning: Data in {filename} is not a list. Skipping.")
                    continue
                
                # 각 trial 데이터를 개별 작업으로 분리
                for ev_set in level_dataset:
                    all_tasks.append((level, ev_set))
                    
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Skipping level {level}.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print(f"Total Tasks Created: {len(all_tasks)}")
    print(f"Starting Simulation on {cpu_count()} Cores...")

    results_map = {level: {k: 0 for k in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']} for level in congestion_levels}
    trial_counts = {level: 0 for level in congestion_levels}

    # 병렬 처리 실행
    # 데이터가 메모리에 있으므로 Pool이 자동으로 피클링하여 워커에 전달
    with Pool(processes=cpu_count()) as pool:
        raw_results = pool.map(worker_task_data, all_tasks)
        
    # 결과 집계
    for level, res in raw_results:
        trial_counts[level] += 1
        for algo, score in res.items():
            results_map[level][algo] += score

    # 결과 출력 및 데이터 정리
    ratios = {k: [] for k in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']}
    print(f"\n[Simulation Results - Success Rate]")
    
    for level in congestion_levels:
        count = trial_counts[level]
        if count == 0: 
            # 데이터가 없으면 0으로 채움
            for k in ratios: ratios[k].append(0.0)
            continue
        
        print(f"Level {level:2d} (n={count}): ", end="")
        for algo in ['EDF', 'LLF', 'sLLF', 'NEW_ALGO']:
            rate = results_map[level][algo] / count
            ratios[algo].append(rate)
            print(f"{algo}={rate:.2f} ", end="")
        print("")

    print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")

    # ---------------------------------------------------------
    # 6. 그래프 출력
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(congestion_levels, ratios['EDF'], marker='o', label='EDF', linestyle=':', color='gray', alpha=0.5)
    plt.plot(congestion_levels, ratios['LLF'], marker='s', label='LLF', linestyle='--', color='blue', alpha=0.5)
    plt.plot(congestion_levels, ratios['NEW_ALGO'], marker='^', label='NEW_ALGO', linestyle='-', color='green', linewidth=2)
    plt.plot(congestion_levels, ratios['sLLF'], marker='*', label='sLLF', linestyle='-', color='red', linewidth=2)

    plt.title('Algorithm Performance Comparison', fontsize=14)
    plt.xlabel('Congestion Level (Stress)', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(congestion_levels)
    plt.ylim(-0.05, 1.05)
    plt.show()