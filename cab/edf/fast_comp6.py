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

TOTAL_STATION_POWER = 4
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
# 2. 알고리즘 로직
# ---------------------------------------------------------

def calculate_sllf_power(current_time, active_evs, grid_capacity, max_ev_power, time_step):
    count = len(active_evs)
    if count == 0: return []

    current_laxities = []
    phys_limits = []

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
    """
    [수정된 New Algorithm]
    1. Mandatory Allocation: Deadline 빠른 순으로 필수량 할당
    2. Minimax: 남은 전력 공평 분배
    3. Greedy: 자투리 전력 밀도 높은 순 할당
    """
    count = len(active_evs)
    if count == 0: return []
    
    durations = [max(time_step, float(ev.deadline) - current_time) for ev in active_evs]
    remains = [ev.remaining for ev in active_evs]
    phys_limits = [min(max_ev_power, r / time_step) for r in remains]
    
    allocation = [0.0] * count
    remaining_grid = grid_capacity

    # Step 1: Mandatory Allocation (Deadline Priority)
    mandatory_powers = []
    for i in range(count):
        future_time = durations[i] - time_step
        max_future_charge = max(0.0, future_time * max_ev_power)
        mandatory_energy = max(0.0, remains[i] - max_future_charge)
        mandatory_powers.append(mandatory_energy / time_step)

    # 데드라인 빠른 순 정렬
    sorted_indices = sorted(range(count), key=lambda i: active_evs[i].deadline)

    for idx in sorted_indices:
        if remaining_grid <= EPSILON: break
        needed = min(mandatory_powers[idx], phys_limits[idx])
        if needed > EPSILON:
            actual = min(needed, remaining_grid)
            allocation[idx] += actual
            remaining_grid -= actual

    # Step 2: Minimax
    residual_needs = []
    residual_caps = []
    has_needs = False
    for i in range(count):
        e_need = remains[i] - (allocation[i] * time_step)
        residual_needs.append(e_need)
        p_cap = max(0.0, phys_limits[i] - allocation[i])
        residual_caps.append(p_cap)
        if e_need > EPSILON: has_needs = True

    if remaining_grid > EPSILON and has_needs:
        low, high = -1000.0, 1000.0
        best_extra = [0.0] * count
        for _ in range(30):
            stress = (low + high) / 2.0
            prop_total = 0.0
            curr_prop = []
            for i in range(count):
                target_e = residual_needs[i] - (stress * durations[i])
                p = max(0.0, min(target_e / time_step, residual_caps[i]))
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
            room = phys_limits[i] - allocation[i]
            if room > EPSILON:
                score = remains[i] / durations[i] if durations[i] > 0 else 999.0
                urgency.append((score, i))
        urgency.sort(key=lambda x: x[0], reverse=True)
        for _, idx in urgency:
            if remaining_grid <= EPSILON: break
            room = phys_limits[idx] - allocation[idx]
            add = min(remaining_grid, room)
            allocation[idx] += add
            remaining_grid -= add
            
    return allocation

# ---------------------------------------------------------
# 3. 시뮬레이션 엔진 (Deadline Check 수정됨)
# ---------------------------------------------------------
def run_simulation(ev_set: List[EVRequest], algorithm: str) -> bool:
    evs = copy.deepcopy(ev_set)
    current_time = 0.0
    finished_cnt = 0
    total_evs = len(evs)
    
    # 시뮬레이션 종료 안전장치 (가장 늦은 데드라인 + 여유분)
    max_deadline = max(e.deadline for e in evs) if evs else 0

    while finished_cnt < total_evs:
        # 1. 큐 구성 (현재 시각 도착 & 잔여량 있음)
        ready_queue = [
            e for e in evs 
            if e.arrival <= current_time + EPSILON and e.remaining > EPSILON
        ]
        
        # 2. [수정됨] 데드라인 Miss 판정
        # 정의: current_time이 deadline에 도달했는데(>=), 
        # ready_queue에 존재한다면(remaining > 0) 실패.
        # 충전 로직 수행 *전*에 체크.
        for e in ready_queue:
            if current_time >= float(e.deadline) - EPSILON:
                return False 

        # 3. 전력 할당 계산
        allocated_powers = []
        if algorithm == 'sLLF':
            allocated_powers = calculate_sllf_power(current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP)
        elif algorithm == 'NEW_ALGO':
            allocated_powers = calculate_new_algo_power(current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP)
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

        # 4. 충전 수행
        for i, ev in enumerate(ready_queue):
            charged = min(ev.remaining, allocated_powers[i] * TIME_STEP)
            ev.remaining -= charged
            if ev.remaining <= EPSILON: ev.remaining = 0.0

        # 5. 상태 업데이트
        finished_cnt = sum(1 for e in evs if e.remaining <= EPSILON)
        current_time += TIME_STEP
        
        if current_time > max_deadline + 20.0: return False

    return True

# ---------------------------------------------------------
# 4. 병렬 작업 워커
# ---------------------------------------------------------
def worker_task_data(args):
    level, ev_requests = args
    result_vector = {}
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
    all_tasks = []
    
    print(f"Loading data from '{DATA_SAVE_PATH}' and preparing tasks...")

    for level in congestion_levels:
        filename = f"ev_level_{level}.pkl"
        full_path = os.path.join(DATA_SAVE_PATH, filename)
        try:
            with open(full_path, 'rb') as f:
                level_dataset = pickle.load(f)
                if not isinstance(level_dataset, list): continue
                for ev_set in level_dataset:
                    all_tasks.append((level, ev_set))
        except FileNotFoundError:
            print(f"Warning: {filename} not found.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print(f"Total Tasks Created: {len(all_tasks)}")
    print(f"Starting Simulation on {cpu_count()} Cores...")

    results_map = {level: {k: 0 for k in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']} for level in congestion_levels}
    trial_counts = {level: 0 for level in congestion_levels}

    with Pool(processes=cpu_count()) as pool:
        raw_results = pool.map(worker_task_data, all_tasks)
        
    for level, res in raw_results:
        trial_counts[level] += 1
        for algo, score in res.items():
            results_map[level][algo] += score

    ratios = {k: [] for k in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']}
    print(f"\n[Simulation Results - Success Rate]")
    
    for level in congestion_levels:
        count = trial_counts[level]
        if count == 0: 
            for k in ratios: ratios[k].append(0.0)
            continue
        print(f"Level {level:2d} (n={count}): ", end="")
        for algo in ['EDF', 'LLF', 'sLLF', 'NEW_ALGO']:
            rate = results_map[level][algo] / count
            ratios[algo].append(rate)
            print(f"{algo}={rate:.2f} ", end="")
        print("")

    print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")

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