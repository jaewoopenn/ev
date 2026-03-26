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
FIG_PATH = "/Users/jaewoo/data/ev/cab/result_0.png" 
NAME_NEW_ALGO='Q-FAS'

TRIAL_NUM = 100
TIME_STEP = 1
EPSILON = 1e-6

# [환경 제약: 최소 충전 속도 없음 (I_min = 0)]
TOTAL_STATION_POWER = 4.8     # 총 전력량 (C)
MAX_EV_POWER = 1.0            # 단일 차량 최대 속도

# 시뮬레이션 레벨 설정
STRESS_START = 2
STRESS_NUM = 11


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


def calculate_fluid_edf_zl_power(current_time, ready_evs, grid_capacity, max_ev_power, min_ev_power, time_step):
    """
    [개선된 LGF 알고리즘: Binned Laxity + 가벼운 작업 우선(SRPT)]
    - 미세한 Laxity 차이로 인한 요동을 줄이기 위해 Laxity를 구간(Bin)으로 묶음.
    - 같은 위험도 구간 내에서는 빨리 끝낼 수 있는(Fluid Rate가 작은) 차량을 우선 할당하여
      시스템 이탈을 가속하고 전체 Success Ratio를 방어함.
    """
    if not ready_evs: return []
    
    allocations = {ev.ev_id: 0.0 for ev in ready_evs}
    surplus = grid_capacity
    
    ev_metrics = []
    
    # Bin 크기 설정 (기본적으로 1 time_step 단위를 하나의 '위험 구간'으로 묶음)
    bin_size = time_step 
    
    # 1. 모든 차량의 Fluid Rate, Laxity 및 Binned Laxity 계산
    for ev in ready_evs:
        time_to_deadline = ev.deadline - current_time
        max_req = min(max_ev_power, ev.remaining / time_step)
        
        if time_to_deadline <= EPSILON:
            fluid_rate = max_req
            laxity = -float('inf') # 초긴급 상태
            binned_laxity = -float('inf')
        else:
            fluid_rate = ev.remaining / time_to_deadline
            laxity = time_to_deadline - (ev.remaining / max_ev_power)
            
            # [핵심 변경 1] Laxity를 bin_size 단위로 내림 처리하여 같은 구간으로 그룹화
            binned_laxity = (laxity // bin_size) * bin_size
            
        fluid_rate = max(0.0, min(fluid_rate, max_req))
        
        ev_metrics.append({
            'ev_id': ev.ev_id,
            'laxity': laxity,           # 원본 값 (참고용/디버깅용)
            'binned_laxity': binned_laxity, # 정렬에 사용할 그룹화된 값
            'fluid_rate': fluid_rate,
            'max_req': max_req
        })
        
    # 2. [핵심 변경 2] 정렬: Binned Laxity 오름차순 (위험 구간별), Fluid Rate 오름차순 (가벼운 작업 우선)
    ev_metrics.sort(key=lambda x: (x['binned_laxity'], x['fluid_rate']))
    
    # 3. Phase 1 (기본 보장): 긴급한 차량부터 1순위로 Fluid Rate만큼 전력 할당
    for item in ev_metrics:
        if surplus <= EPSILON: break
        alloc = min(surplus, item['fluid_rate'])
        allocations[item['ev_id']] = alloc
        surplus -= alloc
        
    # 4. Phase 2 (잉여 가속): 단순성 원칙에 따라 기존의 직관적인 Greedy 몰아주기 방식 유지
    if surplus > EPSILON:
        for item in ev_metrics:
            if surplus <= EPSILON: break
            ev_id = item['ev_id']
            current_alloc = allocations[ev_id]
            room = item['max_req'] - current_alloc
            
            if room > EPSILON:
                add = min(surplus, room)
                allocations[ev_id] += add
                surplus -= add
                
    return [allocations[ev.ev_id] for ev in ready_evs]

def run_simulation(ev_set: List[EVRequest], algorithm: str) -> bool:
    evs = copy.deepcopy(ev_set)
    current_time = 0.0
    finished_cnt = 0
    total_evs = len(evs)
    
    max_deadline = max(e.deadline for e in evs) if evs else 0

    while finished_cnt < total_evs:
        # 1. 큐 구성
        ready_queue = [
            e for e in evs 
            if e.arrival <= current_time + EPSILON and e.remaining > EPSILON
        ]
        
        # 2. 데드라인 Miss 판정
        for e in ready_queue:
            if current_time >= float(e.deadline) - EPSILON:
                return False 

        # 3. 전력 할당 계산
        allocated_powers = []
        if algorithm == 'sLLF':
            allocated_powers = calculate_sllf_power(current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP)
        elif algorithm == 'NEW_ALGO': # NEW_ALGO = S-RAS
            allocated_powers = calculate_fluid_edf_zl_power(current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, 0, TIME_STEP)
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
        # exit(1) # 주석 처리 (코드 확인용)

    congestion_levels = list(range(STRESS_START, STRESS_START + STRESS_NUM))
    all_tasks = []
    
    print(f"Loading data from '{DATA_SAVE_PATH}' and preparing tasks...")

    # 실제 파일 로딩 부분 (경로가 유효해야 함)
    for level in congestion_levels:
        filename = f"ev_level_{level}.pkl"
        full_path = os.path.join(DATA_SAVE_PATH, filename)
        if os.path.exists(full_path):
            with open(full_path, 'rb') as f:
                level_dataset = pickle.load(f)
                if isinstance(level_dataset, list):
                    for ev_set in level_dataset:
                        all_tasks.append((level, ev_set))
        else:
            # print(f"Warning: {filename} not found.")
            pass

    if not all_tasks:
        print("No tasks loaded. Generating Mock Data for test...")
        # 테스트를 위해 임의의 데이터 생성 가능
    
    print(f"Total Tasks Created: {len(all_tasks)}")
    print(f"Starting Simulation on {cpu_count()} Cores (I_min=0 mode)...")

    results_map = {level: {k: 0 for k in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']} for level in congestion_levels}
    trial_counts = {level: 0 for level in congestion_levels}

    if all_tasks:
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
    else:
        print("Skipping simulation (no data).")
        ratios = {k: [] for k in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']}

    print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")

    if all_tasks:
        plt.figure(figsize=(10, 6))
        plt.plot(congestion_levels, ratios['EDF'], marker='o', label='EDF', linestyle=':', color='gray', alpha=0.5)
        plt.plot(congestion_levels, ratios['LLF'], marker='s', label='LLF', linestyle='--', color='blue', alpha=0.5)
        plt.plot(congestion_levels, ratios['NEW_ALGO'], marker='^', label=NAME_NEW_ALGO, linestyle='-', color='green', linewidth=2)
        plt.plot(congestion_levels, ratios['sLLF'], marker='*', label='sLLF', linestyle='-', color='red', linewidth=2)

        plt.title('Algorithm Performance Comparison (No Min Rate)', fontsize=14)
        plt.xlabel('Congestion Level (Stress)', fontsize=12)
        plt.ylabel('Success Rate', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xticks(congestion_levels)
        plt.ylim(-0.05, 1.05)
        plt.savefig(FIG_PATH)
        # plt.show()