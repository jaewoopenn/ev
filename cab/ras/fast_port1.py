import os
import pickle
import copy
import time
import matplotlib.pyplot as plt
import ras.fast_port_def1 as port
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

# [환경 제약: 이중 병목]
TOTAL_STATION_POWER = 4.8     # 제약 1: 전력 총량 (C)
MAX_EV_POWER = 1.0            # 단일 차량 속도 한계
MAX_SESSIONS = 6              # 제약 2: 세션(플러그) 수 (K)

STRESS_START = 0
STRESS_NUM = 12

@dataclass
class EVRequest:
    ev_id: int
    arrival: int
    required_energy: float
    deadline: int
    remaining: float

    def __repr__(self):
        return f"EV{self.ev_id}(Rem={self.remaining:.2f}, D={self.deadline})"



# ---------------------------------------------------------
# 3. 시뮬레이션 엔진 (Global Admission Control 적용)
# ---------------------------------------------------------
def run_simulation(ev_set: List[EVRequest], algorithm: str) -> bool:
    evs = copy.deepcopy(ev_set)
    current_time = 0.0
    finished_cnt = 0
    total_evs = len(evs)
    max_deadline = max(e.deadline for e in evs) if evs else 0

    while finished_cnt < total_evs:
        # 1. Ready Queue
        ready_queue = [e for e in evs if e.arrival <= current_time + EPSILON and e.remaining > EPSILON]
        
        # 2. Deadline Check
        for e in ready_queue:
            if current_time >= float(e.deadline) - EPSILON: return False 

        # 3. [핵심] Global Admission Control (Feasibility Optimality의 열쇠)
        # 세션(K)보다 대기차가 많으면, 누구를 들여보낼 것인가?
        active_evs = []
        
        if len(ready_queue) <= MAX_SESSIONS:
            active_evs = ready_queue
        else:
            # Feasibility Optimal Admission Rule = Least Laxity First
            # Laxity가 낮은(위험한) 순서대로 K명을 뽑아야 수학적으로 최적임.
            # 어떤 알고리즘을 쓰든 Admission은 LLF로 통일해야 공정한 비교가 됨 (혹은 알고리즘의 일부로 간주)
            
            # 단, 비교를 위해 baseline들은 자기만의 방식을 고집하게 둠
            if algorithm == 'EDF':
                # EDF는 Admission도 마감 순서로 하려 함 (이래서 최적성이 깨짐)
                ready_queue.sort(key=lambda x: (x.deadline, x.ev_id))
            elif algorithm == 'FCFS':
                ready_queue.sort(key=lambda x: (x.arrival, x.ev_id))
            else:
                # LLF, sLLF, NEW_ALGO(S-RAS)는 모두 LLF 기반 Admission을 사용
                # Laxity = (Deadline - Current) - (Remaining / MaxPower)
                ready_queue.sort(key=lambda x: (
                    (x.deadline - current_time) - (x.remaining / MAX_EV_POWER), 
                    x.ev_id
                ))
            
            active_evs = ready_queue[:MAX_SESSIONS]

        # 4. Power Allocation
        allocated_map = {}
        if algorithm == 'sLLF':
            powers = port.calculate_sllf_power(current_time, active_evs, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP)
            for i, ev in enumerate(active_evs): allocated_map[ev.ev_id] = powers[i]
            
        elif algorithm == 'NEW_ALGO':
            powers = port.calculate_optimal_ras_power(current_time, active_evs, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP)
            for i, ev in enumerate(active_evs): allocated_map[ev.ev_id] = powers[i]
            
        else:
            # Greedy Allocation for Baselines
            used = 0.0
            for ev in active_evs:
                p = max(0.0, min(MAX_EV_POWER, TOTAL_STATION_POWER - used))
                allocated_map[ev.ev_id] = p
                used += p

        # 5. Charging
        for ev in ready_queue:
            p = allocated_map.get(ev.ev_id, 0.0)
            charged = min(ev.remaining, p * TIME_STEP)
            ev.remaining -= charged
            if ev.remaining <= EPSILON: ev.remaining = 0.0

        finished_cnt = sum(1 for e in evs if e.remaining <= EPSILON)
        current_time += TIME_STEP
        if current_time > max_deadline + 50: return False

    return True

# ---------------------------------------------------------
# 4. 실행 및 결과 집계
# ---------------------------------------------------------
def worker_task_data(args):
    level, ev_requests = args
    result_vector = {}
    for algo in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']:
        result_vector[algo] = 1 if run_simulation(ev_requests, algo) else 0
    return level, result_vector

if __name__ == '__main__':
    start_time = time.time()
    
    if not os.path.exists(DATA_SAVE_PATH):
        print("Data path not found. Generating dummy...")
        # (Dummy generation code omitted for brevity)

    # 데이터 로딩
    all_tasks = []
    for level in range(STRESS_START, STRESS_START + STRESS_NUM):
        fname = os.path.join(DATA_SAVE_PATH, f"ev_level_{level}.pkl")
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                data = pickle.load(f)
                for evs in data: all_tasks.append((level, evs))

    print(f"Simulation Start (Max Sessions={MAX_SESSIONS})...")
    
    results = {l: {k: 0 for k in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']} for l in range(STRESS_START, STRESS_START + STRESS_NUM)}
    counts = {l: 0 for l in range(STRESS_START, STRESS_START + STRESS_NUM)}

    with Pool(cpu_count()) as pool:
        mapped = pool.map(worker_task_data, all_tasks)

    for level, res in mapped:
        counts[level] += 1
        for algo, val in res.items(): results[level][algo] += val

    # Print Results
    print(f"\n[Success Rate (K={MAX_SESSIONS})]")
    ratios = {k: [] for k in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']}
    levels = sorted(results.keys())
    
    for l in levels:
        print(f"Level {l:2d}: ", end="")
        for algo in ['EDF', 'LLF', 'sLLF', 'NEW_ALGO']:
            rate = results[l][algo] / counts[l] if counts[l] > 0 else 0
            ratios[algo].append(rate)
            print(f"{algo}={rate:.3f} ", end="")
        print("")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(levels, ratios['EDF'], marker='o', label='fluid-EDF', linestyle=':', color='gray', alpha=0.5)
    plt.plot(levels, ratios['LLF'], marker='s', label='fluid-LLF', linestyle='--', color='blue', alpha=0.5)
    plt.plot(levels, ratios['NEW_ALGO'], marker='^', label='NEW_ALGO', linestyle='-', color='green', linewidth=2)
    plt.plot(levels, ratios['sLLF'], marker='*', label='sLLF (Binary)', linestyle='-', color='red', linewidth=2)
    
    
    plt.title(f'Feasibility under Session Limit (K={MAX_SESSIONS})')
    plt.xlabel('Stress Level'); plt.ylabel('Success Rate')
    plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(-0.05, 1.05)
    plt.show()