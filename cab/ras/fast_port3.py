import os
import pickle
import copy
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
from multiprocessing import Pool, cpu_count
import ras.fast_port_def3 as port

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
# 경로 설정 (사용자 환경에 맞게 수정)
DATA_SAVE_PATH = "/Users/jaewoo/data/ev/cab/data" 
FIG_PATH = "/Users/jaewoo/data/ev/cab/result.png" 
NAME_NEW_ALGO='Q-FAS'


TRIAL_NUM = 100
TIME_STEP = 1
EPSILON = 1e-6


# [환경 제약: 최소 충전 속도 보장]
TOTAL_STATION_POWER = 4.8     # 총 전력량 (C)
MAX_EV_POWER = 1.0            # 단일 차량 최대 속도
MIN_CHARGING_RATE = 0.2       # 최소 충전 속도 제약 (I_min)

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
        return f"EV{self.ev_id}(Rem={self.remaining:.2f}, D={self.deadline})"


# ---------------------------------------------------------
# 2. Greedy Baseline Logic (EDF, LLF)
# ---------------------------------------------------------
def calculate_greedy_allocation(ready_evs, total_power, min_rate, max_rate, time_step, mode='EDF'):
    """
    Baseline 알고리즘 (EDF, LLF)
    - Admission Control 없이 Priority 순으로 I_min 이상 줄 수 있으면 할당.
    - 남는 전력은 Priority 순으로 채움 (Greedy).
    """
    if not ready_evs: return []
    
    # 1. 정렬 (Priority)
    current_time = 0 # LLF 계산용 (상대값 비교라 0이어도 순서는 동일)
    sorted_evs = []
    if mode == 'EDF':
        sorted_evs = sorted(ready_evs, key=lambda x: x.deadline)
    elif mode == 'LLF':
        sorted_evs = sorted(ready_evs, key=lambda x: (x.deadline - x.remaining/max_rate))
    
    allocations = {ev.ev_id: 0.0 for ev in ready_evs}
    used_power = 0.0
    
    # Step A: Activation (Min Rate 보장하며 순차적 활성화)
    active_evs = []
    for ev in sorted_evs:
        # 물리적 요구량
        phys_req = min(max_rate, ev.remaining / time_step)
        # 최소 제약 적용 (물리적 요구량이 min_rate보다 작으면 그것만, 아니면 min_rate)
        min_req = min(min_rate, phys_req)
        
        # I_min 제약: 최소 min_rate를 줄 수 있는 경우에만 활성화
        # 단, 배터리가 거의 꽉 차서 min_rate보다 덜 필요한 경우는 예외 허용
        req_to_activate = min_req
        
        if used_power + req_to_activate <= total_power + EPSILON:
            allocations[ev.ev_id] = req_to_activate
            used_power += req_to_activate
            active_evs.append(ev)
        else:
            # 전력 부족으로 대기
            allocations[ev.ev_id] = 0.0
            
    # Step B: Surplus Filling (활성화된 차량에 대해 Greedy하게 가속)
    surplus = max(0.0, total_power - used_power)
    for ev in active_evs:
        if surplus <= EPSILON: break
        
        curr = allocations[ev.ev_id]
        limit = min(max_rate, ev.remaining / time_step)
        room = limit - curr
        
        if room > EPSILON:
            add = min(surplus, room)
            allocations[ev.ev_id] += add
            surplus -= add
            
    return [allocations[ev.ev_id] for ev in ready_evs]


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
        # 1. Ready Queue: 도착했으면서 아직 완료되지 않은 모든 차량
        # 기존의 Global Admission Control 삭제 -> 모든 대기 차량을 알고리즘에 전달
        ready_queue = [e for e in evs if e.arrival <= current_time + EPSILON and e.remaining > EPSILON]
        
        # 2. Deadline Check (Fail Condition)
        for e in ready_queue:
            if current_time >= float(e.deadline) - EPSILON: return False 

        # 3. Power Allocation
        allocated_powers = []
        
        if algorithm == 'S-RAS':
            # S-RAS: 문서 로직에 따라 필수부하/Imin 고려하여 할당량 결정
            allocated_powers = port.calculate_sras_power(
                current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, MIN_CHARGING_RATE, TIME_STEP
            )
            
        elif algorithm in ['EDF', 'LLF']:
            # Baseline: Priority 순서대로 Imin 챙겨주고 남으면 더 줌
            allocated_powers = calculate_greedy_allocation(
                ready_queue, TOTAL_STATION_POWER, MIN_CHARGING_RATE, MAX_EV_POWER, TIME_STEP, mode=algorithm
            )
        
        # Map back to EV IDs
        allocated_map = {}
        for i, ev in enumerate(ready_queue):
            allocated_map[ev.ev_id] = allocated_powers[i]

        # 4. Charging Update
        for ev in ready_queue:
            p = allocated_map.get(ev.ev_id, 0.0)
            
            # 충전량 반영
            charged = min(ev.remaining, p * TIME_STEP)
            ev.remaining -= charged
            if ev.remaining <= EPSILON: ev.remaining = 0.0

        finished_cnt = sum(1 for e in evs if e.remaining <= EPSILON)
        current_time += TIME_STEP
        
        # 무한 루프 방지 (모든 데드라인을 한참 지났는데도 안 끝남)
        if current_time > max_deadline + 50: return False

    return True

# ---------------------------------------------------------
# 4. 실행 및 결과 집계
# ---------------------------------------------------------
def worker_task_data(args):
    level, ev_requests = args
    result_vector = {}
    # 비교 알고리즘 목록: EDF, LLF, S-RAS (sLLF 삭제됨)
    for algo in ['EDF', 'LLF', 'S-RAS']:
        result_vector[algo] = 1 if run_simulation(ev_requests, algo) else 0
    return level, result_vector

if __name__ == '__main__':
    start_time = time.time()
    
    if not os.path.exists(DATA_SAVE_PATH):
        print("Data path not found. Please check path.")

    # 데이터 로딩
    all_tasks = []
    print("Loading Data...")
    for level in range(STRESS_START, STRESS_START + STRESS_NUM):
        fname = os.path.join(DATA_SAVE_PATH, f"ev_level_{level}.pkl")
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                data = pickle.load(f)
                for evs in data: 
                    all_tasks.append((level, evs))
        else:
            print(f"File not found: {fname}")

    print(f"Simulation Start with MinRate={MIN_CHARGING_RATE}kW (Total Cap={TOTAL_STATION_POWER}kW)...")
    
    # 결과 저장소
    results = {l: {k: 0 for k in ['EDF', 'LLF', 'S-RAS']} for l in range(STRESS_START, STRESS_START + STRESS_NUM)}
    counts = {l: 0 for l in range(STRESS_START, STRESS_START + STRESS_NUM)}

    # 멀티프로세싱 실행
    with Pool(cpu_count()) as pool:
        mapped = pool.map(worker_task_data, all_tasks)

    # 결과 집계
    for level, res in mapped:
        counts[level] += 1
        for algo, val in res.items(): results[level][algo] += val

    # Print Results
    print(f"\n[Success Rate (MinRate={MIN_CHARGING_RATE})]")
    ratios = {k: [] for k in ['EDF', 'LLF', 'S-RAS']}
    levels = sorted([l for l in results.keys() if counts[l] > 0])
    
    for l in levels:
        print(f"Level {l:2d} (n={counts[l]}): ", end="")
        for algo in ['EDF', 'LLF', 'S-RAS']:
            rate = results[l][algo] / counts[l] if counts[l] > 0 else 0
            ratios[algo].append(rate)
            print(f"{algo}={rate:.3f} ", end="")
        print("")

    # Plotting
    if levels:
        plt.figure(figsize=(10, 6))
        plt.plot(levels, ratios['EDF'], marker='o', label='EDF', linestyle=':', color='gray', alpha=0.5)
        plt.plot(levels, ratios['LLF'], marker='s', label='LLF', linestyle='--', color='blue', alpha=0.5)
        plt.plot(levels, ratios['S-RAS'], marker='^', label=NAME_NEW_ALGO, linestyle='-', color='green', linewidth=2)
        
        plt.title(f'Feasibility under Min Charging Rate ({MIN_CHARGING_RATE}kW)')
        plt.xlabel('Stress Level')
        plt.ylabel('Success Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        plt.savefig(FIG_PATH)
        # plt.show()
    else:
        print("No data to plot.")