import os
import pickle
import copy
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
from multiprocessing import Pool, cpu_count
import ras.fast_port_def2 as port


# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
# 경로 설정 (사용자 환경에 맞게 수정)
DATA_SAVE_PATH = "/Users/jaewoo/data/ev/cab/data" 
FIG_PATH = "/Users/jaewoo/data/ev/cab/result.png" 

TRIAL_NUM = 100
TIME_STEP = 1
EPSILON = 1e-6

# [환경 제약 수정: 최소 충전 속도 보장]
TOTAL_STATION_POWER = 4.8     # 총 전력량 (C)
MAX_EV_POWER = 1.0            # 단일 차량 최대 속도
MIN_CHARGING_RATE = 0.2       # [NEW] 최소 충전 속도 제약 (r_min)

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
# 3. 시뮬레이션 엔진 (Admission Control Updated)
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
        
        # 2. Deadline Check (Fail Condition)
        for e in ready_queue:
            if current_time >= float(e.deadline) - EPSILON: return False 

        # 3. [수정됨] Global Admission Control (Min Rate Constraint)
        # 제약 조건: Active된 모든 차량에게 MIN_CHARGING_RATE를 줄 수 있어야 함.
        # 즉, Count * MIN_CHARGING_RATE <= TOTAL_STATION_POWER
        
        max_concurrent_evs = int(TOTAL_STATION_POWER // MIN_CHARGING_RATE)
        
        active_evs = []
        
        if len(ready_queue) <= max_concurrent_evs:
            active_evs = ready_queue
        else:
            # 수용 가능 인원을 초과하면 Admission Rule 적용
            # Feasibility Optimal Admission = Least Laxity First
            # (어떤 알고리즘이든 '누구를 들여보낼지'는 가장 급한 놈을 들여보내는게 전체 성공률에 유리함)
            
            if algorithm == 'EDF':
                # Baseline: EDF는 보통 마감 순으로 줄세움
                ready_queue.sort(key=lambda x: (x.deadline, x.ev_id))
            elif algorithm == 'FCFS':
                ready_queue.sort(key=lambda x: (x.arrival, x.ev_id))
            else:
                # LLF, sLLF, NEW_ALGO는 LLF Admission 사용
                ready_queue.sort(key=lambda x: (
                    (x.deadline - current_time) - (x.remaining / MAX_EV_POWER), 
                    x.ev_id
                ))
            
            active_evs = ready_queue[:max_concurrent_evs]

        # 4. Power Allocation
        allocated_map = {}
        if algorithm == 'sLLF':
            powers = port.calculate_sllf_power(
                current_time, active_evs, TOTAL_STATION_POWER, MAX_EV_POWER, MIN_CHARGING_RATE, TIME_STEP
            )
            for i, ev in enumerate(active_evs): allocated_map[ev.ev_id] = powers[i]
            
        elif algorithm == 'NEW_ALGO':
            powers = port.calculate_optimal_ras_power(
                current_time, active_evs, TOTAL_STATION_POWER, MAX_EV_POWER, MIN_CHARGING_RATE, TIME_STEP
            )
            for i, ev in enumerate(active_evs): allocated_map[ev.ev_id] = powers[i]
            
        else:
            # Baseline (EDF/LLF/FCFS) - Greedy Allocation
            # 단, Baseline들도 Min Rate 제약은 지켜야 함 (Policy Compliance)
            # Greedy: 급한 순서대로 Max Power 주고, 남는거 다음 사람... 
            # 하지만 Min Rate 제약 하에서는 "모두에게 Min 주고 시작"이 맞음.
            
            # 1. Base Allocation (Min Rate)
            temp_alloc = {}
            used_power = 0.0
            
            # 정렬 순서는 알고리즘 특성 따름
            sorted_active = []
            if algorithm == 'EDF': sorted_active = sorted(active_evs, key=lambda x: x.deadline)
            elif algorithm == 'LLF': sorted_active = sorted(active_evs, key=lambda x: (x.deadline - current_time) - x.remaining/MAX_EV_POWER)
            else: sorted_active = active_evs # FCFS etc
            
            # Step A: Min Rate 보장
            for ev in sorted_active:
                req = min(MIN_CHARGING_RATE, ev.remaining/TIME_STEP)
                temp_alloc[ev.ev_id] = req
                used_power += req
                
            # Step B: Surplus Greedy Allocation
            surplus = max(0.0, TOTAL_STATION_POWER - used_power)
            for ev in sorted_active:
                current_p = temp_alloc[ev.ev_id]
                max_p = min(MAX_EV_POWER, ev.remaining/TIME_STEP)
                room = max_p - current_p
                
                add = min(surplus, room)
                temp_alloc[ev.ev_id] += add
                surplus -= add
                
            allocated_map = temp_alloc

        # 5. Charging Update
        for ev in ready_queue:
            # Active가 아닌 대기 차량은 충전량 0
            if ev not in active_evs:
                continue
                
            p = allocated_map.get(ev.ev_id, 0.0)
            
            # 안전장치: Min Rate 제약 검증 (마지막 타임스텝 제외)
            # if p < MIN_CHARGING_RATE - EPSILON and ev.remaining > MIN_CHARGING_RATE * TIME_STEP:
                 # print(f"Warning: Min Rate Violation {p}")
            
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
    # 비교 알고리즘 목록
    for algo in ['EDF', 'LLF', 'sLLF', 'NEW_ALGO']:
        result_vector[algo] = 1 if run_simulation(ev_requests, algo) else 0
    return level, result_vector

if __name__ == '__main__':
    start_time = time.time()
    
    # 더미 데이터 생성 (파일이 없을 경우)
    if not os.path.exists(DATA_SAVE_PATH):
        print("Data path not found. Please check path.")
        # 여기서 더미 데이터를 생성하거나 종료할 수 있습니다.
        # 이 코드는 데이터가 있다고 가정하고 진행합니다.

    # 데이터 로딩
    all_tasks = []
    print("Loading Data...")
    for level in range(STRESS_START, STRESS_START + STRESS_NUM):
        fname = os.path.join(DATA_SAVE_PATH, f"ev_level_{level}.pkl")
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                data = pickle.load(f)
                # 데이터 구조에 따라 수정 필요 (List[List[EVRequest]])
                for evs in data: 
                    all_tasks.append((level, evs))
        else:
            print(f"File not found: {fname}")

    print(f"Simulation Start with MinRate={MIN_CHARGING_RATE}kW (Total Cap={TOTAL_STATION_POWER}kW)...")
    
    # 결과 저장소
    results = {l: {k: 0 for k in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']} for l in range(STRESS_START, STRESS_START + STRESS_NUM)}
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
    ratios = {k: [] for k in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']}
    levels = sorted([l for l in results.keys() if counts[l] > 0])
    
    for l in levels:
        print(f"Level {l:2d} (n={counts[l]}): ", end="")
        for algo in ['EDF', 'LLF', 'sLLF', 'NEW_ALGO']:
            rate = results[l][algo] / counts[l] if counts[l] > 0 else 0
            ratios[algo].append(rate)
            print(f"{algo}={rate:.3f} ", end="")
        print("")

    # Plotting
    if levels:
        plt.figure(figsize=(10, 6))
        plt.plot(levels, ratios['EDF'], marker='o', label='EDF', linestyle=':', color='gray', alpha=0.5)
        plt.plot(levels, ratios['LLF'], marker='s', label='LLF', linestyle='--', color='blue', alpha=0.5)
        plt.plot(levels, ratios['sLLF'], marker='*', label='sLLF', linestyle='-', color='red', linewidth=2)
        plt.plot(levels, ratios['NEW_ALGO'], marker='^', label='S-RAS', linestyle='-', color='green', linewidth=2)
        
        plt.title(f'Feasibility under Min Charging Rate ({MIN_CHARGING_RATE}kW)')
        plt.xlabel('Stress Level')
        plt.ylabel('Success Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        plt.show()
        # plt.savefig(FIG_PATH)
    else:
        print("No data to plot.")