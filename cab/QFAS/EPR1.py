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
FIG_PATH = "/Users/jaewoo/data/ev/cab/result_epr.png" # 파일명을 epr로 변경
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
# 2. 알고리즘 로직 (기존과 동일)
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

def calculate_sras_power(current_time, active_evs, grid_capacity, max_ev_power, time_step):
    if not active_evs: return []

    segments = []
    first_seg_end = current_time + time_step
    
    segments.append({
        "start": current_time, 
        "end": first_seg_end, 
        "capacity": grid_capacity * time_step, 
        "index": 0 
    })
    
    deadlines = sorted(list(set(ev.deadline for ev in active_evs)))
    deadlines = [d for d in deadlines if d > first_seg_end]
    
    start_t = first_seg_end
    for d in deadlines:
        segments.append({
            "start": start_t, 
            "end": d, 
            "capacity": grid_capacity * (d - start_t), 
            "index": 1
        })
        start_t = d

    must_run_load = {ev.ev_id: 0.0 for ev in active_evs}
    
    sorted_evs_backward = sorted(active_evs, key=lambda x: x.deadline, reverse=True)
    temp_segments = copy.deepcopy(segments)
    
    for ev in sorted_evs_backward:
        energy_needed = ev.remaining
        
        for seg in reversed(temp_segments):
            if energy_needed <= EPSILON: break
            if seg["start"] >= ev.deadline: continue
            
            max_processable = max_ev_power * (seg["end"] - seg["start"])
            fill = min(energy_needed, seg["capacity"], max_processable)
            
            if fill > EPSILON:
                seg["capacity"] -= fill
                energy_needed -= fill
                
                if seg["index"] == 0: 
                    must_run_load[ev.ev_id] += fill

    final_allocations = {ev.ev_id: 0.0 for ev in active_evs}
    total_allocated = 0.0
    
    for ev in active_evs:
        p_req = must_run_load[ev.ev_id] / time_step
        final_allocations[ev.ev_id] = p_req
        total_allocated += p_req
        
    surplus = max(0.0, grid_capacity - total_allocated)
    
    if surplus > EPSILON:
        DANGER_THRESHOLD = max_ev_power * 0.99
        
        danger_zone = []
        safe_zone = []
        
        for ev in active_evs:
            if final_allocations[ev.ev_id] >= DANGER_THRESHOLD:
                danger_zone.append(ev)
            else:
                safe_zone.append(ev)
                
        danger_zone.sort(key=lambda x: x.deadline)
        for ev in danger_zone:
            if surplus <= EPSILON: break
            current_p = final_allocations[ev.ev_id]
            max_p = min(max_ev_power, ev.remaining / time_step)
            room = max_p - current_p
            
            if room > EPSILON:
                bonus = min(surplus, room)
                final_allocations[ev.ev_id] += bonus
                surplus -= bonus
                
        active_safe = safe_zone.copy()
        
        while surplus > EPSILON and active_safe:
            alpha = surplus / len(active_safe)
            next_active_safe = []
            progress_made = False
            
            for ev in active_safe:
                current_p = final_allocations[ev.ev_id]
                max_p = min(max_ev_power, ev.remaining / time_step)
                room = max_p - current_p
                
                if room > EPSILON:
                    bonus = min(alpha, room)
                    final_allocations[ev.ev_id] += bonus
                    surplus -= bonus
                    progress_made = True
                    
                    if max_p - final_allocations[ev.ev_id] > EPSILON:
                        next_active_safe.append(ev)
                        
            if not progress_made:
                break
            
            active_safe = next_active_safe

    return [final_allocations[ev.ev_id] for ev in active_evs]

# ---------------------------------------------------------
# 3. 시뮬레이션 엔진 (EPR 측정을 위해 대폭 수정됨)
# ---------------------------------------------------------
def run_simulation(ev_set: List[EVRequest], algorithm: str) -> float:
    evs = copy.deepcopy(ev_set)
    # 초기 총 요구 전력량 계산
    total_requested_energy = sum(e.required_energy for e in evs)
    
    if total_requested_energy <= EPSILON:
        return 1.0  # 요구량이 없으면 100% 충족으로 간주

    current_time = min(e.arrival for e in evs) if evs else 0.0
    max_time = max(e.deadline for e in evs) if evs else 0.0

    while current_time < max_time:
        # 1. 큐 구성: 도착했고, 데드라인이 남았고, 충전이 필요한 차량만 활성화
        ready_queue = [
            e for e in evs 
            if e.arrival <= current_time + EPSILON 
            and e.deadline > current_time + EPSILON 
            and e.remaining > EPSILON
        ]
        
        # 활성 차량이 없으면 시간을 다음 이벤트(가장 가까운 도착 시간)로 점프
        if not ready_queue:
            future_arrivals = [e.arrival for e in evs if e.arrival > current_time + EPSILON and e.remaining > EPSILON]
            if future_arrivals:
                current_time = min(future_arrivals)
                continue
            else:
                break # 더 이상 처리할 차량이 없으면 종료

        # 3. 전력 할당 계산
        allocated_powers = []
        if algorithm == 'sLLF':
            allocated_powers = calculate_sllf_power(current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP)
        elif algorithm == 'NEW_ALGO': 
            allocated_powers = calculate_sras_power(current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP)
        else:
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

        current_time += TIME_STEP

    # 5. EPR 산출: 시뮬레이션 종료 후 처리 못한 잔여 에너지를 뺀 값을 비율로 반환
    total_missed_energy = sum(e.remaining for e in evs)
    total_provided_energy = total_requested_energy - total_missed_energy
    
    epr = total_provided_energy / total_requested_energy
    return epr

# ---------------------------------------------------------
# 4. 병렬 작업 워커 (True/False 대신 EPR 실수값 반환)
# ---------------------------------------------------------
def worker_task_data(args):
    level, ev_requests = args
    result_vector = {}
    for algo in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']:
        epr_score = run_simulation(ev_requests, algo)
        result_vector[algo] = epr_score # 성공 여부가 아닌 연속적인 비율 저장
    return level, result_vector

# ---------------------------------------------------------
# 5. 메인 실행 (출력 라벨 수정)
# ---------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()
    
    if not os.path.exists(DATA_SAVE_PATH):
        print(f"Error: Data directory '{DATA_SAVE_PATH}' not found.")

    congestion_levels = list(range(STRESS_START, STRESS_START + STRESS_NUM))
    all_tasks = []
    
    print(f"Loading data from '{DATA_SAVE_PATH}' and preparing tasks...")

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
            pass

    if not all_tasks:
        print("No tasks loaded. Generating Mock Data for test...")
    
    print(f"Total Tasks Created: {len(all_tasks)}")
    print(f"Starting Simulation on {cpu_count()} Cores (I_min=0 mode)...")

    # 평균 EPR 합산을 위한 저장소
    results_map = {level: {k: 0.0 for k in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']} for level in congestion_levels}
    trial_counts = {level: 0 for level in congestion_levels}

    if all_tasks:
        with Pool(processes=cpu_count()) as pool:
            raw_results = pool.map(worker_task_data, all_tasks)
            
        for level, res in raw_results:
            trial_counts[level] += 1
            for algo, epr_score in res.items():
                results_map[level][algo] += epr_score

        ratios = {k: [] for k in ['EDF', 'LLF', 'NEW_ALGO', 'sLLF']}
        print(f"\n[Simulation Results - Average EPR (Energy Provision Ratio)]")
        
        for level in congestion_levels:
            count = trial_counts[level]
            if count == 0: 
                for k in ratios: ratios[k].append(0.0)
                continue
            print(f"Level {level:2d} (n={count}): ", end="")
            for algo in ['EDF', 'LLF', 'sLLF', 'NEW_ALGO']:
                avg_epr = results_map[level][algo] / count
                ratios[algo].append(avg_epr)
                print(f"{algo}={avg_epr:.3f} ", end="")
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

        plt.title('Algorithm Performance Comparison (Energy Provision Ratio)', fontsize=14)
        plt.xlabel('Congestion Level (Stress)', fontsize=12)
        plt.ylabel('Average EPR (0.0 ~ 1.0)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xticks(congestion_levels)
        plt.ylim(-0.05, 1.05)
        plt.savefig(FIG_PATH)