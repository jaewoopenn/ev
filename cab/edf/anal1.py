import os
import pickle
import copy
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
DATA_SAVE_PATH = "/Users/jaewoo/data/ev/cab/data"
TARGET_LEVEL = 4  # 분석할 레벨

TIME_STEP = 1
EPSILON = 1e-6

TOTAL_STATION_POWER = 3.0
MAX_EV_POWER = 1.0

@dataclass
class EVRequest:
    ev_id: int
    arrival: int
    required_energy: float
    deadline: int
    remaining: float

    def __repr__(self):
        return f"EV{self.ev_id}"

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
# 3. 시뮬레이션 및 로깅
# ---------------------------------------------------------
def run_simulation(ev_set: List[EVRequest], algorithm: str, enable_log: bool = False) -> bool:
    evs = copy.deepcopy(ev_set)
    current_time = 0.0
    finished_cnt = 0
    total_evs = len(evs)
    max_deadline = max(e.deadline for e in evs) if evs else 0
    
    if enable_log:
        print(f"\n{'='*20} [{algorithm}] Simulation Start {'='*20}")
        print(f"Total EVs: {total_evs}, Max Deadline: {max_deadline}")

    while finished_cnt < total_evs:
        # 1. Ready Queue 구성
        ready_queue = [
            e for e in evs 
            if e.arrival <= current_time + EPSILON and e.remaining > EPSILON
        ]
        
        # 2. Deadline Check (충전 전 검사)
        for e in ready_queue:
            if current_time >= float(e.deadline) - EPSILON:
                if enable_log:
                    print(f"\n!!!! FAILURE DETECTED !!!!")
                    print(f"Time: {current_time} | EV{e.ev_id} Missed Deadline")
                    print(f"  -> Deadline: {e.deadline}, Remaining: {e.remaining:.2f}")
                return False 

        # 3. 알고리즘 실행
        allocated_powers = []
        if not ready_queue:
            if enable_log:
                print(f"Time {current_time:5.1f} | [Idle] No active EVs")
        else:
            if algorithm == 'sLLF':
                allocated_powers = calculate_sllf_power(
                    current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP
                )
            elif algorithm == 'NEW_ALGO':
                allocated_powers = calculate_new_algo_power(
                    current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP
                )
            
            # --- 상세 로그 출력 ---
            if enable_log:
                print(f"-"*60)
                print(f"Time {current_time:5.1f} | Active EVs: {len(ready_queue)}")
                print(f"  {'EV_ID':<6} | {'Rem':<7} | {'DL':<4} | {'Alloc':<6} | {'Status'}")
                
                total_alloc_step = sum(allocated_powers)
                
                for i, ev in enumerate(ready_queue):
                    p = allocated_powers[i]
                    status = "FULL" if p >= MAX_EV_POWER - EPSILON else f"{p:.2f}"
                    if p < EPSILON: status = "WAIT"
                    
                    print(f"  EV{ev.ev_id:<4d} | {ev.remaining:6.2f}  | {ev.deadline:<4d} | {p:6.2f} | {status}")
                print(f"  >> Total Grid Usage: {total_alloc_step:.2f} / {TOTAL_STATION_POWER:.1f}")

        # 4. 충전 적용
        for i, ev in enumerate(ready_queue):
            charged = min(ev.remaining, allocated_powers[i] * TIME_STEP)
            ev.remaining -= charged
            if ev.remaining <= EPSILON: ev.remaining = 0.0

        # 5. 시간 증가
        finished_cnt = sum(1 for e in evs if e.remaining <= EPSILON)
        current_time += TIME_STEP
        
        if current_time > max_deadline + 20.0:
            if enable_log: print("\n*** FAIL: Time Limit Exceeded (Global Timeout) ***")
            return False

    if enable_log: 
        print(f"\n{'='*20} [{algorithm}] SUCCESS {'='*20}")
    return True

# ---------------------------------------------------------
# 4. 메인 실행 (비교 로직)
# ---------------------------------------------------------
if __name__ == '__main__':
    filename = f"ev_level_{TARGET_LEVEL}.pkl"
    full_path = os.path.join(DATA_SAVE_PATH, filename)
    
    if not os.path.exists(full_path):
        print(f"Error: File {full_path} not found.")
        exit(1)
        
    print(f"Loading {filename}...")
    with open(full_path, 'rb') as f:
        level_dataset = pickle.load(f)
        
    print(f"Searching for a case where sLLF succeeds but NEW_ALGO fails in Level {TARGET_LEVEL}...")
    
    target_case_index = -1
    
    # 케이스 탐색
    for i, ev_set in enumerate(level_dataset):
        # 복사본으로 테스트 (로그 없이 빠르게)
        sllf_result = run_simulation(ev_set, 'sLLF', enable_log=False)
        new_result = run_simulation(ev_set, 'NEW_ALGO', enable_log=False)
        
        if sllf_result and not new_result:
            target_case_index = i
            print(f"\n!!! Found Target Case at Index {i} !!!")
            break
            
    if target_case_index != -1:
        target_ev_set = level_dataset[target_case_index]
        
        print("\n\n")
        print("############################################################")
        print("                RUNNING sLLF (SUCCESS CASE)                 ")
        print("############################################################")
        run_simulation(target_ev_set, 'sLLF', enable_log=True)
        
        print("\n\n")
        print("############################################################")
        print("                RUNNING NEW_ALGO (FAILURE CASE)             ")
        print("############################################################")
        run_simulation(target_ev_set, 'NEW_ALGO', enable_log=True)
        
    else:
        print(f"\nNo case found in Level {TARGET_LEVEL} where sLLF succeeds and NEW_ALGO fails.")