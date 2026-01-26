import os
import pickle
import copy
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
DATA_SAVE_PATH = "/Users/jaewoo/data/ev/cab/data"
TARGET_LEVEL = 4

TIME_STEP = 1
EPSILON = 1e-6

TOTAL_STATION_POWER = 2.9
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
    if count == 0: return [], []

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
        return phys_limits, ["FULL"] * count

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
    
    statuses = []
    for p, limit in zip(best_allocations, phys_limits):
        if p < EPSILON: statuses.append("WAIT")
        elif p >= limit - EPSILON: statuses.append("FULL")
        else: statuses.append("sLLF")

    return best_allocations, statuses

def calculate_new_algo_power(current_time, active_evs, grid_capacity, max_ev_power, time_step) -> Tuple[List[float], List[str]]:
    """
    [Revised Logic]
    Step 1: Mandatory (Laxity Priority)
    Step 2: Minimax Laxity (Target Laxity Maximization) - CHANGED HERE
    Step 3: Greedy (Laxity Priority)
    """
    count = len(active_evs)
    if count == 0: return [], []
    
    allocation = [0.0] * count
    status_codes = ["WAIT"] * count
    remaining_grid = grid_capacity
    
    # ---------------------------------------------------------
    # Step 0: 상태 변수 계산
    # ---------------------------------------------------------
    ev_status = []
    
    for i, ev in enumerate(active_evs):
        phys_limit = min(max_ev_power, ev.remaining / time_step)
        
        # Mandatory (정수 슬롯)
        time_to_deadline = float(ev.deadline) - current_time
        total_slots = math.ceil(time_to_deadline / time_step)
        future_slots = max(0, total_slots - 1)
        
        max_future_charge = future_slots * max_ev_power
        mandatory_energy = max(0.0, ev.remaining - max_future_charge)
        mandatory_power = mandatory_energy / time_step
        
        # Laxity
        charge_time_needed = ev.remaining / max_ev_power
        laxity = time_to_deadline - charge_time_needed
        
        ev_status.append({
            'idx': i,
            'phys_limit': phys_limit,
            'mandatory': mandatory_power,
            'laxity': laxity,
            'deadline': ev.deadline
        })

    # ---------------------------------------------------------
    # Step 1: Mandatory Allocation (Laxity Priority)
    # ---------------------------------------------------------
    ev_status.sort(key=lambda x: (x['laxity'], x['deadline']))
    
    for st in ev_status:
        idx = st['idx']
        if remaining_grid <= EPSILON: break
            
        needed = min(st['mandatory'], st['phys_limit'])
        if needed > EPSILON:
            actual = min(needed, remaining_grid)
            allocation[idx] += actual
            remaining_grid -= actual
            status_codes[idx] = "MAND"

    # ---------------------------------------------------------
    # Step 2: Minimax Laxity Optimization (NEW)
    # ---------------------------------------------------------
    # 남은 전력을 사용해 '최소 Laxity'를 최대한 끌어올리는 Target Laxity 탐색
    
    residual_caps = []
    current_rems_after_s1 = []
    time_dists = []
    has_capacity = False
    
    for i in range(count):
        ev = active_evs[i]
        # Step 1 이후 남은 에너지 (이 에너지에서 더 깎아야 Laxity가 올라감)
        curr_rem = ev.remaining - (allocation[i] * time_step)
        current_rems_after_s1.append(curr_rem)
        
        time_dist = float(ev.deadline) - current_time
        time_dists.append(time_dist)
        
        phys_limit = min(max_ev_power, ev.remaining / time_step)
        # 추가 할당 가능한 용량
        cap = max(0.0, phys_limit - allocation[i])
        residual_caps.append(cap)
        
        if cap > EPSILON: has_capacity = True

    if remaining_grid > EPSILON and has_capacity:
        # Laxity 범위 설정 (Binary Search)
        # Laxity가 클수록 좋음 (여유 많음). Target Laxity가 높으면 더 많은 충전 필요.
        low_L = -100.0  # 매우 급한 상태
        high_L = 100.0  # 매우 여유로운 상태
        best_extra = [0.0] * count
        best_L = -100.0
        
        for _ in range(30):
            mid_L = (low_L + high_L) / 2.0
            prop_total = 0.0
            curr_prop = []
            
            for i in range(count):
                # 목표 Laxity인 mid_L을 맞추기 위한 목표 잔여 에너지
                # L = TimeDist - (Rem / MaxPower)
                # Rem / MaxPower = TimeDist - L
                # Rem = MaxPower * (TimeDist - L)
                
                target_rem = max_ev_power * (time_dists[i] - mid_L)
                
                # 현재 잔여량(Step1 후)에서 목표 잔여량까지 줄여야 하는 에너지
                energy_needed = current_rems_after_s1[i] - target_rem
                power_needed = energy_needed / time_step
                
                # 물리적 한계 및 Step1 이후 남은 용량 내에서만 가능
                p = max(0.0, min(power_needed, residual_caps[i]))
                
                curr_prop.append(p)
                prop_total += p
            
            if prop_total > remaining_grid:
                # 전력이 부족함 -> Target Laxity가 너무 높음(욕심부림) -> 낮춰야 함
                high_L = mid_L
            else:
                # 전력이 남음 or 딱 맞음 -> Target Laxity를 더 높여볼 수 있음
                low_L = mid_L
                best_extra = curr_prop
                best_L = mid_L
        
        # 적용
        for i in range(count):
            if best_extra[i] > EPSILON:
                allocation[i] += best_extra[i]
                remaining_grid -= best_extra[i]
                # Status: Minimax에 의해 달성된 Laxity 표시
                status_codes[i] = f"MINI(L={best_L:.1f})"

    # ---------------------------------------------------------
    # Step 3: Greedy Allocation (Laxity Priority)
    # ---------------------------------------------------------
    if remaining_grid > EPSILON:
        urgency = []
        for i in range(count):
            phys_limit = min(max_ev_power, active_evs[i].remaining / time_step)
            room = phys_limit - allocation[i]
            if room > EPSILON:
                # 현재 상태 기준 Laxity 재계산
                curr_rem = active_evs[i].remaining - (allocation[i] * time_step)
                l = (float(active_evs[i].deadline) - current_time) - (curr_rem / max_ev_power)
                urgency.append((l, i))
        
        urgency.sort(key=lambda x: x[0]) # Laxity 작을수록 급함
        
        for _, idx in urgency:
            if remaining_grid <= EPSILON: break
            phys_limit = min(max_ev_power, active_evs[idx].remaining / time_step)
            room = phys_limit - allocation[idx]
            add = min(remaining_grid, room)
            
            allocation[idx] += add
            remaining_grid -= add
            status_codes[idx] = "GRDY"
            
    return allocation, status_codes

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

    while finished_cnt < total_evs:
        # 1. Ready Queue
        ready_queue = [
            e for e in evs 
            if e.arrival <= current_time + EPSILON and e.remaining > EPSILON
        ]
        
        # 2. Deadline Check
        for e in ready_queue:
            if current_time >= float(e.deadline) - EPSILON:
                if enable_log:
                    print(f"\n!!!! FAILURE DETECTED !!!!")
                    print(f"Time: {current_time} | EV{e.ev_id} Missed Deadline")
                    print(f"  -> Deadline: {e.deadline}, Remaining: {e.remaining:.2f}")
                return False 

        # 3. 알고리즘 실행
        allocated_powers = []
        statuses = []
        
        if ready_queue:
            if algorithm == 'sLLF':
                allocated_powers, statuses = calculate_sllf_power(
                    current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP
                )
            elif algorithm == 'NEW_ALGO':
                allocated_powers, statuses = calculate_new_algo_power(
                    current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP
                )
            else:
                allocated_powers = [0.0]*len(ready_queue)
                statuses = ["UNK"]*len(ready_queue)

            # --- 상세 로그 출력 ---
            if enable_log:
                print(f"-"*75)
                print(f"Time {current_time:5.1f} | Active EVs: {len(ready_queue)}")
                print(f"  {'EV_ID':<6} | {'Rem':<7} | {'DL':<4} | {'Mand':<5} | {'Alloc':<6} | {'Status':<12}")
                
                display_mand = []
                for ev in ready_queue:
                    time_to_d = float(ev.deadline) - current_time
                    slots = math.ceil(time_to_d / TIME_STEP)
                    fut = max(0, slots - 1)
                    mand = max(0.0, ev.remaining - (fut * MAX_EV_POWER))
                    display_mand.append(mand)

                total_alloc_step = sum(allocated_powers)
                
                for i, ev in enumerate(ready_queue):
                    p = allocated_powers[i]
                    m = display_mand[i]
                    st = statuses[i]
                    print(f"  EV{ev.ev_id:<4d} | {ev.remaining:6.2f}  | {ev.deadline:<4d} | {m:5.2f} | {p:6.2f} | {st:<12}")
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
            if enable_log: print("\n*** FAIL: Time Limit Exceeded ***")
            return False

    if enable_log: 
        print(f"\n{'='*20} [{algorithm}] SUCCESS {'='*20}")
    return True

# ---------------------------------------------------------
# 4. 메인 실행
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
    for i, ev_set in enumerate(level_dataset):
        sllf_result = run_simulation(ev_set, 'sLLF', enable_log=False)
        new_result = run_simulation(ev_set, 'NEW_ALGO', enable_log=False)
        
        if not sllf_result and  new_result:
            target_case_index = i
            print(f"\n!!! Found Target Case at Index {i} !!!")
            break
            
    if target_case_index != -1:
        target_ev_set = level_dataset[target_case_index]
        
        print("\n\n" + "#"*60)
        print("                RUNNING NEW_ALGO (DEBUG)                 ")
        print("#"*60)
        run_simulation(target_ev_set, 'NEW_ALGO', enable_log=True)
    else:
        print(f"\nNo case found where sLLF succeeds and NEW_ALGO fails.")