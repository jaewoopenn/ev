import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.optimize import linprog

# --- EV Class (동일) ---
class EV:
    def __init__(self, ev_id, arrival, deadline, demand, max_rate):
        self.id = ev_id
        self.arrival = arrival
        self.deadline = deadline
        self.demand = demand
        self.remaining_demand = demand
        self.max_rate = max_rate

    def get_laxity(self, current_time):
        if current_time >= self.deadline:
            return 0
        time_remaining = self.deadline - current_time
        if self.max_rate > 0:
            time_to_charge = self.remaining_demand / self.max_rate
        else:
            time_to_charge = float('inf')
        return time_remaining - time_to_charge

# --- 1. Offline LP Solver (동일) ---
def is_offline_feasible(evs, duration, P_limit):
    num_evs = len(evs)
    c = np.zeros(num_evs * duration)
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    for t in range(duration):
        row = np.zeros(num_evs * duration)
        for i in range(num_evs):
            row[i * duration + t] = 1
        A_ub.append(row)
        b_ub.append(P_limit)
        
    for i, ev in enumerate(evs):
        row = np.zeros(num_evs * duration)
        for t in range(duration):
            if ev.arrival <= t < ev.deadline:
                row[i * duration + t] = 1
        A_eq.append(row)
        b_eq.append(ev.demand)

    bounds = []
    for i in range(num_evs):
        for t in range(duration):
            if evs[i].arrival <= t < evs[i].deadline:
                bounds.append((0, evs[i].max_rate))
            else:
                bounds.append((0, 0))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return res.success

# --- 2. Random Generator (동일) ---
def generate_mixed_evs(num_evs, duration, power_limit):
    evs = []
    for i in range(num_evs):
        arrival = np.random.randint(0, int(duration * 0.6))
        if np.random.rand() < 0.5: 
            stay = np.random.randint(5, 10)
            factor = 0.9 
        else:
            stay = np.random.randint(15, 25)
            factor = 0.4
            
        deadline = min(arrival + stay, duration)
        max_rate = power_limit * 0.5 
        demand = max_rate * (deadline - arrival) * factor
        evs.append(EV(i, arrival, deadline, demand, max_rate))
    return evs

# --- Algorithms ---

def run_llf(evs_input, duration, P_limit):
    evs = copy.deepcopy(evs_input)
    for t in range(duration):
        active_evs = [ev for ev in evs if ev.arrival <= t < ev.deadline and ev.remaining_demand > 0.001]
        active_evs.sort(key=lambda x: x.get_laxity(t))
        P_remain = P_limit
        for ev in active_evs:
            rate = min(ev.max_rate, ev.remaining_demand, P_remain)
            ev.remaining_demand -= rate
            P_remain -= rate
    return evs

# [수정된 부분] Residual Filling이 추가된 sLLF
def run_sllf(evs_input, duration, P_limit):
    evs = copy.deepcopy(evs_input)
    for t in range(duration):
        active_evs = [ev for ev in evs if ev.arrival <= t < ev.deadline and ev.remaining_demand > 0.001]
        if not active_evs: continue

        # 1. Target Power Calculation
        sum_min_demand = sum([min(ev.max_rate, ev.remaining_demand) for ev in active_evs])
        target_P = min(P_limit, sum_min_demand)
        
        # 2. Binary Search for L
        L_min, L_max = -100.0, 100.0
        best_L = 0
        for _ in range(20): # 정밀도 향상
            L_mid = (L_min + L_max) / 2
            total_r = 0
            for ev in active_evs:
                l_i = ev.get_laxity(t)
                raw_rate = ev.max_rate * (L_mid - l_i + 1)
                rate = np.clip(raw_rate, 0, min(ev.max_rate, ev.remaining_demand))
                total_r += rate
            if total_r > target_P: L_max = L_mid
            else: L_min = L_mid
            best_L = L_mid
            
        # 3. Base Allocation
        current_step_allocation = {ev.id: 0.0 for ev in active_evs}
        total_allocated = 0
        
        for ev in active_evs:
            l_i = ev.get_laxity(t)
            raw_rate = ev.max_rate * (best_L - l_i + 1)
            rate = np.clip(raw_rate, 0, min(ev.max_rate, ev.remaining_demand))
            current_step_allocation[ev.id] = rate
            total_allocated += rate
            
        # 4. [핵심] Residual Filling (남은 전력 털어넣기)
        # 이진 탐색 오차로 인해 target_P보다 덜 썼다면, 남은 양을 비례 배분 혹은 Greedy하게 채움
        P_remain = target_P - total_allocated
        
        if P_remain > 0.001:
            # 더 받을 수 있는 여력이 있는 EV 찾기
            candidates = []
            for ev in active_evs:
                allocated = current_step_allocation[ev.id]
                max_possible = min(ev.max_rate, ev.remaining_demand)
                if allocated < max_possible:
                    candidates.append(ev)
            
            # 남은 전력을 후보들에게 공평하게 분배 (혹은 급한 순으로)
            # 여기서는 sLLF의 취지에 맞게 Laxity 순으로 채워넣음 (Hybrid approach)
            candidates.sort(key=lambda x: x.get_laxity(t))
            
            for ev in candidates:
                allocated = current_step_allocation[ev.id]
                max_possible = min(ev.max_rate, ev.remaining_demand)
                additional = min(max_possible - allocated, P_remain)
                current_step_allocation[ev.id] += additional
                P_remain -= additional
                if P_remain <= 0.0001: break
        
        # 5. Apply Allocation
        for ev in active_evs:
            rate = current_step_allocation[ev.id]
            ev.remaining_demand -= rate

    return evs

def is_successful(evs):
    return all(ev.remaining_demand < 0.01 for ev in evs)

# --- Main Experiment ---
STRESS_LEVELS = [4, 5, 6, 7, 8,9,10]
TRIALS = 100 # 통계적 유의성을 위해 횟수 유지
DURATION = 30
P_LIMIT = 12.0

llf_rates = []
sllf_rates = []
off_rates = []

print("Running Simulation with Optimized sLLF (Work-Conserving)...")

for n_ev in STRESS_LEVELS:
    llf_succ = 0
    sllf_succ = 0
    off_succ = 0
    valid_trials = 0
    
    while valid_trials < TRIALS:
        scenario = generate_mixed_evs(n_ev, DURATION, P_LIMIT)
        
        # if not is_offline_feasible(scenario, DURATION, P_LIMIT):
        #     continue 
            
        valid_trials += 1
        
        if is_successful(run_llf(scenario, DURATION, P_LIMIT)):
            llf_succ += 1
        
        if is_successful(run_sllf(scenario, DURATION, P_LIMIT)):
            sllf_succ += 1
        if  is_offline_feasible(scenario, DURATION, P_LIMIT):
            off_succ += 1

            
    llf_rate = llf_succ / TRIALS
    sllf_rate = sllf_succ / TRIALS
    off_rate = off_succ / TRIALS
    
    llf_rates.append(llf_rate)
    sllf_rates.append(sllf_rate)
    off_rates.append(sllf_rate)
    print(f"EVs {n_ev}: LLF={llf_rate:.2f}, sLLF={sllf_rate:.2f}")

# --- Plot ---
x = np.arange(len(STRESS_LEVELS))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, llf_rates, width/2, label='LLF', color='skyblue', hatch='///')
rects2 = ax.bar(x , sllf_rates, width/2, label='sLLF', color='salmon', alpha=0.9)
rects3 = ax.bar(x + width/2, off_rates, width/2, label='off', color='red', alpha=0.9)

ax.set_ylabel('Success Rate')
ax.set_title('Success Rate: LLF vs sLLF (Corrected)')
ax.set_xticks(x)
ax.set_xticklabels(STRESS_LEVELS)
ax.set_xlabel('Stress Level (Num EVs)')
ax.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()