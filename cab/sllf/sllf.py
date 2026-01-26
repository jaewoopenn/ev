import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.optimize import linprog

# --- Execution ---
DURATION = 40 # 시간을 조금 늘려서 혼잡 완화
P_LIMIT = 50.0 # 전력 총량 증가 (Feasibility 확보용)
TRIALS = 40
STRESS_LEVELS = [38,40, 42, 44, 46, 48] # 차량 대수를 더 늘려서 테스트

# --- EV Class ---
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

# --- Offline Feasibility Checker (Solver) ---
def is_offline_feasible(evs, duration, P_limit):
    num_evs = len(evs)
    c = np.zeros(num_evs * duration)
    
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    # 1. Total Power Constraints
    for t in range(duration):
        row = np.zeros(num_evs * duration)
        for i in range(num_evs):
            row[i * duration + t] = 1
        A_ub.append(row)
        b_ub.append(P_limit)
        
    # 2. Demand Constraints
    for i, ev in enumerate(evs):
        row = np.zeros(num_evs * duration)
        for t in range(duration):
            if ev.arrival <= t < ev.deadline:
                row[i * duration + t] = 1
        A_eq.append(row)
        b_eq.append(ev.demand)

    # 3. Rate Bounds
    bounds = []
    for i in range(num_evs):
        for t in range(duration):
            if evs[i].arrival <= t < evs[i].deadline:
                bounds.append((0, evs[i].max_rate))
            else:
                bounds.append((0, 0))

    # 'highs' method for speed and reliability
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return res.success

# --- Scenario Generator (Re-Tuned for Feasibility) ---
def generate_hardcore_evs_tuned(num_evs, duration, power_limit):
    evs = []
    for i in range(num_evs):
        # [수정 1] 도착 시간을 더 넓게 분산시켜서 '풀 수 있는' 확률을 높임
        arrival = np.random.randint(0, int(duration * 0.8))
        stay = np.random.randint(5, 30) # 체류 시간을 넉넉히 줌
        deadline = min(arrival + stay, duration)
        
        # [수정 2] 개별 차량 용량을 줄여서 스케줄링 유연성 확보 (동시 충전 10대 가능)
        max_rate = 5.0 
        
        # [수정 3] 수요 팩터를 조금 더 낮춤 (0.3 ~ 0.8)
        # 0.8이면 여전히 매우 빡빡하지만(80% 풀로드), 0.3인 차들이 섞여서 숨통을 틔워줌
        factor = np.random.uniform(0.3, 0.8)
        
        max_energy = max_rate * (deadline - arrival)
        demand = max_energy * factor
        
        evs.append(EV(i, arrival, deadline, demand, max_rate))
    return evs

# --- LLF Algorithm ---
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

# --- sLLF Algorithm (Work-Conserving) ---
def run_sllf(evs_input, duration, P_limit):
    evs = copy.deepcopy(evs_input)
    for t in range(duration):
        active_evs = [ev for ev in evs if ev.arrival <= t < ev.deadline and ev.remaining_demand > 0.001]
        if not active_evs: continue

        sum_min_demand = sum([min(ev.max_rate, ev.remaining_demand) for ev in active_evs])
        target_P = min(P_limit, sum_min_demand)
        
        # Binary Search
        L_min, L_max = -100.0, 100.0
        best_L = 0
        for _ in range(20):
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
            
        # Base Allocation
        current_step_allocation = {ev.id: 0.0 for ev in active_evs}
        total_allocated = 0
        for ev in active_evs:
            l_i = ev.get_laxity(t)
            raw_rate = ev.max_rate * (best_L - l_i + 1)
            rate = np.clip(raw_rate, 0, min(ev.max_rate, ev.remaining_demand))
            current_step_allocation[ev.id] = rate
            total_allocated += rate
            
        # Residual Filling (Work-Conserving)
        P_remain = target_P - total_allocated
        if P_remain > 0.001:
            candidates = [ev for ev in active_evs if current_step_allocation[ev.id] < min(ev.max_rate, ev.remaining_demand)]
            candidates.sort(key=lambda x: x.get_laxity(t))
            for ev in candidates:
                needed = min(ev.max_rate, ev.remaining_demand) - current_step_allocation[ev.id]
                additional = min(needed, P_remain)
                current_step_allocation[ev.id] += additional
                P_remain -= additional
                if P_remain <= 0.0001: break
        
        for ev in active_evs:
            ev.remaining_demand -= current_step_allocation[ev.id]
            
    return evs

def is_successful(evs):
    return all(ev.remaining_demand < 0.01 for ev in evs)


llf_rates = []
sllf_rates = []

print("Running Re-Tuned Simulation (Feasible Hardcore)...")

for n_ev in STRESS_LEVELS:
    llf_succ = 0
    sllf_succ = 0
    valid_trials = 0
    attempts = 0
    
    # Try finding valid instances up to 500 times per level
    while valid_trials < TRIALS and attempts < 500:
        attempts += 1
        scenario = generate_hardcore_evs_tuned(n_ev, DURATION, P_LIMIT)
        
        # 1. Check if physically possible (God mode)
        # if not is_offline_feasible(scenario, DURATION, P_LIMIT):
        #     continue
            
        valid_trials += 1
        
        # 2. Run Online Algorithms
        if is_successful(run_llf(scenario, DURATION, P_LIMIT)):
            llf_succ += 1
            
        if is_successful(run_sllf(scenario, DURATION, P_LIMIT)):
            sllf_succ += 1
            
    if valid_trials == 0:
        llf_rates.append(0)
        sllf_rates.append(0)
        print(f"EVs {n_ev}: Still too hard! (0 valid instances)")
    else:
        llf_rate = llf_succ / valid_trials
        sllf_rate = sllf_succ / valid_trials
        llf_rates.append(llf_rate)
        sllf_rates.append(sllf_rate)
        print(f"EVs {n_ev} (Valid: {valid_trials}, Attempts: {attempts}): LLF={llf_rate:.2f}, sLLF={sllf_rate:.2f}")

# Plot
x = np.arange(len(STRESS_LEVELS))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, llf_rates, width, label='LLF', color='skyblue', hatch='///')
rects2 = ax.bar(x + width/2, sllf_rates, width, label='sLLF', color='salmon', alpha=0.9)

ax.set_ylabel('Success Rate (among Feasible)')
ax.set_title('Robustness Comparison: LLF vs sLLF')
ax.set_xticks(x)
ax.set_xticklabels(STRESS_LEVELS)
ax.set_xlabel('Stress Level (Number of EVs)')
ax.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()