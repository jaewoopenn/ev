import random
import copy
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
TIME_STEP = 0.1         
EPSILON = 1e-6          

TOTAL_STATION_POWER = 4.2  # 충전소 전체 가용 전력
MAX_EV_POWER = 1.0         # 개별 EV의 최대 충전 속도

@dataclass
class EVRequest:
    ev_id: int
    arrival: int        
    required_energy: float 
    deadline: int       
    remaining: float    

    def __repr__(self):
        return f"EV{self.ev_id}(A={self.arrival}, E={self.required_energy:.2f}, D={self.deadline})"

# ---------------------------------------------------------
# 2. EV 요청 생성기
# ---------------------------------------------------------
def generate_ev_set(congestion_level):
    max_time = 60
    base_jobs = int(TOTAL_STATION_POWER * 2)
    step = int(TOTAL_STATION_POWER * 1.5)
    num_jobs = int(base_jobs + (congestion_level * step))
    
    ev_requests = []
    for i in range(num_jobs):
        arrival = random.randint(0, int(max_time * 0.6))
        required_energy = round(random.uniform(0.5, 8.0), 2)
        min_required_time = math.ceil(required_energy / MAX_EV_POWER)
        slack = random.randint(0, min_required_time * 2) 
        deadline = arrival + min_required_time + slack
        
        ev_requests.append(EVRequest(i, arrival, required_energy, deadline, required_energy))
    return ev_requests

# ---------------------------------------------------------
# 3. 시뮬레이션 엔진
# ---------------------------------------------------------
def run_simulation(ev_set: List[EVRequest], algorithm: str) -> bool:
    evs = copy.deepcopy(ev_set)
    max_deadline = max(e.deadline for e in evs) if evs else 0
    
    current_time = 0.0
    finished_cnt = 0
    total_evs = len(evs)

    while finished_cnt < total_evs:
        # 1. Ready Queue Filter
        ready_queue = [
            e for e in evs 
            if e.arrival <= current_time + EPSILON and e.remaining > EPSILON
        ]
        
        # 2. Deadline Miss Check
        for e in ready_queue:
            if current_time > float(e.deadline) + EPSILON:
                return False 

        # 3. Sort (Algorithm Logic)
        if algorithm == 'EDF':
            # 마감 시간(Deadline) 빠른 순
            ready_queue.sort(key=lambda x: (x.deadline, x.ev_id))
        elif algorithm == 'LLF':
            # 여유 시간(Laxity) 적은 순
            ready_queue.sort(key=lambda x: (x.deadline - current_time - (x.remaining / MAX_EV_POWER), x.ev_id))
        elif algorithm == 'FCFS':
            # 도착 시간(Arrival) 빠른 순 (먼저 온 차가 장땡)
            ready_queue.sort(key=lambda x: (x.arrival, x.ev_id))

        # 4. Power Allocation (Water-filling)
        current_used_power = 0.0
        
        for ev in ready_queue:
            available_power = TOTAL_STATION_POWER - current_used_power
            
            if available_power <= EPSILON:
                break 
            
            charging_rate = min(MAX_EV_POWER, available_power)
            
            charged_amount = charging_rate * TIME_STEP
            actual_charged = min(ev.remaining, charged_amount)
            
            ev.remaining -= actual_charged
            
            current_used_power += charging_rate
            
            if ev.remaining <= EPSILON:
                ev.remaining = 0.0

        finished_cnt = sum(1 for e in evs if e.remaining <= EPSILON)
        current_time += TIME_STEP
        
        if current_time > max_deadline + 10.0:
            return False

    return True

# ---------------------------------------------------------
# 4. 메인 실험 루프
# ---------------------------------------------------------
congestion_levels = list(range(1, 11))
edf_ratios = []
llf_ratios = []
fcfs_ratios = []  # FCFS 결과 저장용 리스트 추가
trials_per_level = 100

print(f"EV Charging Simulation Start (EDF vs LLF vs FCFS)...")
print(f"Total Power: {TOTAL_STATION_POWER}, Max EV Rate: {MAX_EV_POWER}")

for level in congestion_levels:
    edf_wins = 0
    llf_wins = 0
    fcfs_wins = 0
    
    for _ in range(trials_per_level):
        ev_requests = generate_ev_set(level)
        
        if run_simulation(ev_requests, 'EDF'):
            edf_wins += 1
        if run_simulation(ev_requests, 'LLF'):
            llf_wins += 1
        if run_simulation(ev_requests, 'FCFS'):
            fcfs_wins += 1
            
    edf_ratios.append(edf_wins / trials_per_level)
    llf_ratios.append(llf_wins / trials_per_level)
    fcfs_ratios.append(fcfs_wins / trials_per_level)
    
    print(f"Level {level}: EDF={edf_ratios[-1]:.2f}, LLF={llf_ratios[-1]:.2f}, FCFS={fcfs_ratios[-1]:.2f}")

# ---------------------------------------------------------
# 5. 결과 그래프
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(congestion_levels, edf_ratios, marker='o', label='EDF', linestyle='-', color='blue')
plt.plot(congestion_levels, llf_ratios, marker='s', label='LLF', linestyle='--', color='red')
plt.plot(congestion_levels, fcfs_ratios, marker='^', label='FCFS', linestyle='-.', color='green')

plt.title(f'EV Charging Success Ratio Comparison\n(Power={TOTAL_STATION_POWER}, EV Max={MAX_EV_POWER})', fontsize=14)
plt.xlabel('Congestion Level', fontsize=12)
plt.ylabel('Success Ratio', fontsize=12)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle=':', alpha=0.7)
plt.xticks(congestion_levels)
plt.legend(fontsize=12)

plt.show()