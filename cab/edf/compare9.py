import random
import copy
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
from collections import deque

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
TRIAL_NUM = 200
TIME_STEP = 0.1         
EPSILON = 1e-6          

TOTAL_STATION_POWER = 6
MAX_EV_POWER = 1.0         

STRESS_START = 5
STRESS_NUM = 10

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
# [추가] Max Flow Solver (Dinic's Algorithm)
# ---------------------------------------------------------
class DinicMaxFlow:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.level = []

    def add_edge(self, u, v, capacity):
        # Forward edge with capacity
        self.graph[u].append([v, capacity, len(self.graph[v])])
        # Backward edge with 0 capacity
        self.graph[v].append([u, 0.0, len(self.graph[u]) - 1])

    def bfs(self, s, t):
        self.level = [-1] * self.n
        self.level[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v, cap, rev_idx in self.graph[u]:
                if cap > EPSILON and self.level[v] < 0:
                    self.level[v] = self.level[u] + 1
                    queue.append(v)
        return self.level[t] >= 0

    def dfs(self, u, t, flow, ptr):
        if u == t or flow < EPSILON:
            return flow
        for i in range(ptr[u], len(self.graph[u])):
            ptr[u] = i
            v, cap, rev_idx = self.graph[u][i]
            if self.level[v] == self.level[u] + 1 and cap > EPSILON:
                pushed = self.dfs(v, t, min(flow, cap), ptr)
                if pushed > EPSILON:
                    self.graph[u][i][1] -= pushed
                    self.graph[v][rev_idx][1] += pushed
                    return pushed
        return 0

    def max_flow(self, s, t):
        flow = 0.0
        while self.bfs(s, t):
            ptr = [0] * self.n
            while True:
                pushed = self.dfs(s, t, float('inf'), ptr)
                if pushed < EPSILON:
                    break
                flow += pushed
        return flow

# ---------------------------------------------------------
# [추가] Offline Optimal 판별 함수 (Canonical Interval Method)
# ---------------------------------------------------------
def check_offline_optimal(ev_requests: List[EVRequest]) -> bool:
    if not ev_requests:
        return True

    # 1. 모든 도착 및 마감 시간을 수집하여 정렬 (Time Points)
    time_points = set()
    total_required_energy = 0.0
    for ev in ev_requests:
        time_points.add(float(ev.arrival))
        time_points.add(float(ev.deadline))
        total_required_energy += ev.required_energy
    
    sorted_points = sorted(list(time_points))
    
    # 2. Canonical Intervals 생성
    intervals = []
    for i in range(len(sorted_points) - 1):
        start = sorted_points[i]
        end = sorted_points[i+1]
        if end > start + EPSILON:
            intervals.append((start, end))

    # 3. 그래프 노드 설정
    # Node 0: Source (S)
    # Node 1: Sink (T)
    # Node 2 ~ 2+num_evs-1: EV Nodes
    # Node 2+num_evs ~ ... : Interval Nodes
    
    num_evs = len(ev_requests)
    num_intervals = len(intervals)
    source = 0
    sink = 1
    ev_node_start = 2
    interval_node_start = 2 + num_evs
    total_nodes = 2 + num_evs + num_intervals
    
    solver = DinicMaxFlow(total_nodes)

    # 4. 에지 연결
    
    # (1) Source -> EV Nodes
    for i, ev in enumerate(ev_requests):
        u = source
        v = ev_node_start + i
        # EV가 필요한 총 에너지
        solver.add_edge(u, v, ev.required_energy)

    # (2) EV Nodes -> Interval Nodes -> Sink
    for j, (start, end) in enumerate(intervals):
        interval_idx = interval_node_start + j
        duration = end - start
        
        # Interval -> Sink (스테이션 전체 용량 제한)
        # 이 구간 동안 스테이션이 공급할 수 있는 총량 = Power * duration
        solver.add_edge(interval_idx, sink, TOTAL_STATION_POWER * duration)
        
        # EV -> Interval (개별 EV 충전 가능 여부)
        for i, ev in enumerate(ev_requests):
            # EV가 이 구간(start~end)에 존재하면 연결
            if start >= ev.arrival - EPSILON and end <= ev.deadline + EPSILON:
                ev_idx = ev_node_start + i
                # 이 구간 동안 EV 하나가 최대로 받을 수 있는 양 = MaxPower * duration
                solver.add_edge(ev_idx, interval_idx, MAX_EV_POWER * duration)

    # 5. Max Flow 계산
    max_flow_val = solver.max_flow(source, sink)
    
    # Max Flow가 총 요구량과 같으면 모든 EV 충전 가능
    return abs(max_flow_val - total_required_energy) < 1e-4

# ---------------------------------------------------------
# 2. EV 요청 생성기 (Super Fine-grained 12 Levels)
# ---------------------------------------------------------
def generate_ev_set(congestion_level):
    max_time = 60
    
    arrival_rate = 0.55 + (congestion_level * 0.075)
    base_avg_energy = 2.5
    avg_energy = base_avg_energy + (congestion_level * 0.09)
    std_energy = 1.3 
    slack_mean_factor = max(0.4, 1.9 - (congestion_level * 0.098))

    ev_requests = []
    current_time = 0.0
    ev_id = 0
    
    while True:
        inter_arrival = random.expovariate(arrival_rate)
        current_time += inter_arrival
        if current_time > max_time:
            break
        arrival_int = int(current_time)
        req_energy = random.gauss(avg_energy, std_energy)
        req_energy = max(0.5, min(8.0, req_energy))
        req_energy = round(req_energy, 2)
        min_required_time = math.ceil(req_energy / MAX_EV_POWER)
        raw_slack = random.expovariate(1.0) * (min_required_time * slack_mean_factor)
        slack = int(raw_slack)
        deadline = arrival_int + min_required_time + slack
        ev_requests.append(EVRequest(ev_id, arrival_int, req_energy, deadline, req_energy))
        ev_id += 1
        
    return ev_requests

# ---------------------------------------------------------
# 3. New Algorithm Logic (유지)
# ---------------------------------------------------------
def calculate_new_algo_power(current_time, active_evs, grid_capacity, max_ev_power, time_step):
    count = len(active_evs)
    if count == 0: return []
    durations = []
    remains = []
    for ev in active_evs:
        d = max(time_step, float(ev.deadline) - current_time)
        durations.append(d)
        remains.append(ev.remaining)
    allocation = [0.0] * count
    for i in range(count):
        future_time = durations[i] - time_step
        max_future_charge = max(0.0, future_time * max_ev_power)
        mandatory_energy = max(0.0, remains[i] - max_future_charge)
        mandatory_power = mandatory_energy / time_step
        phys_limit_power = min(max_ev_power, remains[i] / time_step)
        allocation[i] = min(mandatory_power, phys_limit_power)
    current_load = sum(allocation)
    if current_load > grid_capacity + EPSILON:
        scale_factor = grid_capacity / current_load
        for i in range(count): allocation[i] *= scale_factor
        remaining_grid = 0.0
    else: remaining_grid = grid_capacity - current_load
    if remaining_grid > EPSILON:
        residual_caps, residual_needs = [], []
        for i in range(count):
            phys_limit_power = min(max_ev_power, remains[i] / time_step)
            res_cap = max(0.0, phys_limit_power - allocation[i])
            residual_caps.append(res_cap)
            allocated_energy = allocation[i] * time_step
            res_need = remains[i] - allocated_energy
            residual_needs.append(res_need)
        if sum(residual_needs) > EPSILON:
            low, high = -1000.0, 1000.0
            best_extra = [0.0] * count
            for _ in range(15):
                stress_level = (low + high) / 2.0
                proposed_total = 0.0
                current_proposal = []
                for i in range(count):
                    target_energy = residual_needs[i] - (stress_level * durations[i])
                    target_power = target_energy / time_step
                    p_power = max(0.0, min(target_power, residual_caps[i]))
                    current_proposal.append(p_power)
                    proposed_total += p_power
                if proposed_total > remaining_grid: low = stress_level
                else: high = stress_level; best_extra = current_proposal
            for i in range(count):
                allocation[i] += best_extra[i]
                remaining_grid -= best_extra[i]
    if remaining_grid > EPSILON:
        urgency_indices = []
        for i in range(count):
            phys_limit_power = min(max_ev_power, remains[i] / time_step)
            if phys_limit_power - allocation[i] > EPSILON:
                score = remains[i] / durations[i] if durations[i] > 0 else 9999
                urgency_indices.append((score, i))
        urgency_indices.sort(key=lambda x: x[0], reverse=True)
        for score, idx in urgency_indices:
            if remaining_grid <= EPSILON: break
            phys_limit_power = min(max_ev_power, remains[idx] / time_step)
            room_to_charge = phys_limit_power - allocation[idx]
            if room_to_charge > EPSILON:
                top_up = min(remaining_grid, room_to_charge)
                allocation[idx] += top_up
                remaining_grid -= top_up
    return allocation

# ---------------------------------------------------------
# 4. 시뮬레이션 엔진 (유지)
# ---------------------------------------------------------

def run_simulation(ev_set: List[EVRequest], algorithm: str) -> bool:
    # [유지] Offline Optimal 체크
    if algorithm == 'OFFLINE_OPT':
        return check_offline_optimal(ev_set)

    evs = copy.deepcopy(ev_set)
    max_deadline = max(e.deadline for e in evs) if evs else 0
    current_time = 0.0
    finished_cnt = 0
    total_evs = len(evs)

    while finished_cnt < total_evs:
        # 1. Ready Queue 구성
        ready_queue = [
            e for e in evs 
            if e.arrival <= current_time + EPSILON and e.remaining > EPSILON
        ]
        
        # 2. [수정됨] 엄격한 데드라인 체크
        # 현재 시간이 데드라인과 같거나 지났다면, 더 이상 충전할 시간이 없는 것입니다.
        # (남은 에너지가 있는데 데드라인에 도달했다면 실패)
        for e in ready_queue:
            # 수정 전: if current_time > float(e.deadline) + EPSILON:
            # 수정 후: current_time이 deadline에 도달했는지 체크 (EPSILON 여유 고려)
            if current_time >= float(e.deadline) - EPSILON:
                return False 

        # 3. 알고리즘 수행 (이하 동일)
        if algorithm == 'NEW_ALGO':
            allocated_powers = calculate_new_algo_power(
                current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP
            )
            # Power Cap 보정
            if sum(allocated_powers) > TOTAL_STATION_POWER + EPSILON:
                scale = TOTAL_STATION_POWER / sum(allocated_powers)
                allocated_powers = [p * scale for p in allocated_powers]
            
            for i, ev in enumerate(ready_queue):
                charged = min(ev.remaining, allocated_powers[i] * TIME_STEP)
                ev.remaining -= charged
                if ev.remaining <= EPSILON: ev.remaining = 0.0

        else: # EDF, LLF, FCFS 등 기존 로직
            if algorithm == 'EDF':
                ready_queue.sort(key=lambda x: (x.deadline, x.ev_id))
            elif algorithm == 'LLF':
                ready_queue.sort(key=lambda x: (x.deadline - current_time - (x.remaining/MAX_EV_POWER), x.ev_id))
            elif algorithm == 'FCFS':
                ready_queue.sort(key=lambda x: (x.arrival, x.ev_id))

            current_used = 0.0
            for ev in ready_queue:
                available = TOTAL_STATION_POWER - current_used
                if available <= EPSILON: break
                rate = min(MAX_EV_POWER, available)
                charged = min(ev.remaining, rate * TIME_STEP)
                ev.remaining -= charged
                current_used += rate
                if ev.remaining <= EPSILON: ev.remaining = 0.0

        # 4. 종료 조건 및 시간 업데이트
        finished_cnt = sum(1 for e in evs if e.remaining <= EPSILON)
        current_time += TIME_STEP
        
        # 안전장치: 너무 오래 돌면 강제 종료
        if current_time > max_deadline + 10.0:
            return False

    return True
# ---------------------------------------------------------
# 5. 메인 실험 (12단계)
# ---------------------------------------------------------
congestion_levels = list(range(STRESS_START, STRESS_START+STRESS_NUM))
edf_ratios, llf_ratios, fcfs_ratios, new_ratios, opt_ratios = [], [], [], [], [] # opt_ratios 추가
trials_per_level = TRIAL_NUM

print(f"EV Simulation (Includes Offline Optimal with Max Flow)")

for level in congestion_levels:
    wins = {'EDF':0, 'LLF':0, 'FCFS':0, 'NEW_ALGO':0, 'OFFLINE_OPT':0}
    for _ in range(trials_per_level):
        ev_requests = generate_ev_set(level)
        if not ev_requests: continue

        if run_simulation(ev_requests, 'EDF'): wins['EDF'] += 1
        if run_simulation(ev_requests, 'LLF'): wins['LLF'] += 1
        if run_simulation(ev_requests, 'FCFS'): wins['FCFS'] += 1
        if run_simulation(ev_requests, 'NEW_ALGO'): wins['NEW_ALGO'] += 1
        
        # [추가] Offline Optimal 실행
        if run_simulation(ev_requests, 'OFFLINE_OPT'): wins['OFFLINE_OPT'] += 1
            
    edf_ratios.append(wins['EDF'] / trials_per_level)
    llf_ratios.append(wins['LLF'] / trials_per_level)
    fcfs_ratios.append(wins['FCFS'] / trials_per_level)
    new_ratios.append(wins['NEW_ALGO'] / trials_per_level)
    opt_ratios.append(wins['OFFLINE_OPT'] / trials_per_level) # 결과 저장
    
    print(f"Level {level:2d}: EDF={edf_ratios[-1]:.2f}, NEW={new_ratios[-1]:.2f} | OPT={opt_ratios[-1]:.2f}")

# ---------------------------------------------------------
# 6. 결과 그래프
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

# 기존 알고리즘
plt.plot(congestion_levels, edf_ratios, marker='o', label='EDF', linestyle='-', color='blue', alpha=0.3)
plt.plot(congestion_levels, llf_ratios, marker='s', label='LLF', linestyle='--', color='red', alpha=0.3)
plt.plot(congestion_levels, fcfs_ratios, marker='^', label='FCFS', linestyle=':', color='green', alpha=0.3)
plt.plot(congestion_levels, new_ratios, marker='*', label='NEW_ALGO', linestyle='-', linewidth=3, color='purple')

# [추가] Offline Optimal 그래프 (검정색 점선)
plt.plot(congestion_levels, opt_ratios, marker='D', label='Offline Optimal', linestyle='--', linewidth=2, color='black')

plt.title(f'Performance Comparison (Included Offline Optimal)', fontsize=14)
plt.xlabel('Stress Level', fontsize=12)
plt.ylabel('Success Ratio', fontsize=12)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle=':', alpha=0.7)
plt.xticks(congestion_levels)
plt.legend(fontsize=12)
plt.show()