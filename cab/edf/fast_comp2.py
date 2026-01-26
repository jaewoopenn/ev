import random
import copy
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
from collections import deque
import time
from multiprocessing import Pool, cpu_count

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
TRIAL_NUM = 200
TIME_STEP = 1         
EPSILON = 1e-6          

TOTAL_STATION_POWER = 5
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

    time_points = set()
    total_required_energy = 0.0
    for ev in ev_requests:
        time_points.add(float(ev.arrival))
        time_points.add(float(ev.deadline))
        total_required_energy += ev.required_energy
    
    sorted_points = sorted(list(time_points))
    
    intervals = []
    for i in range(len(sorted_points) - 1):
        start = sorted_points[i]
        end = sorted_points[i+1]
        if end > start + EPSILON:
            intervals.append((start, end))

    num_evs = len(ev_requests)
    num_intervals = len(intervals)
    source = 0
    sink = 1
    ev_node_start = 2
    interval_node_start = 2 + num_evs
    total_nodes = 2 + num_evs + num_intervals
    
    solver = DinicMaxFlow(total_nodes)

    for i, ev in enumerate(ev_requests):
        u = source
        v = ev_node_start + i
        solver.add_edge(u, v, ev.required_energy)

    for j, (start, end) in enumerate(intervals):
        interval_idx = interval_node_start + j
        duration = end - start
        solver.add_edge(interval_idx, sink, TOTAL_STATION_POWER * duration)
        
        for i, ev in enumerate(ev_requests):
            if start >= ev.arrival - EPSILON and end <= ev.deadline + EPSILON:
                ev_idx = ev_node_start + i
                solver.add_edge(ev_idx, interval_idx, MAX_EV_POWER * duration)

    max_flow_val = solver.max_flow(source, sink)
    return abs(max_flow_val - total_required_energy) < 1e-4

# ---------------------------------------------------------
# 2. EV 요청 생성기
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
# 3. New Algorithm Logic
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
# 4. 시뮬레이션 엔진
# ---------------------------------------------------------
def run_simulation(ev_set: List[EVRequest], algorithm: str) -> bool:
    if algorithm == 'OFFLINE_OPT':
        return check_offline_optimal(ev_set)

    evs = copy.deepcopy(ev_set)
    max_deadline = max(e.deadline for e in evs) if evs else 0
    current_time = 0.0
    finished_cnt = 0
    total_evs = len(evs)

    while finished_cnt < total_evs:
        ready_queue = [
            e for e in evs 
            if e.arrival <= current_time + EPSILON and e.remaining > EPSILON
        ]
        
        for e in ready_queue:
            if current_time >= float(e.deadline) - EPSILON:
                return False 

        if algorithm == 'NEW_ALGO':
            allocated_powers = calculate_new_algo_power(
                current_time, ready_queue, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP
            )
            if sum(allocated_powers) > TOTAL_STATION_POWER + EPSILON:
                scale = TOTAL_STATION_POWER / sum(allocated_powers)
                allocated_powers = [p * scale for p in allocated_powers]
            
            for i, ev in enumerate(ready_queue):
                charged = min(ev.remaining, allocated_powers[i] * TIME_STEP)
                ev.remaining -= charged
                if ev.remaining <= EPSILON: ev.remaining = 0.0

        else:
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

        finished_cnt = sum(1 for e in evs if e.remaining <= EPSILON)
        current_time += TIME_STEP
        
        if current_time > max_deadline + 10.0:
            return False

    return True

# ---------------------------------------------------------
# 5. [수정됨] 병렬 처리를 위한 단일 작업 함수
# ---------------------------------------------------------
def run_batch_simulation(level):
    """
    하나의 congestion_level에 대해 단일 Trial을 수행하고
    각 알고리즘별 성공 여부를 반환하는 함수 (Worker Process에서 실행됨)
    """
    # 랜덤 시드 재설정 (프로세스 별로 다른 난수 생성을 위해 권장되나, Python MP에서는 보통 자동 처리됨)
    # random.seed() 
    
    ev_requests = generate_ev_set(level)
    results = {'EDF':0, 'LLF':0, 'FCFS':0, 'NEW_ALGO':0, 'OFFLINE_OPT':0}
    
    if not ev_requests:
        # 요청이 없는 경우 모두 성공으로 간주하거나, 스킵 (여기선 스킵 처리 위해 -1 등 사용 가능하나 편의상 1로 처리)
        return level, {k:1 for k in results}

    if run_simulation(ev_requests, 'EDF'): results['EDF'] = 1
    if run_simulation(ev_requests, 'LLF'): results['LLF'] = 1
    if run_simulation(ev_requests, 'FCFS'): results['FCFS'] = 1
    if run_simulation(ev_requests, 'NEW_ALGO'): results['NEW_ALGO'] = 1
    if run_simulation(ev_requests, 'OFFLINE_OPT'): results['OFFLINE_OPT'] = 1
    
    return level, results

# ---------------------------------------------------------
# 6. 메인 실행 블록 (Multiprocessing 진입점)
# ---------------------------------------------------------
if __name__ == '__main__':
    # 윈도우/Mac에서의 프로세스 생성 보호를 위해 필수
    
    start_time = time.time()
    
    congestion_levels = list(range(STRESS_START, STRESS_START+STRESS_NUM))
    
    # 작업 목록 생성: (Level 5 * 200번) + (Level 6 * 200번) ...
    tasks = []
    for level in congestion_levels:
        for _ in range(TRIAL_NUM):
            tasks.append(level)
    
    # CPU 코어 수 확인
    num_cores = cpu_count()
    print(f"Starting Parallel Simulation on {num_cores} cores.")
    print(f"Total Tasks: {len(tasks)} (Levels: {STRESS_NUM}, Trials/Level: {TRIAL_NUM})")

    # 병렬 처리 실행
    # tasks 리스트의 각 항목을 run_batch_simulation 함수에 전달하여 병렬 실행
    with Pool(processes=num_cores) as pool:
        # map은 순서를 보장하여 결과를 반환함
        results = pool.map(run_batch_simulation, tasks)

    # 결과 집계
    # 구조: aggregated_results[level] = {'EDF': sum_wins, 'LLF': sum_wins, ...}
    aggregated_results = {lvl: {'EDF':0, 'LLF':0, 'FCFS':0, 'NEW_ALGO':0, 'OFFLINE_OPT':0} for lvl in congestion_levels}
    
    for level, res in results:
        for algo in res:
            aggregated_results[level][algo] += res[algo]

    # 비율 계산 및 리스트 변환 (그래프용)
    edf_ratios, llf_ratios, fcfs_ratios, new_ratios, opt_ratios = [], [], [], [], []

    print("-" * 60)
    print(f"{'Level':<6} | {'EDF':<6}    {'FCFS':<6}    {'LLF':<6}    {'NEW':<6}    {'OPT':<6}")
    print("-" * 60)

    for level in congestion_levels:
        wins = aggregated_results[level]
        
        edf_r = wins['EDF'] / TRIAL_NUM
        llf_r = wins['LLF'] / TRIAL_NUM
        fcfs_r = wins['FCFS'] / TRIAL_NUM
        new_r = wins['NEW_ALGO'] / TRIAL_NUM
        opt_r = wins['OFFLINE_OPT'] / TRIAL_NUM
        
        edf_ratios.append(edf_r)
        llf_ratios.append(llf_r)
        fcfs_ratios.append(fcfs_r)
        new_ratios.append(new_r)
        opt_ratios.append(opt_r)
        
        print(f"{level:<6} | {edf_r:.2f}    {fcfs_r:.2f}    {llf_r:.2f}    {new_r:.2f}    {opt_r:.2f}")

    print("-" * 60)
    print(f"Simulation Finished in {time.time() - start_time:.2f} seconds")

    # ---------------------------------------------------------
    # 7. 결과 그래프
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))

    plt.plot(congestion_levels, edf_ratios, marker='o', label='EDF', linestyle='-', color='blue', alpha=0.3)
    plt.plot(congestion_levels, llf_ratios, marker='s', label='LLF', linestyle='--', color='red', alpha=0.3)
    plt.plot(congestion_levels, fcfs_ratios, marker='^', label='FCFS', linestyle=':', color='green', alpha=0.3)
    plt.plot(congestion_levels, new_ratios, marker='*', label='NEW_ALGO', linestyle='-', linewidth=3, color='purple')
    plt.plot(congestion_levels, opt_ratios, marker='D', label='Offline Optimal', linestyle='--', linewidth=2, color='black')

    plt.title(f'Performance Comparison (Parallel Execution)', fontsize=14)
    plt.xlabel('Stress Level', fontsize=12)
    plt.ylabel('Success Ratio', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(congestion_levels)
    plt.legend(fontsize=12)
    plt.show()