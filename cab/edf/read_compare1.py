import pandas as pd
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
TIME_STEP = 0.1              # 시뮬레이션 시간 단위 (0.1시간 = 6분)
EPSILON = 1e-6               # 부동소수점 오차 처리용 상수
TOTAL_STATION_POWER = 45   # 전체 충전소 전력 용량 (kW) - 혼잡 상황 유도를 위해 제한
MAX_EV_POWER = 5.8          # 개별 전기차 최대 충전 속도 (kW)
# F_PATH='/users/jaewoo/data/acn/acn_data_caltech_20190901_20191001.csv'
F_PATH='/users/jaewoo/data/acn/acn_data_1week.csv'



@dataclass
class EVRequest:
    ev_id: int
    arrival: float      # 도착 시간 (상대 시간, 시간 단위)
    required_energy: float # 필요 에너지 (kWh)
    deadline: float     # 마감 시간 (상대 시간, 시간 단위)
    remaining: float    # 남은 충전 필요량 (kWh)

    def __repr__(self):
        return f"EV{self.ev_id}(A={self.arrival:.2f}, E={self.required_energy:.2f}, D={self.deadline:.2f})"

# ---------------------------------------------------------
# 2. 데이터 로딩 및 전처리
# ---------------------------------------------------------
def load_ev_data(filename):
    try:
        df_raw = pd.read_csv(filename)
        
        # 시간 형식 변환
        df_raw['connectionTime'] = pd.to_datetime(df_raw['connectionTime'], utc=True)
        df_raw['disconnectTime'] = pd.to_datetime(df_raw['disconnectTime'], utc=True)
        
        # 시간순 정렬
        df_raw = df_raw.sort_values('connectionTime').reset_index(drop=True)
        
        # 시뮬레이션 시작 시간 (첫 도착 시간)
        start_time = df_raw['connectionTime'].min()
        
        # 절대 시간을 시뮬레이션용 상대 시간(0.0부터 시작하는 시간)으로 변환
        df_raw['arrival_h'] = (df_raw['connectionTime'] - start_time).dt.total_seconds() / 3600.0
        df_raw['deadline_h'] = (df_raw['disconnectTime'] - start_time).dt.total_seconds() / 3600.0
        
        # 유효성 검사 (도착 시간이 마감 시간보다 빠른 경우만 사용)
        df_clean = df_raw[df_raw['deadline_h'] > df_raw['arrival_h']].copy()
        
        # EVRequest 객체 리스트 생성
        requests = []
        for idx, row in df_clean.iterrows():
            requests.append(EVRequest(
                ev_id=idx,
                arrival=row['arrival_h'],
                required_energy=row['kWhDelivered'],
                deadline=row['deadline_h'],
                remaining=row['kWhDelivered']
            ))
            
        return requests
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

# ---------------------------------------------------------
# 3. 제안 알고리즘 (NEW_ALGO) 로직
# ---------------------------------------------------------
def calculate_new_algo_power(current_time, active_evs, grid_capacity, max_ev_power, time_step):
    count = len(active_evs)
    if count == 0: return []

    durations = []
    remains = []
    for ev in active_evs:
        # 남은 시간 계산 (최소 1 time step 보장)
        d = max(time_step, float(ev.deadline) - current_time)
        durations.append(d)
        remains.append(ev.remaining)

    # [Step 1] 필수 요구량(Mandatory) 계산
    allocation = [0.0] * count
    for i in range(count):
        future_time = durations[i] - time_step
        max_future_charge = max(0.0, future_time * max_ev_power)
        # 다음 step에 미루면 안 되는 최소 충전량
        mandatory_energy = max(0.0, remains[i] - max_future_charge)
        
        mandatory_power = mandatory_energy / time_step
        phys_limit_power = min(max_ev_power, remains[i] / time_step)
        allocation[i] = min(mandatory_power, phys_limit_power)

    # 용량 초과 시 비율대로 축소
    current_load = sum(allocation)
    remaining_grid = 0.0
    if current_load > grid_capacity + EPSILON:
        scale_factor = grid_capacity / current_load
        allocation = [a * scale_factor for a in allocation]
    else:
        remaining_grid = grid_capacity - current_load

    # [Step 2] 여유 전력 분배 (Minimax Stress)
    if remaining_grid > EPSILON:
        residual_caps = [max(0.0, min(max_ev_power, remains[i]/time_step) - allocation[i]) for i in range(count)]
        residual_needs = [max(0.0, remains[i] - allocation[i]*time_step) for i in range(count)]
        
        if sum(residual_needs) > EPSILON:
            # 이진 탐색으로 최적의 stress level 찾기
            low, high = -1000.0, 1000.0
            best_extra = [0.0] * count
            for _ in range(15):
                stress = (low + high) / 2.0
                proposal = []
                p_total = 0.0
                for i in range(count):
                    # 목표: 잔여량 - (stress * 남은시간)
                    target = residual_needs[i] - (stress * durations[i])
                    p = max(0.0, min(target/time_step, residual_caps[i]))
                    proposal.append(p)
                    p_total += p
                
                if p_total > remaining_grid:
                    low = stress
                else:
                    high = stress
                    best_extra = proposal
            
            for i in range(count):
                allocation[i] += best_extra[i]
                remaining_grid -= best_extra[i]

    # [Step 3] 남은 전력 최대한 채우기 (Greedy)
    if remaining_grid > EPSILON:
        # 긴급도(Laxity) 순으로 정렬하여 잔여 전력 할당
        indices = sorted(range(count), key=lambda i: remains[i]/durations[i] if durations[i]>0 else 999, reverse=True)
        for idx in indices:
            if remaining_grid <= EPSILON: break
            cap = min(max_ev_power, remains[idx]/time_step)
            room = cap - allocation[idx]
            if room > EPSILON:
                add = min(remaining_grid, room)
                allocation[idx] += add
                remaining_grid -= add
                
    return allocation

# ---------------------------------------------------------
# 4. 시뮬레이션 엔진
# ---------------------------------------------------------
def run_simulation(ev_set: List[EVRequest], algorithm: str) -> float:
    evs = copy.deepcopy(ev_set)
    total_evs_count = len(evs)
    
    active_evs = [] # 현재 충전 중인 EV
    waiting_evs = sorted(evs, key=lambda x: x.arrival) # 아직 도착 안 한 EV
    
    success_count = 0
    failure_count = 0
    current_time = 0.0
    
    # 모든 EV가 처리될 때까지 반복
    while success_count + failure_count < total_evs_count:
        # 1. 도착 처리
        while waiting_evs and waiting_evs[0].arrival <= current_time + EPSILON:
            active_evs.append(waiting_evs.pop(0))
            
        # 2. 실패(Deadline Miss) 처리
        for i in range(len(active_evs) - 1, -1, -1):
            ev = active_evs[i]
            if current_time > ev.deadline + EPSILON:
                failure_count += 1
                active_evs.pop(i)
        
        if not active_evs and not waiting_evs:
            break
            
        # 최적화: 할 일이 없으면 다음 도착 시간으로 점프
        if not active_evs and waiting_evs:
            current_time = waiting_evs[0].arrival
            continue

        # 3. 전력 할당 계산
        if algorithm == 'NEW_ALGO':
            powers = calculate_new_algo_power(current_time, active_evs, TOTAL_STATION_POWER, MAX_EV_POWER, TIME_STEP)
            # 안전장치: 전체 용량 초과 방지
            if sum(powers) > TOTAL_STATION_POWER + EPSILON:
                scale = TOTAL_STATION_POWER / sum(powers)
                powers = [p * scale for p in powers]
                
            # 실제 충전 수행
            for i, ev in enumerate(active_evs):
                charged = min(ev.remaining, powers[i] * TIME_STEP)
                ev.remaining -= charged
        else:
            # 기존 알고리즘 스케줄링 (정렬)
            if algorithm == 'EDF':
                active_evs.sort(key=lambda x: (x.deadline, x.ev_id))
            elif algorithm == 'LLF':
                # Least Laxity First
                active_evs.sort(key=lambda x: (x.deadline - current_time - (x.remaining/MAX_EV_POWER), x.ev_id))
            elif algorithm == 'FCFS':
                active_evs.sort(key=lambda x: (x.arrival, x.ev_id))
                
            used_power = 0.0
            for ev in active_evs:
                available = TOTAL_STATION_POWER - used_power
                if available <= EPSILON: break
                
                rate = min(MAX_EV_POWER, available)
                charged = min(ev.remaining, rate * TIME_STEP)
                
                ev.remaining -= charged
                used_power += rate

        # 4. 성공(충전 완료) 처리
        for i in range(len(active_evs) - 1, -1, -1):
            if active_evs[i].remaining <= EPSILON:
                success_count += 1
                active_evs.pop(i)
                
        current_time += TIME_STEP
        
        # 무한 루프 방지 (데이터 범위 초과 시)
        if current_time > 500.0:
             failure_count += len(active_evs)
             break
             
    return (success_count / total_evs_count) * 100.0

# ---------------------------------------------------------
# 5. 실행 및 시각화
# ---------------------------------------------------------
# 데이터 로드
filename = F_PATH
requests = load_ev_data(filename)
print(f"Loaded {len(requests)} requests from {filename}")

# 알고리즘별 시뮬레이션
algorithms = ['EDF', 'LLF', 'FCFS', 'NEW_ALGO']
results = {}

for algo in algorithms:
    score = run_simulation(requests, algo)
    results[algo] = score
    print(f"{algo}: {score:.2f}%")

# 그래프 그리기
plt.figure(figsize=(10, 6))
bars = plt.bar(results.keys(), results.values(), 
               color=['blue', 'red', 'green', 'purple'], alpha=0.7)

# 막대 위에 값 표시
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.0, 
             f"{yval:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Algorithm Success Rate Comparison (Real Data)', fontsize=14)
plt.ylabel('Success Rate (%)', fontsize=12)
plt.ylim(0, 110)
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.show()