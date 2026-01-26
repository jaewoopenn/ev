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
TOTAL_STATION_POWER = 35   # 전체 충전소 전력 용량 (kW) - 혼잡 상황 유도를 위해 제한
MAX_EV_POWER = 6.6          # 개별 전기차 최대 충전 속도 (kW)
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
    """
    Event-based Backward Analysis (Staircase Method)
    - Grid(배열)를 사용하지 않고, 데드라인 기반의 '구간(Segment)' 수식 계산으로 최적화함.
    - 시간 복잡도: O(T) -> O(N^2) (T가 클 때 압도적으로 유리, N은 보통 작음)
    - 공간 복잡도: O(T) -> O(N)
    - 결과: Grid 방식과 수학적으로 100% 동일한 Must Load 산출 (Exact Analytic Solution)
    """
    if not active_evs:
        return []

    # -----------------------------------------------------
    # 1. 구간(Segment) 생성 (Time Discretization by Events)
    # -----------------------------------------------------
    # 모든 EV의 데드라인을 수집하여 정렬 (중복 제거)
    # 현재 시간(current_time)은 시작점, 데드라인은 끝점들이 됨.
    # 예: Current=0, D=[10, 30] -> Segments: [0, 10], [10, 30]
    
    deadlines = sorted(list(set(ev.deadline for ev in active_evs)))
    
    # 현재 시간이 데드라인보다 뒤에 있는 경우 필터링 (이미 지난 데드라인 방어)
    deadlines = [d for d in deadlines if d > current_time]
    
    # Segment 정의: [start_time, end_time, remaining_capacity]
    segments = []
    start_t = current_time
    for d in deadlines:
        duration = d - start_t
        # 각 구간의 총 가용 에너지 = (시간 길이) * (시스템 용량)
        cap_energy = grid_capacity * duration
        segments.append({
            "start": start_t,
            "end": d,
            "capacity": cap_energy
        })
        start_t = d

    # -----------------------------------------------------
    # 2. 역방향 채우기 (Analytic Backward Filling)
    # -----------------------------------------------------
    # Must Load 저장소
    individual_must_loads = {ev.ev_id: 0.0 for ev in active_evs}
    
    # 데드라인이 늦은 차부터 처리 (충돌 최소화 & 정확한 Spillover 계산)
    # 정렬된 복사본 사용
    sorted_evs = sorted(active_evs, key=lambda x: x.deadline, reverse=True)
    
    for ev in sorted_evs:
        energy_needed = ev.remaining
        
        # 자신의 데드라인보다 앞선 구간들만 탐색 (역순 탐색: 늦은 구간 -> 빠른 구간)
        # 내 데드라인 안에 포함되는 구간들을 뒤에서부터 훑음
        for seg in reversed(segments):
            if energy_needed <= EPSILON:
                break
            
            # 이 구간이 내 데드라인 이후라면 패스 (넣을 수 없음)
            if seg["start"] >= ev.deadline:
                continue
            
            # 이 구간에 넣을 수 있는 나의 물리적 한계
            # 1. 구간의 남은 시스템 용량 (seg["capacity"])
            # 2. 나의 최대 충전 속도 한계 (duration * max_power)
            duration = seg["end"] - seg["start"]
            my_phys_limit = max_ev_power * duration
            
            # 실제로 채울 양
            fill = min(energy_needed, seg["capacity"], my_phys_limit)
            
            if fill > EPSILON:
                seg["capacity"] -= fill
                energy_needed -= fill
                
                # 핵심: 이 구간의 시작점이 '현재 시간(current_time)'이라면
                # 이 양은 '지금 당장' 처리해야 하는 Must Load에 기여함
                # (첫 번째 세그먼트가 항상 [current_time, d_min] 이므로)
                if abs(seg["start"] - current_time) < EPSILON:
                    # 주의: Must Load는 '일률(Power)'이 아니라 '에너지(Energy)' 관점에서
                    # 현재 타임스텝(time_step) 내에 처리해야 할 양임.
                    # 하지만 여기서는 'Must Rate'를 구해야 하므로, 
                    # 이 Spillover가 첫 번째 TimeStep(1칸) 안에 다 들어가야 하는지 확인 필요.
                    
                    # 더 정확한 로직: 
                    # Spillover된 에너지는 t=0 시점에 몰린 부하임.
                    # 이를 Power로 환산: Energy / time_step
                    # 단, 이 Energy는 첫 번째 구간(길이 duration) 전체에 걸친 것일 수 있음.
                    # 우리는 "지금 당장(time_step)" 필요한 Power를 원함.
                    
                    # 세그먼트 방식의 미세 조정:
                    # 첫 번째 세그먼트의 길이가 time_step보다 길 수 있음 (예: [0, 10]).
                    # 하지만 Backward Filling의 특성상, 가장 앞쪽(0)으로 밀려난 에너지는
                    # '가장 급한' 에너지임. 따라서 이를 time_step으로 나눠서 즉시 처리해야 함.
                    
                    # *수정*: 여기서 구해진 energy_needed 잔여분이 아님. 
                    # fill 자체가 t=0 근처 구간에 할당된 것임.
                    # 엄밀히 말해 't=0~step' 사이의 Must Load를 구하려면
                    # 첫 세그먼트를 [current, current+step]으로 쪼개는 게 가장 정확함.
                    pass 

    # -----------------------------------------------------
    # 2.1 (개선) 세그먼트 미세화 전략
    # -----------------------------------------------------
    # 위 로직에서 첫 세그먼트가 길면 Must Load가 희석될 수 있음.
    # 정확한 '이번 턴(time_step)'의 부하를 위해 첫 구간을 강제로 time_step 만큼 자릅니다.
    
    segments = []
    # 첫 구간: [current, current + time_step]
    segments.append({"start": current_time, "end": current_time + time_step, "capacity": grid_capacity * time_step})
    
    # 나머지 구간들: 데드라인 기반
    deadlines = sorted(list(set(ev.deadline for ev in active_evs)))
    deadlines = [d for d in deadlines if d > current_time + time_step] # 첫 스텝 이후만
    
    start_t = current_time + time_step
    for d in deadlines:
        duration = d - start_t
        if duration > EPSILON:
            segments.append({"start": start_t, "end": d, "capacity": grid_capacity * duration})
            start_t = d
            
    # 다시 채우기 (Logic 동일)
    individual_must_loads = {ev.ev_id: 0.0 for ev in active_evs}
    
    for ev in sorted_evs:
        energy_needed = ev.remaining
        for seg in reversed(segments):
            if energy_needed <= EPSILON: break
            if seg["start"] >= ev.deadline: continue
            
            duration = seg["end"] - seg["start"]
            my_phys_limit = max_ev_power * duration
            fill = min(energy_needed, seg["capacity"], my_phys_limit)
            
            if fill > EPSILON:
                seg["capacity"] -= fill
                energy_needed -= fill
                
                # 첫 번째 세그먼트(Current Step)에 할당된 양만 Must Load로 인정
                if abs(seg["start"] - current_time) < EPSILON:
                    individual_must_loads[ev.ev_id] += fill # Energy

    # -----------------------------------------------------
    # 3. 배분 (Allocation) - 2-Pass Work-Conserving
    # -----------------------------------------------------
    allocations = {}
    
    # Step 1: Must Load (Power로 변환)
    for ev in active_evs:
        must_energy = individual_must_loads.get(ev.ev_id, 0.0)
        allocations[ev.ev_id] = must_energy / time_step # Power = Energy / TimeStep

    # Step 2: Surplus 배분 (EDF)
    current_total_power = sum(allocations.values())
    surplus_power = max(0.0, grid_capacity - current_total_power)
    
    if surplus_power > EPSILON:
        # 마감 빠른 순 보너스
        for ev in sorted(active_evs, key=lambda x: x.deadline):
            if surplus_power <= EPSILON: break
            
            current_p = allocations[ev.ev_id]
            max_p = min(max_ev_power, ev.remaining / time_step)
            room = max(0.0, max_p - current_p)
            
            if room > EPSILON:
                bonus = min(surplus_power, room)
                allocations[ev.ev_id] += bonus
                surplus_power -= bonus

    return [allocations.get(ev.ev_id, 0.0) for ev in active_evs]

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