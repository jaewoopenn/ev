import random
import copy
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
TIME_STEP = 0.1       # 시간 단위 (작을수록 정밀)
EPSILON = 1e-6        # 부동소수점 오차 허용값
FIXED_PROCESSORS = 4  # 프로세서 개수 고정

@dataclass
class Job:
    job_id: int
    arrival: int      # [요청사항] 정수
    cost: float       # [요청사항] 실수
    deadline: int     # [요청사항] 정수
    remaining: float

    def __repr__(self):
        return f"J{self.job_id}(A={self.arrival}, C={self.cost:.2f}, D={self.deadline})"

# ---------------------------------------------------------
# 2. Job Set 생성기 (정수 Arrival/Deadline + 실수 Cost)
# ---------------------------------------------------------
def generate_mixed_job_set(congestion_level, m_processors):
    max_time = 60
    
    # 혼잡도 설정 (1~10단계)
    base_jobs = m_processors * 2
    step = m_processors * 1.5
    num_jobs = int(base_jobs + (congestion_level * step))
    
    jobs = []
    for i in range(num_jobs):
        # 1. 도착 시간: 정수 (Integer)
        arrival = random.randint(0, int(max_time * 0.6))
        
        # 2. 실행 시간: 실수 (Float)
        # 0.5초 ~ 8.0초 사이의 랜덤 실수
        cost = round(random.uniform(0.5, 8.0), 2)
        
        # 3. 마감 기한: 정수 (Integer)
        # 실행 시간이 4.2초라면, 최소 5초 이상의 여유가 있어야 함.
        # math.ceil(cost)를 통해 올림 처리 후 정수 Slack 추가
        min_required_time = math.ceil(cost)
        slack = random.randint(0, min_required_time * 2) 
        
        deadline = arrival + min_required_time + slack
        
        jobs.append(Job(i, arrival, cost, deadline, cost))
    return jobs

# ---------------------------------------------------------
# 3. 시뮬레이션 엔진 (Time Slicing)
# ---------------------------------------------------------
def run_simulation(job_set: List[Job], m_processors: int, algorithm: str) -> bool:
    jobs = copy.deepcopy(job_set)
    max_deadline = max(j.deadline for j in jobs) if jobs else 0
    
    current_time = 0.0 # 시뮬레이션 시간은 실수로 흐름 (TIME_STEP 단위)
    finished_cnt = 0
    total_jobs = len(jobs)

    while finished_cnt < total_jobs:
        # 1. Ready Queue Filter
        # arrival은 정수지만, current_time은 실수이므로 비교 가능
        ready_queue = [
            j for j in jobs 
            if j.arrival <= current_time + EPSILON and j.remaining > EPSILON
        ]
        
        # 2. Deadline Miss Check
        for j in ready_queue:
            # deadline은 정수. 현재 시간이 이를 초과하면 실패.
            if current_time > float(j.deadline) + EPSILON:
                return False

        # 3. Sort (Algorithm Logic)
        if algorithm == 'EDF':
            # Deadline(정수) 기준 오름차순
            ready_queue.sort(key=lambda x: (x.deadline, x.job_id))
        elif algorithm == 'LLF':
            # Laxity(실수) = Deadline(int) - Current(float) - Remaining(float)
            ready_queue.sort(key=lambda x: (x.deadline - current_time - x.remaining, x.job_id))

        # 4. Processor Allocation
        active_jobs = ready_queue[:m_processors]
        
        # 5. Execute
        if active_jobs:
            for job in active_jobs:
                decrement = min(job.remaining, TIME_STEP)
                job.remaining -= decrement
                
                # 0 미만으로 내려가는 것 방지 및 완료 처리
                if job.remaining <= EPSILON:
                    job.remaining = 0.0
        
        # 완료된 작업 수 갱신
        finished_cnt = sum(1 for j in jobs if j.remaining <= EPSILON)

        # 6. Time Advance
        current_time += TIME_STEP
        
        # Safety Break
        if current_time > max_deadline + 10.0:
            return False

    return True

# ---------------------------------------------------------
# 4. 메인 실험 루프
# ---------------------------------------------------------
congestion_levels = list(range(1, 11))
edf_ratios = []
llf_ratios = []
trials_per_level = 100

print(f"Simulation Start (m={FIXED_PROCESSORS}, Arrival/Deadline=Int, Cost=Float)...")

for level in congestion_levels:
    edf_wins = 0
    llf_wins = 0
    
    for _ in range(trials_per_level):
        jobs = generate_mixed_job_set(level, FIXED_PROCESSORS)
        
        if run_simulation(jobs, FIXED_PROCESSORS, 'EDF'):
            edf_wins += 1
        if run_simulation(jobs, FIXED_PROCESSORS, 'LLF'):
            llf_wins += 1
            
    edf_ratios.append(edf_wins / trials_per_level)
    llf_ratios.append(llf_wins / trials_per_level)
    
    print(f"Level {level}: EDF={edf_ratios[-1]:.2f}, LLF={llf_ratios[-1]:.2f}")

# ---------------------------------------------------------
# 5. 결과 그래프
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(congestion_levels, edf_ratios, marker='o', label='Global EDF', linestyle='-', color='blue')
plt.plot(congestion_levels, llf_ratios, marker='s', label='Global LLF', linestyle='--', color='red')

plt.title(f'Acceptance Ratio: Int Arrival/Deadline vs Float Cost\n(m={FIXED_PROCESSORS}, TimeStep={TIME_STEP})', fontsize=14)
plt.xlabel('Congestion Level (Job Density)', fontsize=12)
plt.ylabel('Acceptance Ratio', fontsize=12)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle=':', alpha=0.7)
plt.xticks(congestion_levels)
plt.legend(fontsize=12)

plt.show()