import random
import copy
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. Job 데이터 구조
# ---------------------------------------------------------
@dataclass
class Job:
    job_id: int
    arrival: int
    cost: int
    deadline: int
    remaining: int

# ---------------------------------------------------------
# 2. Job Set 생성기 (혼잡도 기반)
# ---------------------------------------------------------
def generate_job_set_by_congestion(congestion_level, m_processors):
    """
    congestion_level: 1 (여유) ~ 10 (과부하)
    혼잡도가 높을수록 동일 시간 내에 더 많은 작업이 생성됩니다.
    """
    max_time = 60  # 시뮬레이션 시간 범위 고정
    
    # 혼잡도 1단계: 프로세서당 약 2개 작업 (여유)
    # 혼잡도 10단계: 프로세서당 약 12개 작업 (과부하 예상)
    base_jobs = m_processors * 2
    step = m_processors * 1.5  # 단계별 증가량
    
    # int()로 정수 변환
    num_jobs = int(base_jobs + (congestion_level * step))
    
    jobs = []
    for i in range(num_jobs):
        # 작업들이 초중반에 몰리도록 설정 (병목 유도)
        arrival = random.randint(0, int(max_time * 0.6))
        
        # 실행 시간 랜덤 (1~8)
        cost = random.randint(1, 8)
        
        # 마감기한 설정
        # 혼잡도가 높을수록 Slack(여유)을 줄이는 옵션을 줄 수도 있지만,
        # 여기서는 순수하게 '작업량'만 늘리기 위해 Slack은 랜덤 유지
        slack = random.randint(0, cost * 3)
        deadline = arrival + cost + slack
        
        jobs.append(Job(i, arrival, cost, deadline, cost))
    return jobs

# ---------------------------------------------------------
# 3. 스케줄링 시뮬레이션 (이전과 동일 로직)
# ---------------------------------------------------------
def run_simulation(job_set: List[Job], m_processors: int, algorithm: str) -> bool:
    jobs = copy.deepcopy(job_set)
    max_deadline = max(j.deadline for j in jobs) if jobs else 0
    current_time = 0
    finished_jobs = 0
    total_jobs = len(jobs)

    while finished_jobs < total_jobs:
        # Ready Queue: 도착했으나 완료되지 않은 작업
        ready_queue = [j for j in jobs if j.arrival <= current_time and j.remaining > 0]
        
        # Deadline Miss Check
        for j in ready_queue:
            if current_time >= j.deadline:
                return False

        # Priority Sort
        if algorithm == 'EDF':
            # Deadline 오름차순
            ready_queue.sort(key=lambda x: (x.deadline, x.job_id))
        elif algorithm == 'LLF':
            # Laxity 오름차순 (L = D - t - C_remain)
            ready_queue.sort(key=lambda x: (x.deadline - current_time - x.remaining, x.job_id))

        # Select Top m Jobs
        active_jobs = ready_queue[:m_processors]
        
        # Execute (1 Tick)
        if active_jobs:
            for job in active_jobs:
                job.remaining -= 1
                if job.remaining == 0:
                    finished_jobs += 1
        
        current_time += 1
        
        if current_time > max_deadline + 20: 
            return False

    return True

# ---------------------------------------------------------
# 4. 메인 실험 루프 (x축: 혼잡도 1~10)
# ---------------------------------------------------------
fixed_processors = 4  # 프로세서 개수 고정
congestion_levels = list(range(1, 11)) # 1단계 ~ 10단계

edf_ratios = []
llf_ratios = []
trials_per_level = 200 # 각 단계별 시뮬레이션 횟수

print(f"Starting Simulation with m={fixed_processors}...")

for level in congestion_levels:
    edf_wins = 0
    llf_wins = 0
    
    for _ in range(trials_per_level):
        jobs = generate_job_set_by_congestion(level, fixed_processors)
        
        if run_simulation(jobs, fixed_processors, 'EDF'):
            edf_wins += 1
        if run_simulation(jobs, fixed_processors, 'LLF'):
            llf_wins += 1
            
    edf_ratios.append(edf_wins / trials_per_level)
    llf_ratios.append(llf_wins / trials_per_level)
    
    print(f"Level {level}: EDF={edf_wins/trials_per_level:.2f}, LLF={llf_wins/trials_per_level:.2f}")

# ---------------------------------------------------------
# 5. 그래프 시각화
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# 라인 그래프
plt.plot(congestion_levels, edf_ratios, marker='o', label='Global EDF', linestyle='-', linewidth=2, color='blue')
plt.plot(congestion_levels, llf_ratios, marker='s', label='Global LLF', linestyle='--', linewidth=2, color='red')

# 그래프 꾸미기
plt.title(f'Acceptance Ratio by Congestion Level\n(Processors m={fixed_processors})', fontsize=14)
plt.xlabel('Congestion Level (Job Density 1 -> 10)', fontsize=12)
plt.ylabel('Acceptance Ratio', fontsize=12)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle=':', alpha=0.7)
plt.xticks(congestion_levels)
plt.legend(fontsize=12)

# 임계점 표시 (예시)
plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
plt.text(1, 0.52, '50% Threshold', color='gray', fontsize=9)

plt.show()