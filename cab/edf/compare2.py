import random
import copy
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. 데이터 구조 정의 (Job)
# ---------------------------------------------------------
@dataclass
class Job:
    job_id: int
    arrival: int
    cost: int
    deadline: int
    remaining: int

# ---------------------------------------------------------
# 2. Job Set 생성기 (프로세서 수에 비례하여 부하 생성)
# ---------------------------------------------------------
def generate_job_set(m_processors):
    # 프로세서가 늘어나면 Job도 늘려야 "부하"가 유지됩니다.
    # 프로세서 당 약 5개의 작업을 처리하도록 설정
    num_jobs = m_processors * 5  
    max_time = 50
    
    jobs = []
    for i in range(num_jobs):
        arrival = random.randint(0, int(max_time * 0.4))
        
        # 무거운 작업(Heavy)과 가벼운 작업(Light)을 섞어서 생성
        if random.random() < 0.3:
            cost = random.randint(5, 15) # 무거운 작업
        else:
            cost = random.randint(1, 5)  # 가벼운 작업
            
        # 마감기한 설정 (Tight ~ Loose 랜덤)
        slack = random.randint(0, cost * 2)
        deadline = arrival + cost + slack
        
        jobs.append(Job(i, arrival, cost, deadline, cost))
    return jobs

# ---------------------------------------------------------
# 3. 스케줄링 시뮬레이션 엔진
# ---------------------------------------------------------
def run_simulation(job_set: List[Job], m_processors: int, algorithm: str) -> bool:
    # 원본 데이터 보호를 위한 깊은 복사
    jobs = copy.deepcopy(job_set)
    max_deadline = max(j.deadline for j in jobs) if jobs else 0
    current_time = 0
    finished_jobs = 0
    total_jobs = len(jobs)

    while finished_jobs < total_jobs:
        # 현재 도착해 있고, 아직 안 끝난 작업들
        ready_queue = [j for j in jobs if j.arrival <= current_time and j.remaining > 0]
        
        # Deadline Miss 체크
        for j in ready_queue:
            if current_time >= j.deadline:
                return False # 스케줄링 실패

        # 알고리즘별 정렬 (우선순위 결정)
        if algorithm == 'EDF':
            # 마감기한이 빠른 순서
            ready_queue.sort(key=lambda x: (x.deadline, x.job_id))
        elif algorithm == 'LLF':
            # 여유시간(Laxity)이 적은 순서
            # Laxity = (Deadline - current_time) - remaining_cost
            ready_queue.sort(key=lambda x: (x.deadline - current_time - x.remaining, x.job_id))

        # 상위 m개 작업 선택 (프로세서 할당)
        active_jobs = ready_queue[:m_processors]
        
        # 1 Time Step 실행
        if active_jobs:
            for job in active_jobs:
                job.remaining -= 1
                if job.remaining == 0:
                    finished_jobs += 1
        
        current_time += 1
        
        # 무한 루프 방지용 (안전장치)
        if current_time > max_deadline + 20: 
            return False

    return True

# ---------------------------------------------------------
# 4. 메인 실험 루프 (x축: 프로세서 수)
# ---------------------------------------------------------
processors_range = list(range(2, 17, 2)) # 2, 4, ..., 16개 코어
edf_ratios = []
llf_ratios = []

trials_per_point = 100 # 각 포인트 당 반복 횟수

print("Simulation Running...")
for m in processors_range:
    edf_wins = 0
    llf_wins = 0
    
    for _ in range(trials_per_point):
        jobs = generate_job_set(m)
        
        if run_simulation(jobs, m, 'EDF'):
            edf_wins += 1
        if run_simulation(jobs, m, 'LLF'):
            llf_wins += 1
            
    edf_ratios.append(edf_wins / trials_per_point)
    llf_ratios.append(llf_wins / trials_per_point)
    print(f"Processors: {m}, EDF: {edf_wins/trials_per_point:.2f}, LLF: {llf_wins/trials_per_point:.2f}")

# ---------------------------------------------------------
# 5. 그래프 그리기
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(processors_range, edf_ratios, marker='o', label='Global EDF', linestyle='-', linewidth=2)
plt.plot(processors_range, llf_ratios, marker='s', label='Global LLF', linestyle='--', linewidth=2)

plt.title('Acceptance Ratio: Global EDF vs Global LLF\n(Scaled Workload: Jobs ~ 5 * m)', fontsize=14)
plt.xlabel('Number of Processors (m)', fontsize=12)
plt.ylabel('Acceptance Ratio', fontsize=12)
plt.ylim(0, 1.1)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(processors_range)

plt.show() # 그래프 출력