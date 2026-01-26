import random
import copy
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. Job 데이터 구조 정의
# ---------------------------------------------------------
@dataclass
class Job:
    job_id: int
    arrival: int      # r_i
    cost: int         # C_i
    deadline: int     # d_i
    remaining: int    # 남은 실행 시간 (시뮬레이션용)

    def __repr__(self):
        return f"J{self.job_id}(r={self.arrival}, c={self.cost}, d={self.deadline})"

# ---------------------------------------------------------
# 2. 랜덤 Job Set 생성기
# ---------------------------------------------------------
def generate_job_set(num_jobs, max_time, m_processors):
    jobs = []
    for i in range(num_jobs):
        arrival = random.randint(0, int(max_time * 0.3)) # 초반에 주로 도착하도록 설정
        
        # 실행 시간은 짧은 것과 긴 것을 섞어서 생성 (Dhall's effect 유도 가능성 높임)
        cost = random.randint(1, 10)
        
        # 마감기한은 실행시간보다 커야 함 (Tight ~ Loose 랜덤)
        slack = random.randint(0, 20) 
        deadline = arrival + cost + slack
        
        jobs.append(Job(i, arrival, cost, deadline, cost))
    return jobs

# ---------------------------------------------------------
# 3. 스케줄러 시뮬레이션 엔진 (Time-Stepped)
# ---------------------------------------------------------
def run_simulation(job_set: List[Job], m_processors: int, algorithm: str) -> bool:
    """
    algorithm: 'EDF' or 'LLF'
    Returns: True if schedulable, False if deadline miss
    """
    # 원본 데이터 보존을 위해 깊은 복사
    jobs = copy.deepcopy(job_set)
    
    # 시뮬레이션 종료 시간 계산 (가장 늦은 마감기한)
    max_deadline = max(j.deadline for j in jobs)
    current_time = 0
    
    finished_jobs = 0
    total_jobs = len(jobs)

    while finished_jobs < total_jobs:
        # 1. 현재 시간(current_time)에 도착해 있고, 아직 안 끝난 Job 필터링
        ready_queue = [j for j in jobs if j.arrival <= current_time and j.remaining > 0]
        
        # 2. Deadline Miss 체크
        for j in ready_queue:
            if current_time >= j.deadline:
                return False # 실패

        # 3. 우선순위 정렬 (알고리즘별 로직)
        if algorithm == 'EDF':
            # Deadline이 작은 순서 (같으면 ID순)
            ready_queue.sort(key=lambda x: (x.deadline, x.job_id))
        
        elif algorithm == 'LLF':
            # Laxity = (Deadline - Current_Time) - Remaining_Cost
            # Laxity가 작은 순서 (같으면 ID순 -> Tie-breaking)
            # LLF는 매 틱마다 Laxity가 변하므로 매번 재계산됨
            ready_queue.sort(key=lambda x: (x.deadline - current_time - x.remaining, x.job_id))

        # 4. 프로세서 할당 (상위 m개)
        active_jobs = ready_queue[:m_processors]
        
        # 5. 실행 (1 Time Step 진행)
        # 문맥 교환 오버헤드 0 가정 (Ideal)
        if not active_jobs:
            # 실행할 Job이 없으면 시간만 흐름
            pass
        else:
            for job in active_jobs:
                job.remaining -= 1
                if job.remaining == 0:
                    finished_jobs += 1
        
        # 시간 증가
        current_time += 1
        
        # 무한 루프 방지 (모든 마감기한을 넘겼는데 안 끝난 경우)
        if current_time > max_deadline:
            return False

    return True

# ---------------------------------------------------------
# 4. 메인 실험 루프
# ---------------------------------------------------------
def compare_schedulers(trials=1000, num_jobs=10, m_processors=2):
    edf_success = 0
    llf_success = 0
    
    both_success = 0
    both_fail = 0
    edf_only = 0  # EDF는 성공, LLF는 실패 (Laxity Thrashing 등)
    llf_only = 0  # LLF는 성공, EDF는 실패 (Dhall's Effect 등)

    print(f"--- Simulation Start: {trials} Trials (m={m_processors}) ---")
    
    for _ in range(trials):
        # 랜덤 Job Set 생성
        jobs = generate_job_set(num_jobs, max_time=50, m_processors=m_processors)
        
        # 각 알고리즘 수행
        is_edf_ok = run_simulation(jobs, m_processors, 'EDF')
        is_llf_ok = run_simulation(jobs, m_processors, 'LLF')
        
        # 통계 집계
        if is_edf_ok: edf_success += 1
        if is_llf_ok: llf_success += 1
        
        if is_edf_ok and is_llf_ok: both_success += 1
        elif not is_edf_ok and not is_llf_ok: both_fail += 1
        elif is_edf_ok and not is_llf_ok: edf_only += 1
        elif not is_edf_ok and is_llf_ok: llf_only += 1

    # 결과 출력
    print(f"\n[Result Summary]")
    print(f"Total Trials: {trials}")
    print(f"Processors (m): {m_processors}")
    print(f"-"*30)
    print(f"Global EDF Acceptance Ratio: {edf_success/trials:.2%}")
    print(f"Global LLF Acceptance Ratio: {llf_success/trials:.2%}")
    print(f"-"*30)
    print(f"[Dominance Check]")
    print(f"Both Succeeded : {both_success}")
    print(f"Both Failed    : {both_fail}")
    print(f"EDF OK, LLF Fail: {edf_only}  <-- Proof: LLF does not dominate EDF")
    print(f"LLF OK, EDF Fail: {llf_only}  <-- Proof: EDF does not dominate LLF")

# ---------------------------------------------------------
# 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    # 프로세서 2개, Job 8개 정도로 설정하여 충돌 유도
    compare_schedulers(trials=5000, num_jobs=8, m_processors=2)