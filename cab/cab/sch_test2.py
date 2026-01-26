import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

DATA_PATH =  "/users/jaewoo/data/ev/fluid/data/"

# ==========================================
# 1. 전역 설정 및 파라미터 (문서 기반 [cite: 1, 2, 3])
# ==========================================
P_CAPACITY = 100    # Grid Capacity (P) [cite: 2]
R_MAX = 20          # Maximum Charging Rate (R_max) [cite: 3]
TIME_STEP = 1     # 시뮬레이션 시간 해상도
NUM_JOBS = 50       # 인스턴스 당 Job 개수
NUM_INSTANCES = 50  # 시나리오당 테스트 횟수 (정확도를 위해 상향 가능)

# ==========================================
# 2. 10단계 가혹 환경 데이터 생성 함수
# ==========================================
def generate_dataset(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    for i in range(1, 11):
        # 뒤로 갈수록 도착은 빽빽해지고(lambda 증가), 여유 시간(slack)은 줄어듬
        arrival_rate = 0.3 + (i - 1) * 0.25 
        slack_factor = max(0.01, 1.5 - (i - 1) * 0.15)
        
        path = os.path.join(base_dir, f"step_{i:02d}")
        os.makedirs(path, exist_ok=True)
        
        for n in range(NUM_INSTANCES):
            jobs = []
            current_time = 0
            for j_id in range(NUM_JOBS):
                current_time += np.random.exponential(1/arrival_rate)
                r_i = round(current_time, 2) # Arrival time [cite: 4]
                e_i = round(np.random.uniform(20, 50), 2) # Energy Requirement [cite: 4]
                
                # 최소 충전 시간(e_i/R_max)에 기반한 타이트한 마감 기한 설정 [cite: 8]
                min_duration = e_i / R_MAX
                d_i = round(r_i + min_duration + (min_duration * np.random.uniform(0, slack_factor)), 2)
                jobs.append([j_id, r_i, d_i, e_i])
            
            df = pd.DataFrame(jobs, columns=['job_id', 'arrival_time', 'deadline', 'energy_requirement'])
            df.to_csv(os.path.join(path, f"inst_{n}.csv"), index=False)

# ==========================================
# 3. 스케줄링 시뮬레이션 엔진
# ==========================================
def simulate(df, algo_type):
    """
    각 Job 인스턴스에 대해 스케줄 가능 여부(All met deadlines) 판단
    """
    jobs = df.copy()
    jobs['rem_e'] = jobs['energy_requirement']
    
    start_t = jobs['arrival_time'].min()
    end_t = jobs['deadline'].max()
    
    # 시간 단위로 시뮬레이션 진행
    for t in np.arange(start_t, end_t, TIME_STEP):
        # 현재 활성화된(시스템에 있는) 작업들 추출 [cite: 4]
        mask = (jobs['arrival_time'] <= t) & (jobs['deadline'] > t) & (jobs['rem_e'] > 1e-5)
        active_idx = jobs[mask].index.tolist()
        
        if not active_idx: continue
        
        # 1. 우선순위 정렬
        if algo_type == 'EDF':
            active_jobs = jobs.loc[active_idx].sort_values(by='deadline')
        elif algo_type in ['LLF', 'Fluid-LLF']:
            # Laxity 계산: (남은 시간) - (R_max로 충전 시 남은 소요 시간) 
            jobs.loc[active_idx, 'laxity'] = (jobs.loc[active_idx, 'deadline'] - t) - (jobs.loc[active_idx, 'rem_e'] / R_MAX)
            active_jobs = jobs.loc[active_idx].sort_values(by='laxity')
        
        # 2. 전력 할당 (Work-Conserving Greedy Allocation )
        p_remaining = P_CAPACITY # Grid Capacity [cite: 2, 7]
        for idx in active_jobs.index:
            if p_remaining <= 0: break
            
            # 할당량 결정: min(개별 R_max, 남은 Grid 용량, 실제 필요한 전력) [cite: 8, 9]
            alloc = min(R_MAX, p_remaining)
            needed = jobs.loc[idx, 'rem_e'] / TIME_STEP
            actual_alloc = min(alloc, needed)
            
            jobs.loc[idx, 'rem_e'] -= actual_alloc * TIME_STEP
            p_remaining -= actual_alloc

    # 모든 Job이 마감기한(deadline) 내에 에너지를 채웠는지 확인 [cite: 4]
    return (jobs['rem_e'] <= 1e-3).all()

# ==========================================
# 4. 분석 및 시각화 실행
# ==========================================
def run_analysis(data_dir):
    scenarios = sorted(os.listdir(data_dir))
    results = {'EDF': [], 'LLF': [], 'Fluid-LLF': []}
    
    for step in scenarios:
        path = os.path.join(data_dir, step)
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        
        counts = {algo: 0 for algo in results.keys()}
        print(f"Analyzing {step}...")
        
        for f in files:
            df = pd.read_csv(os.path.join(path, f))
            for algo in results.keys():
                if simulate(df, algo):
                    counts[algo] += 1
        
        for algo in results.keys():
            results[algo].append(counts[algo] / len(files))

    # 그래프 출력
    plt.figure(figsize=(10, 6))
    steps = [f"Step {i+1}" for i in range(len(scenarios))]
    plt.plot(steps, results['EDF'], 'o--', label='EDF', alpha=0.6)
    plt.plot(steps, results['LLF'], 's--', label='LLF', alpha=0.6)
    plt.plot(steps, results['Fluid-LLF'], 'd-', label='Fluid-LLF', linewidth=2, color='red')
    
    plt.title("Schedulability Analysis: Acceptance Ratio", fontsize=14)
    plt.xlabel("Traffic Density (Sparse -> Harsh)", fontsize=12)
    plt.ylabel("Success Rate (All Jobs Met Deadline)", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

# 실행 흐름
if __name__ == "__main__":
    # generate_dataset(DATA_PATH)
    run_analysis(DATA_PATH)