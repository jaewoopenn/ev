import numpy as np
import pandas as pd
import os

base_dir = "/users/jaewoo/data/ev/fluid/data"

# 알고리즘 파라미터 [cite: 2, 3]
P_CAPACITY = 100       
R_MAX = 20             
NUM_JOBS = 100         # 통계적 유의성을 위해 Job 수를 늘림
NUM_INSTANCES = 100    

# 10단계 시나리오 설정
# 후반부(8~10단계)는 이론적 수용 한계를 넘어서는 '가혹(Extreme)' 환경입니다.
scenario_configs = []
for i in range(1, 11):
    # lambda: 0.2(매우 한산) -> 2.5(극도로 혼잡)
    arrival_rate = 0.2 + (i - 1) * 0.25 
    # slack_factor: 2.0(매우 여유) -> 0.05(물리적 한계에 근접)
    slack_factor = max(0.05, 2.0 - (i - 1) * 0.2)
    
    scenario_configs.append({
        "level": f"step_{i:02d}",
        "lambda": arrival_rate,
        "slack_factor": slack_factor
    })

def generate_harsh_jobs(arrival_rate, slack_factor):
    jobs = []
    current_time = 0
    for i in range(NUM_JOBS):
        inter_arrival = np.random.exponential(1/arrival_rate)
        current_time += inter_arrival
        r_i = round(current_time, 2)
        
        # 에너지 요구량 [cite: 4]
        e_i = round(np.random.uniform(20, 60), 2)
        
        # 마감 기한 (d_i): slack_factor가 작을수록 R_max로 쉬지 않고 충전해야 함 [cite: 3, 4]
        min_duration = e_i / R_MAX
        slack = min_duration * np.random.uniform(0.01, slack_factor)
        d_i = round(r_i + min_duration + slack, 2)
        
        jobs.append([i, r_i, d_i, e_i])
    return pd.DataFrame(jobs, columns=['job_id', 'arrival_time', 'deadline', 'energy_requirement'])

# 파일 저장
os.makedirs(base_dir, exist_ok=True)

for config in scenario_configs:
    path = os.path.join(base_dir, config['level'])
    os.makedirs(path, exist_ok=True)
    for n in range(NUM_INSTANCES):
        df = generate_harsh_jobs(config['lambda'], config['slack_factor'])
        df.to_csv(os.path.join(path, f"instance_{n}.csv"), index=False)

print("10단계 가혹 환경 Job 생성이 완료되었습니다.")