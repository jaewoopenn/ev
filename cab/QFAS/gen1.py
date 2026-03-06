import os
import random
import math
import pickle
from dataclasses import dataclass

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
# [요청사항] 데이터 저장 경로 글로벌 변수 선언
DATA_SAVE_PATH = "/Users/jaewoo/data/ev/cab/data"

TRIAL_NUM = 350         # 레벨당 실험 횟수
STRESS_START = 2        # 시작 혼잡도
STRESS_NUM = 11         # 혼잡도 단계 수 (0~9)
MAX_EV_POWER = 1.0      # 데이터 생성 시 필요한 상수

@dataclass
class EVRequest:
    ev_id: int
    arrival: int        
    required_energy: float 
    deadline: int       
    remaining: float    

    def __repr__(self):
        return f"EV{self.ev_id}(A={self.arrival}, Rem={self.remaining:.2f}, D={self.deadline})"

# ---------------------------------------------------------
# 2. EV 데이터 생성기 (기존 로직 동일)
# ---------------------------------------------------------
def generate_ev_set(congestion_level):
    max_time = 60
    
    arrival_rate = 0.62 + (congestion_level * 0.075)
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
        if current_time > max_time: break
            
        arrival_int = int(current_time)
        req_energy = random.gauss(avg_energy, std_energy)
        req_energy = max(0.5, min(8.0, req_energy))
        req_energy = round(req_energy, 2)
        
        # 최소 충전 시간 계산
        min_required_time = math.ceil(req_energy / MAX_EV_POWER)
        
        raw_slack = random.expovariate(1.0) * (min_required_time * slack_mean_factor)
        slack = int(raw_slack)
        deadline = arrival_int + min_required_time + slack
        
        ev_requests.append(EVRequest(ev_id, arrival_int, req_energy, deadline, req_energy))
        ev_id += 1
        
    return ev_requests

# ---------------------------------------------------------
# 3. 메인 실행: 데이터 생성 및 저장
# ---------------------------------------------------------
if __name__ == '__main__':
    # 폴더가 없으면 생성
    if not os.path.exists(DATA_SAVE_PATH):
        try:
            os.makedirs(DATA_SAVE_PATH)
            print(f"Directory created: {DATA_SAVE_PATH}")
        except OSError as e:
            print(f"Error creating directory: {e}")
            exit(1)
    
    congestion_levels = range(STRESS_START, STRESS_START + STRESS_NUM)
    
    print(f"Start generating datasets into '{DATA_SAVE_PATH}'...")
    print(f"Configuration: {STRESS_NUM} Levels, {TRIAL_NUM} Trials per level.")

    for level in congestion_levels:
        level_trials_data = []  # [수정] 해당 레벨의 모든 trial을 담을 리스트
        
        # 100번의 Trial 수행
        for trial in range(TRIAL_NUM):
            ev_set = generate_ev_set(level)
            level_trials_data.append(ev_set)
            
        filename = f"ev_level_{level}.pkl"
        full_path = os.path.join(DATA_SAVE_PATH, filename)
        
        with open(full_path, 'wb') as f:
            pickle.dump(level_trials_data, f)
            
        print(f"[Saved] {filename} (Contains {len(level_trials_data)} trials)")

    print(f"\n[Done] All data files are saved in {DATA_SAVE_PATH}")