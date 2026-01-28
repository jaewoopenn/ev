import os
import random
import math
import pickle
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
DATA_SAVE_PATH = "/Users/jaewoo/data/ev/cab/data"

TRIAL_NUM = 200         # 레벨당 실험 횟수
STRESS_START = 0        # 시작 혼잡도
STRESS_NUM = 10         # 혼잡도 단계 수 (0~9)
MAX_EV_POWER = 1.0      # 고정 상수 (수정 금지)

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
# 2. EV 데이터 생성기 (수정됨)
# ---------------------------------------------------------
def generate_ev_set(congestion_level):
    max_time = 60
    
    # [수정 1] Arrival을 빽빽하게 (Not loose)
    # 기존 0.55 -> 1.2로 상향 조정하여 단위 시간당 도착 차량 수를 늘림
    # congestion_level이 오를수록 훨씬 더 자주 도착함
    base_rate = 0.8
    arrival_rate = base_rate + (congestion_level * 0.15)

    ev_requests = []
    current_time = 0.0
    ev_id = 0
    
    while True:
        # 도착 시간 간격 생성
        inter_arrival = random.expovariate(arrival_rate)
        current_time += inter_arrival
        if current_time > max_time: break
            
        arrival_int = int(current_time)
        
        # [수정 2] 필요 에너지 다양화 (Bimodal Distribution)
        # 50% 확률로 소형차(적은 에너지), 50% 확률로 대형차(많은 에너지) 생성
        if random.random() < 0.5:
            # 소형: 평균 2.0 ~ 4.0 사이
            req_energy = random.gauss(2.0, 1.0)
        else:
            # 대형: 평균 6.0 ~ 10.0 사이
            req_energy = random.gauss(5.0, 2.0)
            
        # 에너지 범위 클리핑 (0.5 ~ 15.0 정도로 범위를 넓혀 다양성 확보)
        req_energy = max(0.5, min(15.0, req_energy))
        req_energy = round(req_energy, 2)
        
        # 최소 충전 시간 계산 (MAX_EV_POWER = 1.0 이므로 에너지값과 동일)
        min_required_time = math.ceil(req_energy / MAX_EV_POWER)
        
        # [수정 3] 데드라인 이원화 (Tight vs Loose)
        # 랜덤하게 '급한 차량'과 '여유로운 차량'을 구분
        is_urgent = random.random() < 0.4  # 40% 확률로 급한 차량
        
        if is_urgent:
            # 급한 경우: 여유 시간(Slack)을 최소 충전 시간의 10%~50% 정도로 아주 짧게 줌
            slack_factor = random.uniform(0.1, 0.5)
            # 최소 1틱의 여유는 보장
            slack = max(1, int(min_required_time * slack_factor))
        else:
            # 여유로운 경우: 여유 시간을 최소 충전 시간의 2배~4배로 넉넉하게 줌
            slack_factor = random.uniform(2.0, 5.0)
            slack = int(min_required_time * slack_factor)
            
        deadline = arrival_int + min_required_time + slack
        
        ev_requests.append(EVRequest(ev_id, arrival_int, req_energy, deadline, req_energy))
        ev_id += 1
        
    return ev_requests

# ---------------------------------------------------------
# 3. 메인 실행: 데이터 생성 및 저장
# ---------------------------------------------------------
if __name__ == '__main__':
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
        level_trials_data = []
        
        for trial in range(TRIAL_NUM):
            ev_set = generate_ev_set(level)
            level_trials_data.append(ev_set)
            
        filename = f"ev_level_{level}.pkl"
        full_path = os.path.join(DATA_SAVE_PATH, filename)
        
        with open(full_path, 'wb') as f:
            pickle.dump(level_trials_data, f)
            
        print(f"[Saved] {filename} (Contains {len(level_trials_data)} trials)")

    print(f"\n[Done] All data files are saved in {DATA_SAVE_PATH}")