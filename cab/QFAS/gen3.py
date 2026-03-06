import os
import random
import pickle
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. 설정 및 상수  sss
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
# 2. EV 데이터 생성기 (요청하신 로직으로 전면 수정)
# ---------------------------------------------------------
def generate_ev_set(congestion_level):
    """
    요청한 generate_random_evs 로직을 기반으로 EV 데이터를 생성합니다.
    congestion_level이 높을수록 생성되는 차량의 수(num_evs)가 늘어납니다.
    """
    
    # 레벨에 따라 차량 대수 결정 (예: 레벨 0=15대 ~ 레벨 9=33대)
    # 필요에 따라 이 수식(기본 15 + 레벨*2)은 조정 가능합니다.
    num_evs = 15 + (congestion_level * 2)
    
    ev_requests = []
    
    # 요청하신 로직 적용
    # print(f"--- 랜덤 EV {num_evs}대 생성 중 (Wide Distribution) ---")
    
    for i in range(num_evs):
        # [수정 1] 도착 시간을 0~20으로 넓게 퍼뜨림 (Online 특성 강화)
        arrival = random.randint(0, 20)
        
        # [수정 2] 주차 시간을 짧은 것과 긴 것을 섞음 (2 ~ 25)
        # 짧고 급한 차와, 길고 여유로운 차가 섞여야 함
        duration = random.randint(2, 25)
        deadline = arrival + duration
        
        # 해당 주차 시간 동안 최대로 충전 가능한 양
        max_feasible_energy = duration * MAX_EV_POWER
        
        # [수정 3] 에너지 요구량을 Max에 가깝게 설정하여 'Laxity'를 0에 가깝게 만듦 (빡빡한 조건)
        # 최소 50% ~ 최대 100% 요구
        min_req = max_feasible_energy * 0.5 
        req_energy = random.uniform(min_req, max_feasible_energy)
        
        # 소수점 정리 (선택 사항)
        req_energy = round(req_energy, 2)
        
        # EVRequest 객체 생성 (remaining은 초기 required_energy와 동일)
        ev = EVRequest(
            ev_id=i, 
            arrival=arrival, 
            required_energy=req_energy, 
            deadline=deadline, 
            remaining=req_energy
        )
        ev_requests.append(ev)
    
    # 시뮬레이터가 도착 시간 순서대로 처리할 수 있도록 정렬
    ev_requests.sort(key=lambda x: x.arrival)
    
    # ID 재정렬 (정렬 후 ID를 0부터 순차적으로 다시 매기고 싶다면 아래 주석 해제)
    # for idx, ev in enumerate(ev_requests):
    #     ev.ev_id = idx

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
            
        # 첫 번째 trial의 데이터 요약 출력 (디버깅용)
        sample_count = len(level_trials_data[0])
        print(f"[Saved] {filename} (Trials: {len(level_trials_data)}, EVs per set: ~{sample_count})")

    print(f"\n[Done] All data files are saved in {DATA_SAVE_PATH}")