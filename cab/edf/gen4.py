import os
import random
import math
import pickle
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------
# 1. 설정 및 상수
# ---------------------------------------------------------
DATA_SAVE_PATH = "/Users/jaewoo/data/ev/cab/data"  # 사용자 지정 경로 유지

TRIAL_NUM = 350         # 레벨당 실험 횟수
STRESS_START = 1        # 시작 혼잡도
STRESS_NUM = 10         # 혼잡도 단계 수 (0~9)
MAX_EV_POWER = 1.0      # 단일 차량 최대 충전 속도

@dataclass
class EVRequest:
    ev_id: int
    arrival: int        
    required_energy: float 
    deadline: int       
    remaining: float    
    is_urgent: bool = False # [New] 긴급 차량 여부 태그 (분석용)

    def __repr__(self):
        tag = "[URGENT]" if self.is_urgent else ""
        return f"EV{self.ev_id}{tag}(A={self.arrival}, Rem={self.remaining:.2f}, D={self.deadline})"

# ---------------------------------------------------------
# 2. EV 데이터 생성기 (S-RAS 성능 검증용 튜닝)
# ---------------------------------------------------------
def generate_ev_set(congestion_level):
    max_time = 60  # 시뮬레이션 지속 시간 (분)
    
    # [Tuning 1] 도착률 (Arrival Rate) 강화
    # 레벨이 오를수록 차량이 더 자주 도착하게 하여 시스템 부하를 높임
    arrival_rate = 0.65 + (congestion_level * 0.08)
    
    # [Tuning 2] 에너지 요구량 (Energy Demand)
    # 평균 요구량도 늘려서 '용량 부족' 상황을 유도
    base_avg_energy = 2.5
    avg_energy = base_avg_energy + (congestion_level * 0.1)
    std_energy = 1.5  # 표준편차를 키워 가끔 '큰 요구량'이 나오게 함
    
    # [Tuning 3] 여유 시간 (Slack) 단축 가속화
    # 레벨이 높을수록 데드라인이 빡빡해짐
    # 기존보다 더 가파르게 감소하도록 조정 (1.9 -> 1.8 시작, 감소폭 0.098 -> 0.12)
    slack_mean_factor = max(0.3, 1.8 - (congestion_level * 0.12))

    # [New] 긴급 차량 등장 확률 (Urgent Probability)
    # 레벨이 높을수록 '도착하자마자 충전해야 하는' 차량 빈도 증가
    # S-RAS는 이를 잘 버티지만, 다른 알고리즘은 여기서 무너짐
    urgent_prob = 0.05 + (congestion_level * 0.02) # 레벨 0: 5% -> 레벨 9: 약 23%

    ev_requests = []
    current_time = 0.0
    ev_id = 0
    
    while True:
        # 포아송 도착 프로세스
        inter_arrival = random.expovariate(arrival_rate)
        current_time += inter_arrival
        if current_time > max_time: break
            
        arrival_int = int(current_time)
        
        # 에너지 요구량 생성 (최소 0.5, 최대 8.0)
        req_energy = random.gauss(avg_energy, std_energy)
        req_energy = max(0.5, min(8.0, req_energy))
        req_energy = round(req_energy, 2)
        
        # 물리적 최소 충전 시간
        min_required_time = math.ceil(req_energy / MAX_EV_POWER)
        
        # Slack 결정 로직
        raw_slack = random.expovariate(1.0) * (min_required_time * slack_mean_factor)
        
        # [Key Logic] 긴급 차량 주입 (Surprise Injection)
        is_urgent_case = False
        if random.random() < urgent_prob:
            # 긴급 차량은 여유 시간이 거의 없음 (최소 시간의 10% 수준)
            raw_slack = raw_slack * 0.1
            is_urgent_case = True
        
        slack = max(0, int(raw_slack)) # 슬랙은 정수 (0 이상)
        deadline = arrival_int + min_required_time + slack
        
        # 요청 객체 생성
        ev_requests.append(EVRequest(
            ev_id=ev_id, 
            arrival=arrival_int, 
            required_energy=req_energy, 
            deadline=deadline, 
            remaining=req_energy,
            is_urgent=is_urgent_case
        ))
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
    
    print(f"Start generating STRESS datasets for S-RAS validation...")
    print(f"Path: {DATA_SAVE_PATH}")
    print(f"Config: {STRESS_NUM} Levels, {TRIAL_NUM} Trials/level")
    print("Feature: Includes 'Urgent EVs' to test pre-emptive bottleneck clearing.\n")

    for level in congestion_levels:
        level_trials_data = []
        
        for trial in range(TRIAL_NUM):
            ev_set = generate_ev_set(level)
            level_trials_data.append(ev_set)
            
        filename = f"ev_level_{level}.pkl"
        full_path = os.path.join(DATA_SAVE_PATH, filename)
        
        with open(full_path, 'wb') as f:
            pickle.dump(level_trials_data, f)
            
        # 통계 정보 간단 출력 (차량 수, 긴급 차량 비율 등)
        avg_cars = sum(len(x) for x in level_trials_data) / len(level_trials_data)
        urgent_count = sum(sum(1 for ev in x if ev.is_urgent) for x in level_trials_data)
        urgent_ratio = (urgent_count / (avg_cars * TRIAL_NUM)) * 100 if avg_cars > 0 else 0
        
        print(f"[Saved] Level {level}: {filename}")
        print(f"   -> Avg EVs: {avg_cars:.1f}, Urgent Ratio: {urgent_ratio:.1f}%")

    print(f"\n[Done] All stress-test data files are saved.")