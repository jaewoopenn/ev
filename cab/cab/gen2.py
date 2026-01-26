import csv
import random

# 파일 저장 이름
FILENAME = '/users/jaewoo/data/ev/fluid/ev_jobs.csv'
NUM=10
class EV:
    def __init__(self, id1, arrival, energy, departure):
        self.id = id1
        self.arrival = arrival
        self.energy = energy
        self.departure = departure

def generate_random_evs(num_evs=10):
    """랜덤 EV 생성 로직"""
    evs = []
    print(f"--- 랜덤 EV {num_evs}대 생성 중 ---")
    for i in range(num_evs):
        # Arrival: 0 ~ 15 사이 랜덤
        # a = random.randint(0, 15)
        a=0
        # 주차 시간 (Duration): 최소 1시간 ~ 최대 10시간
        duration = random.randint(1, 10)
        departure = a + duration
        
        # 물리적 한계: 6.6kW * 시간
        max_feasible_energy = duration * 5
        min_req = min(5, max_feasible_energy)
        energy = random.uniform(min_req, max_feasible_energy)
        
        evs.append(EV(i+1, a, energy, departure))
        
    return evs

def save_evs_to_csv(evs, filename):
    """EV 리스트를 CSV 파일로 저장"""
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 헤더 작성
            writer.writerow(['ID', 'Arrival', 'Energy', 'Departure'])
            
            # 데이터 작성
            for ev in evs:
                writer.writerow([ev.id, ev.arrival, ev.energy, ev.departure])
        
        print(f"성공: '{filename}' 파일에 {len(evs)}개의 데이터가 저장되었습니다.")
        
    except IOError as e:
        print(f"오류: 파일을 저장하는 중 문제가 발생했습니다. {e}")

if __name__ == "__main__":
    # 1. 10개의 랜덤 EV 생성
    ev_list = generate_random_evs(NUM)
    
    # 2. 파일로 저장
    save_evs_to_csv(ev_list, FILENAME)