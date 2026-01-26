'''
Created on 2026. 1. 14.

@author: jaewoo
'''
import numpy as np

# ---------------------------------------------------------
# 1. 환경 및 작업 정의 (Word doc 기반 파라미터 [cite: 1, 2, 3, 4])
# ---------------------------------------------------------
# P: 그리드 총 용량 (Grid Capacity) [cite: 2]
# R_max: 개별 차량의 최대 충전 속도 [cite: 3]
# J_i: (r, d, e) -> 도착 시간, 마감 시간, 요구 에너지 [cite: 4]

def generate_random_jobs(n, T, P, R_max, load_factor=1.1):
    """실제 환경을 모사한 랜덤 EV 작업 생성"""
    jobs = []
    for i in range(n):
        r = np.random.uniform(0, T * 0.7)
        d = np.random.uniform(r + 2, T) # 최소 2시간 이상의 여유 보장
        # 개별 밀도가 R_max를 초과하지 않도록 에너지 설정 [cite: 8]
        max_e = (d - r) * R_max
        e = np.random.uniform(max_e * 0.1 * load_factor, max_e * 0.6 * load_factor)
        jobs.append({'id': i, 'r': r, 'd': d, 'e': e})
    return jobs

# ---------------------------------------------------------
# 2. 밀도 기반 충분 조건 테스트 (Density-based Test)
# ---------------------------------------------------------
def density_test(jobs, P, R_max):
    """시뮬레이션 없이 다항 시간(O(N z_log N)) 내에 판단"""
    # 조건 1: 개별 차량이 R_max 내에서 충전 가능한지 확인 [cite: 8]
    for job in jobs:
        if job['e'] / (job['d'] - job['r']) > R_max:
            return False
            
    # 조건 2: 활성 구간별 총 밀도 합이 P를 초과하는지 확인 [cite: 7]
    event_times = sorted(list(set([j['r'] for j in jobs] + [j['d'] for j in jobs])))
    for i in range(len(event_times) - 1):
        t_mid = (event_times[i] + event_times[i+1]) / 2
        # 해당 구간에서 활성화된 모든 작업의 평균 밀도 합산
        current_density_sum = sum(j['e'] / (j['d'] - j['r']) for j in jobs if j['r'] <= t_mid < j['d'])
        if current_density_sum > P:
            return False
    return True

# ---------------------------------------------------------
# 3. Fluid-LLF 시뮬레이션 (Actual Algorithm [cite: 6, 9])
# ---------------------------------------------------------
def fluid_llf_simulation(jobs, P, R_max, dt=0.2):
    """Laxity 순서로 정렬하여 전력을 그리디하게 할당 [cite: 9]"""
    sim_jobs = [{'r': j['r'], 'd': j['d'], 'e_rem': j['e'], 'id': j['id']} for j in jobs]
    max_d = max([j['d'] for j in jobs])
    
    for t in np.arange(0, max_d + dt, dt):
        # 마감 시간 초과 여부 체크
        for j in sim_jobs:
            if t >= j['d'] and j['e_rem'] > 1e-5:
                return False # 스케줄 실패
        
        # 현재 충전이 필요한(Active) 작업 추출
        active = [j for j in sim_jobs if j['r'] <= t < j['d'] and j['e_rem'] > 0]
        if not active: continue
        
        # Laxity 계산 및 정렬 (Laxity = 남은 시간 - 필요 충전 시간)
        for j in active:
            j['laxity'] = (j['d'] - t) - (j['e_rem'] / R_max)
        active.sort(key=lambda x: x['laxity'])
        
        # 전력 할당 (Work-Conserving Greedy Allocation [cite: 9])
        p_rem = P
        for j in active:
            if p_rem <= 0: break
            # 속도 제한: R_max, 남은 전력, 현재 필요한 에너지 중 최소치 [cite: 8]
            rate = min(R_max, p_rem, j['e_rem'] / dt)
            j['e_rem'] -= rate * dt
            p_rem -= rate
            
    return True # 모든 작업 마감 시간 내 완료
