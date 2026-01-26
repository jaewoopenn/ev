'''
Created on 2026. 1. 14.

@author: jaewoo
'''
import numpy as np
import pandas as pd
import time

import fluid_LLF.fluid_basic as fl



PATH='/users/jaewoo/data/ev/fluid/'

def generate_random_jobs(n, T, P, R_max):
    """랜덤 EV 충전 작업 생성 (도착시간, 마감시간, 에너지)"""
    jobs = []
    for i in range(n):
        r = np.random.uniform(0, T * 0.7)
        d = np.random.uniform(r + 2, T) # 최소 2시간 이상의 여유
        # 개별 밀도가 R_max를 넘지 않도록 에너지 설정
        max_e = (d - r) * R_max
        e = np.random.uniform(max_e * 0.2, max_e * 0.8)
        jobs.append({'id': i, 'r': r, 'd': d, 'e': e})
    return jobs

def density_test(jobs, P, R_max):
    """밀도 기반 충분 조건 테스트 (O(N z_log N))"""
    start_time = time.time()
    # 1. 개별 밀도 체크
    for job in jobs:
        if job['e'] / (job['d'] - job['r']) > R_max:
            return False, time.time() - start_time
    
    # 2. 총 밀도 체크 (이벤트 기반 구간 검사)
    event_times = sorted(list(set([j['r'] for j in jobs] + [j['d'] for j in jobs])))
    for i in range(len(event_times) - 1):
        t_mid = (event_times[i] + event_times[i+1]) / 2
        current_density_sum = sum(j['e'] / (j['d'] - j['r']) for j in jobs if j['r'] <= t_mid < j['d'])
        if current_density_sum > P:
            return False, time.time() - start_time
            
    return True, time.time() - start_time

def fluid_llf_simulation(jobs, P, R_max, dt=0.1):
    """Fluid-LLF 시계열 시뮬레이션 (O(T/dt * N z_log N))"""
    start_time = time.time()
    sim_jobs = [{'r': j['r'], 'd': j['d'], 'e_rem': j['e'], 'id': j['id']} for j in jobs]
    max_d = max([j['d'] for j in jobs])
    
    for t in np.arange(0, max_d + dt, dt):
        # 마감 시간 준수 체크
        for j in sim_jobs:
            if t >= j['d'] and j['e_rem'] > 1e-6:
                return False, time.time() - start_time
        
        # 활성 작업 식별 및 Laxity 계산
        active = [j for j in sim_jobs if j['r'] <= t < j['d'] and j['e_rem'] > 0]
        if not active: continue
        
        for j in active:
            j['laxity'] = (j['d'] - t) - (j['e_rem'] / R_max)
        
        # Laxity 순 정렬 및 전력 할당
        active.sort(key=lambda x: x['laxity'])
        p_rem = P
        for j in active:
            if p_rem <= 0: break
            rate = min(R_max, p_rem, j['e_rem'] / dt)
            j['e_rem'] -= rate * dt
            p_rem -= rate
            
    return True, time.time() - start_time

# 실행 파라미터
P, R_max, T, N = 100, 40, 50, 15
iterations = 50
results = []

for i in range(iterations):
    jobs = generate_random_jobs(N, T, P, R_max)
    d_sched, d_time = density_test(jobs, P, R_max)
    s_sched, s_time = fluid_llf_simulation(jobs, P, R_max)
    results.append({'density_ok': d_sched, 'sim_ok': s_sched, 'd_time': d_time, 's_time': s_time})

# 결과 출력 및 저장
df = pd.DataFrame(results)
df.to_csv(PATH+'comparison_results.csv', index=False)