import numpy as np
from scipy.optimize import linprog
import random
import sys
def check_offline_feasibility(T, P_limit, evs):
    """
    Offline Optimal (LP) 가능 여부 확인
    """
    num_evs = len(evs)
    c = np.zeros(num_evs * T) 
    
    # 1. Power Limit
    A_ub = []
    b_ub = []
    for t in range(T):
        row = np.zeros(num_evs * T)
        for i in range(num_evs):
            row[i * T + t] = 1
        A_ub.append(row)
        b_ub.append(P_limit)
        
    # 2. Energy Demand
    A_eq = []
    b_eq = []
    for i, ev in enumerate(evs):
        row = np.zeros(num_evs * T)
        for t in range(T):
            row[i * T + t] = 1
        A_eq.append(row)
        b_eq.append(ev['e'])
        
    # 3. Bounds
    bounds = []
    for i, ev in enumerate(evs):
        for t in range(T):
            if ev['a'] <= t < ev['d']:
                bounds.append((0, ev['r_bar']))
            else:
                bounds.append((0, 0))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return res.success

def find_minimal_feasible_capacity(T, evs):
    """
    Offline Optimal이 가능한 최소 P를 이분 탐색으로 찾음
    """
    low = 0.0
    high = len(evs) * 1.0 
    optimal_p = high
    
    # 정밀한 P값을 찾기 위해 반복
    for _ in range(25): 
        mid = (low + high) / 2
        if check_offline_feasibility(T, mid, evs):
            optimal_p = mid
            high = mid
        else:
            low = mid
            
    return optimal_p

def solve_sLLF(T, P_limit, evs, verbose=False):
    """ sLLF 시뮬레이션 (Logging 강화 버전) """
    # 시뮬레이션 중 상태 변화를 반영하기 위해 깊은 복사 유사하게 처리
    live_evs = [{'id': i, 'e_rem': float(ev['e']), **ev} for i, ev in enumerate(evs)]
    
    if verbose:
        print(f"\n[Start sLLF Simulation] T={T}, P_lim={P_limit:.2f}, EVs={len(evs)}")

    for t in range(T):
        # 현재 충전 가능한 차량 필터링
        active_indices = [i for i, ev in enumerate(live_evs) if ev['a'] <= t < ev['d'] and ev['e_rem'] > 1e-6]
        
        if not active_indices:
            if verbose:
                print(f"[t={t:02d}] No active EVs.")
            continue
            
        # --- 1. Laxity 계산 ---
        laxities = {}
        if verbose: print(f"[t={t:02d}] Active EVs: {active_indices}")

        for i in active_indices:
            ev = live_evs[i]
            # Laxity = (남은 시간) - (남은 에너지 / 최대 충전 속도) 
            l_i = (ev['d'] - t) - (ev['e_rem'] / ev['r_bar'])
            laxities[i] = l_i
            
            if verbose:
                print(f"  > EV{i}: rem_e={ev['e_rem']:.2f}, rem_t={ev['d']-t}, Laxity={l_i:.4f}")

            # 수학적으로 이미 데드라인 준수가 불가능한 경우
            if l_i < -1e-4: 
                if verbose: print(f"  [!!!] Fail at t={t}: EV{i} has Negative Laxity ({l_i:.4f})")
                return False

        # --- 2. Threshold L 계산 (Binary Search) ---
        # sLLF의 할당은 P(t)와 총 수요 중 작은 값을 목표로 함 
        total_demand_limit = sum([min(live_evs[i]['r_bar'], live_evs[i]['e_rem']) for i in active_indices])
        target_power = min(P_limit, total_demand_limit)
        
        # 탐색 범위를 Laxity 값에 맞춰 동적으로 설정
        current_l_vals = list(laxities.values())
        L_min = min(current_l_vals) - 2.0
        L_max = max(current_l_vals) + 2.0
        
        if L_max - L_min < 10.0:
            L_min -= 5.0
            L_max += 5.0

        best_L = L_min
        
        for _ in range(50):
            L = (L_min + L_max) / 2
            
            # 리스트 컴프리헨션 내 변수 참조 오류 수정
            current_sum = sum([
                max(0, min(live_evs[i]['r_bar'] * (L - laxities[i] + 1), 
                           min(live_evs[i]['r_bar'], live_evs[i]['e_rem'])))
                for i in active_indices
            ])
            
            if current_sum > target_power:
                L_max = L
            else:
                L_min = L
                best_L = L
        
        if verbose:
            print(f"  > Optimization: Target P={target_power:.2f}, Found Threshold L={best_L:.4f}")

        # --- 3. 충전 실행 (배분) ---
        total_assigned = 0
        for i in active_indices:
            ev = live_evs[i]
            # 충전 속도 결정 공식 
            r_val = max(0, min(ev['r_bar'] * (best_L - laxities[i] + 1), min(ev['r_bar'], ev['e_rem'])))
            
            live_evs[i]['e_rem'] -= r_val
            total_assigned += r_val
            
            if verbose:
                print(f"  -> EV{i} Charged: {r_val:.4f} (New Rem_E: {live_evs[i]['e_rem']:.4f})")

        if verbose:
            print(f"  > Total Power Used: {total_assigned:.4f} / {P_limit:.2f}")

    # 4. 결과 판정
    failures = [i for i, ev in enumerate(live_evs) if ev['e_rem'] > 1e-4]
    
    if verbose:
        if failures:
            print(f"\n[Result] FAILED. Unfinished EVs: {failures}")
            for i in failures:
                print(f"  EV{i} Remainder: {live_evs[i]['e_rem']:.4f}")
        else:
            print(f"\n[Result] SUCCESS. All EVs fully charged.")
            
    return len(failures) == 0

def gen2():
    T_horizon=3
    evs=[]
    r_bar=1.0
    a=0
    evs.append({'a': a, 'd': 2, 'e': 1, 'r_bar': r_bar})
    evs.append({'a': a, 'd': 2, 'e': 1, 'r_bar': r_bar})
    evs.append({'a': a, 'd': 3, 'e': 1.5, 'r_bar': r_bar})
    return T_horizon, evs
    
def generate_random_scenario():
    """ 랜덤 시나리오 생성 """
    # num_evs = random.randint(3, 6) # 차량 수 약간 증가
    # T_horizon = random.randint(6, 12) # 시간 범위 확장
    num_evs = random.randint(3, 4) # 차량 수 약간 증가
    T_horizon = random.randint(2, 6) # 시간 범위 확장
    evs = []
    
    for i in range(num_evs):
        # sLLF의 취약점은 Arrival Time이 다를 때 주로 발생하므로
        # 도착 시간(a)을 0으로 고정하지 않고 랜덤하게 부여하는 것이 좋습니다.
        # a = random.randint(0, T_horizon // 2)
        a = random.randint(0, 2)
        # a=0
        
        # 출발 시간은 도착 이후
        min_duration = 1
        if a + min_duration >= T_horizon:
            d = T_horizon
        else:
            d = random.randint(a + min_duration, T_horizon)
            
        r_bar = 1.0
        
        max_possible = (d - a) * r_bar
        # 너무 널널하면 실패 케이스가 안 나오고, 너무 빡빡하면 Offline도 불가능하므로 적절히 설정
        min_demand = max_possible * 0.3 
        e = random.uniform(min_demand, max_possible * 0.9)
        e= int(e)
        
        evs.append({'a': a, 'd': d, 'e': e, 'r_bar': r_bar})
        
    return T_horizon, evs

# --- 메인 루프 ---
print("Searching for sLLF failure cases (checking Offline Feasibility)...")
attempt = 0

T, evs = gen2()
min_feasible_P = find_minimal_feasible_capacity(T, evs)
success = solve_sLLF(T, min_feasible_P, evs, verbose=True)
sys.exit(1)

while True:
    attempt += 1
    T, evs = generate_random_scenario()
    
    # 1. Offline Optimal이 가능한 최소 P 찾기
    min_feasible_P = find_minimal_feasible_capacity(T, evs)
    
    # P가 1.0 이하면 sLLF도 최적이므로 스킵 (멀티프로세서 상황 유도)
    if min_feasible_P <= 1.01:
        continue

    # 2. sLLF 시도
    success = solve_sLLF(T, min_feasible_P, evs, verbose=False)
    
    # 3. 실패 발견 시 상세 검증
    if not success:
        print(f"\n{'='*20} FOUND FAILURE CASE (Attempt {attempt}) {'='*20}")
        print(f"Scenario: T={T}, Optimal Capacity P={min_feasible_P:.5f}")
        for i, ev in enumerate(evs):
            print(f"  EV {i}: Arriv={ev['a']} Deadline={ev['d']}, Demand={ev['e']:.3f}")

        # [검증 1] sLLF 상세 로그 출력
        solve_sLLF(T, min_feasible_P, evs, verbose=True)
        
        # [검증 2] Offline Optimal 재확인
        print(f"\n--- [Verification] Check Offline Optimal Feasibility ---")
        is_offline_possible = check_offline_feasibility(T, min_feasible_P, evs)
        
        if is_offline_possible:
            print(">>> Offline Optimal Result: ✅ SUCCESS (Feasible)")
            print(">>> Conclusion: Offline can schedule this, but sLLF failed.")
        else:
            print(">>> Offline Optimal Result: ❌ IMPOSSIBLE (Something is wrong with calculation)")
            
        break
        
    if attempt % 100 == 0:
        print(f"... scanned {attempt} scenarios ...")