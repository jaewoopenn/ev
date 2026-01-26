import numpy as np
from scipy.optimize import linprog
import random
import json

# IS_INTEGER_E=True
IS_INTEGER_E=False

def check_offline_feasibility(T, P_limit, evs):
    """ Offline Optimal (LP) 가능 여부 확인 """
    num_evs = len(evs)
    # 목적 함수 (Feasibility만 보므로 0 벡터)
    c = np.zeros(num_evs * T) 
    
    # 1. Power Limit Constraint (A_ub * x <= b_ub)
    A_ub = []
    b_ub = []
    for t in range(T):
        row = np.zeros(num_evs * T)
        for i in range(num_evs):
            # i번째 EV의 t 시간 충전량 r_i(t)
            row[i * T + t] = 1
        A_ub.append(row)
        b_ub.append(P_limit)
        
    # 2. Energy Demand Constraint (A_eq * x = b_eq)
    A_eq = []
    b_eq = []
    for i, ev in enumerate(evs):
        row = np.zeros(num_evs * T)
        for t in range(T):
            row[i * T + t] = 1
        A_eq.append(row)
        b_eq.append(ev['e'])
        
    # 3. Bounds (충전 속도 제한 및 도착/출발 시간 고려)
    bounds = []
    for i, ev in enumerate(evs):
        for t in range(T):
            if ev['a'] <= t < ev['d']:
                bounds.append((0, ev['r_bar']))
            else:
                bounds.append((0, 0))

    # 'highs' 메소드가 대규모 문제에 더 안정적이고 빠릅니다.
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return res.success

def find_minimal_feasible_capacity(T, evs):
    """ Offline Optimal이 가능한 최소 P를 이분 탐색으로 찾음 """
    # P의 하한은 0, 상한은 모든 EV가 동시에 최대 속도로 충전할 때의 합
    max_possible_load = sum(ev['r_bar'] for ev in evs)
    
    low = 0.0
    high = max_possible_load + 1.0
    optimal_p = high
    
    # 정밀도 조절 (반복 횟수 25회면 소수점 단위까지 충분히 수렴)
    for _ in range(25): 
        mid = (low + high) / 2
        if check_offline_feasibility(T, mid, evs):
            optimal_p = mid
            high = mid
        else:
            low = mid
            
    return optimal_p

def solve_sLLF(T, P_limit, evs, verbose=False):
    """ sLLF 시뮬레이션 (수정된 버전) """
    # 시뮬레이션 중 상태 변화를 반영하기 위해 깊은 복사 유사하게 처리
    live_evs = [{'id': i, 'e_rem': float(ev['e']), **ev} for i, ev in enumerate(evs)]
    
    for t in range(T):
        # 현재 충전 가능한 차량 필터링
        active_indices = [i for i, ev in enumerate(live_evs) if ev['a'] <= t < ev['d'] and ev['e_rem'] > 1e-6]
        
        if not active_indices:
            continue
            
        # 1. Laxity 계산
        laxities = {}
        for i in active_indices:
            ev = live_evs[i]
            # [cite_start]Laxity = (남은 시간) - (남은 에너지 / 최대 충전 속도) [cite: 7]
            l_i = (ev['d'] - t) - (ev['e_rem'] / ev['r_bar'])
            laxities[i] = l_i
            
            # 수학적으로 이미 데드라인 준수가 불가능한 경우
            if l_i < -1e-4: 
                if verbose: print(f"Fail at t={t}: Negative Laxity")
                return False

        # 2. Threshold L 계산 (Binary Search)
        # [cite_start]sLLF의 할당은 P(t)와 총 수요 중 작은 값을 목표로 함 [cite: 4]
        total_demand_limit = sum([min(live_evs[i]['r_bar'], live_evs[i]['e_rem']) for i in active_indices])
        target_power = min(P_limit, total_demand_limit)
        
        # [수정 포인트 1] L의 탐색 범위를 Laxity 값에 맞춰 동적으로 설정
        current_l_vals = list(laxities.values())
        L_min = min(current_l_vals) - 2.0
        L_max = max(current_l_vals) + 2.0
        
        # 범위가 너무 좁을 경우 강제 확장
        if L_max - L_min < 10.0:
            L_min -= 5.0
            L_max += 5.0

        best_L = L_min
        
        for _ in range(50):
            L = (L_min + L_max) / 2
            
            # [수정 포인트 2] 리스트 컴프리헨션 내 변수 참조 오류 수정
            # 이전 코드: max(..., min(ev['r_bar'] ...)) -> 'ev'가 갱신되지 않는 버그
            # 수정 코드: live_evs[i]를 직접 참조
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
        
        # 3. 충전 실행 (배분)
        for i in active_indices:
            ev = live_evs[i]
            r_val = max(0, min(ev['r_bar'] * (best_L - laxities[i] + 1), min(ev['r_bar'], ev['e_rem'])))
            live_evs[i]['e_rem'] -= r_val

    # 4. 결과 판정 (남은 에너지가 허용 오차 이내여야 함)
    failures = [i for i, ev in enumerate(live_evs) if ev['e_rem'] > 1e-4]
    return len(failures) == 0

def generate_random_scenario():
    """ 랜덤 시나리오 생성 """
    num_evs = random.randint(3, 6) # 차량 수 약간 증가
    T_horizon = random.randint(6, 12) # 시간 범위 확장
    evs = []
    
    for i in range(num_evs):
        # sLLF의 취약점은 Arrival Time이 다를 때 주로 발생하므로
        # 도착 시간(a)을 0으로 고정하지 않고 랜덤하게 부여하는 것이 좋습니다.
        a = random.randint(0, T_horizon // 2)
        
        # 출발 시간은 도착 이후
        min_duration = 2
        if a + min_duration >= T_horizon:
            d = T_horizon
        else:
            d = random.randint(a + min_duration, T_horizon)
            
        r_bar = 1.0
        
        max_possible = (d - a) * r_bar
        # 너무 널널하면 실패 케이스가 안 나오고, 너무 빡빡하면 Offline도 불가능하므로 적절히 설정
        min_demand = max_possible * 0.3 
        e = random.uniform(min_demand, max_possible * 0.9)
        if IS_INTEGER_E:
            e=int(e)
        
        evs.append({'a': a, 'd': d, 'e': e, 'r_bar': r_bar})
        
    return T_horizon, evs

# --- 메인 실행 루프 ---
if __name__ == "__main__":
    TARGET_COUNT = 50 # 찾을 실패 케이스 개수
    found_cases = []
    attempt = 0

    print(f"{'='*60}")
    print(f"Searching for {TARGET_COUNT} valid sLLF failure cases...")
    print(f"Conditions: Offline Feasible BUT sLLF Infeasible")
    print(f"{'='*60}")

    while len(found_cases) < TARGET_COUNT:
        attempt += 1
        T, evs = generate_random_scenario()
        
        # 1. Offline Optimal이 가능한 최소 P 찾기 (LP 사용)
        min_feasible_P = find_minimal_feasible_capacity(T, evs)
        
        # P가 너무 작거나(0에 수렴), 너무 크면(제약 없음) 테스트 의미가 적음
        if min_feasible_P < 0.1:
            continue

        # 2. 해당 최소 P 환경에서 sLLF 시도
        # 주의: min_feasible_P는 타이트한 제약조건입니다.
        # sLLF가 Optimal이 아니라면 여기서 실패해야 합니다.
        sllf_success = solve_sLLF(T, min_feasible_P, evs, verbose=False)
        
        # 3. 실패 케이스 발견 시 저장
        # (LP는 성공했는데 sLLF는 실패한 경우)
        if not sllf_success:
            # 안전장치: 정말로 LP가 가능한지 다시 한번 확실히 체크
            if check_offline_feasibility(T, min_feasible_P, evs):
                
                case_data = {
                    'case_id': len(found_cases) + 1,
                    'T_horizon': T,
                    'optimal_P': float(min_feasible_P),
                    'evs': evs
                }
                
                found_cases.append(case_data)
                print(f"[{len(found_cases)}/{TARGET_COUNT}] Found Case! (Scanned {attempt} scenarios)")
            
        if attempt % 1000 == 0:
            print(f"... scanned {attempt} scenarios ...")

    # --- 결과 파일 저장 ---
    filename = 'sllf_failure_cases.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(found_cases, f, indent=4)

    print(f"\nCompleted! Saved {len(found_cases)} cases to '{filename}'.")