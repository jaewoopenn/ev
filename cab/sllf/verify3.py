import json

def solve_sLLF(T, P_limit, evs, verbose=False):
    # 원본 데이터 보존을 위해 복사
    live_evs = [{'id': i, 'e_rem': float(ev['e']), **ev} for i, ev in enumerate(evs)]
    
    for t in range(T):
        # 활성화된 EV 필터링
        active_indices = [i for i, ev in enumerate(live_evs) if ev['a'] <= t < ev['d'] and ev['e_rem'] > 1e-6]
        
        if not active_indices:
            continue
            
        # 1. Laxity 계산
        laxities = {}
        for i in active_indices:
            ev = live_evs[i]
            #  Eq. (4)
            l_i = (ev['d'] - t) - (ev['e_rem'] / ev['r_bar'])
            laxities[i] = l_i
            
            if l_i < -1e-4: 
                if verbose: print(f"  [Fail] t={t}, EV{i} Negative Laxity ({l_i:.4f})")
                return False

        # 2. Threshold L 계산 (Binary Search)
        #  Eq. (17) Right Hand Side
        total_demand_limit = sum([min(live_evs[i]['r_bar'], live_evs[i]['e_rem']) for i in active_indices])
        target_power = min(P_limit, total_demand_limit)
        
        # [수정 2] 탐색 범위를 Laxity 기반으로 동적 설정 (Margin을 넉넉히 줌)
        # L은 대략적인 Laxity 값이므로, 현재 Laxity들의 최소/최대값 범위를 커버해야 함
        current_l_vals = list(laxities.values())
        L_min = min(current_l_vals) - 2.0  # 여유분 추가
        L_max = max(current_l_vals) + 2.0 
        
        # 만약 Laxity 범위가 너무 좁으면 기본 탐색 범위 확장
        if L_max - L_min < 10:
            L_min -= 5
            L_max += 5

        best_L = L_min
        
        for _ in range(50): # 50 iterations is sufficient precision
            L = (L_min + L_max) / 2
            
            # [수정 1] 리스트 컴프리헨션 변수 참조 오류 수정 (ev -> live_evs[i])
            #  Eq. (16) Aggregation
            current_sum = sum([
                max(0, min(live_evs[i]['r_bar'] * (L - laxities[i] + 1), 
                           min(live_evs[i]['r_bar'], live_evs[i]['e_rem'])))
                for i in active_indices
            ])
            
            if current_sum > target_power:
                L_max = L # Sum이 너무 크면 L을 줄여야 함 (rate는 L에 비례하므로)
            else:
                L_min = L
                best_L = L # Feasible한 값 중 가장 큰 값을 찾기 위해 갱신
        
        # 3. 충전 실행
        for i in active_indices:
            ev = live_evs[i]
            #  Eq. (16) Individual Rate
            r_val = max(0, min(ev['r_bar'] * (best_L - laxities[i] + 1), min(ev['r_bar'], ev['e_rem'])))
            live_evs[i]['e_rem'] -= r_val

    # 4. 결과 판정
    failures = [i for i, ev in enumerate(live_evs) if ev['e_rem'] > 1e-4]
    
    return len(failures) == 0

def run_verification():
    filename = 'sllf_failure_cases.json'
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            cases = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found. Please run the generation script first.")
        return

    print(f"{'='*60}")
    print(f"Verifying sLLF Feasibility for {len(cases)} Saved Cases")
    print(f"{'='*60}")
    print(f"{'Case ID':<10} | {'Horizon':<8} | {'Power(P)':<10} | {'Result'}")
    print(f"{'-'*60}")

    success_count = 0
    fail_count = 0

    for case in cases:
        c_id = case['case_id']
        T = case['T_horizon']
        P = case['optimal_P']
        evs = case['evs']
        
        # sLLF 실행
        is_feasible = solve_sLLF(T, P, evs, verbose=False)
        
        result_str = "SUCCESS" if is_feasible else "FAIL"
        if is_feasible:
            success_count += 1
        else:
            fail_count += 1
            
        print(f"{c_id:<10} | {T:<8} | {P:<10.4f} | {result_str}")

    print(f"{'-'*60}")
    print(f"Summary: {success_count} Succeeded, {fail_count} Failed.")
    print(f"Note: Since these were generated as failure cases, 'FAIL' is the expected result.")

if __name__ == "__main__":
    run_verification()