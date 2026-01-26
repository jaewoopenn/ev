import json
import numpy as np


def calculate_scheduling_priorities(current_time, active_evs, grid_capacity, max_rate):
    """
    Revised Algorithm:
    1. Mandatory Allocation: 데드라인 임박 순으로 정렬 후, Grid Capacity 내에서 순차 할당
    2. Minimax Allocation: 남은 전력이 있다면 공평 분배
    3. Greedy Allocation: 그래도 남으면 급한 순서로 자투리 소진
    """
    if not active_evs:
        return [], 0.0

    TOLERANCE = 1e-6
    MAX_ITER = 30

    # ---------------------------------------------------------
    # 0. Data Preparation
    # ---------------------------------------------------------
    ids = [ev['id'] for ev in active_evs]
    remains = np.array([ev['remaining_energy'] for ev in active_evs], dtype=float)
    departures = np.array([ev['departure'] for ev in active_evs], dtype=float)
    
    durations = np.maximum(1.0, departures - current_time)
    phys_limits = np.minimum(remains, max_rate)

    # 할당량을 담을 배열 (0으로 초기화)
    allocation = np.zeros_like(remains)
    remaining_grid = grid_capacity

    # ---------------------------------------------------------
    # Step 1: Sequential Mandatory Allocation (Deadline Priority)
    # ---------------------------------------------------------
    # 1. 미래의 모든 턴을 풀로 채워도 부족한 양(Mandatory) 계산
    max_future_charge = (durations - 1.0) * max_rate
    mandatory_needs = np.maximum(0.0, remains - max_future_charge)
    
    # 2. 데드라인(departures)이 빠른 순서대로 인덱스 정렬 (오름차순)
    #    (동일 데드라인일 경우 원래 인덱스 순서 유지)
    sorted_indices = np.argsort(departures)
    
    # 3. 순차 할당
    for idx in sorted_indices:
        # 그리드 전력이 없으면 1차 할당 즉시 종료
        if remaining_grid <= TOLERANCE:
            break
            
        # 이 차량에게 꼭 필요한 양 vs 물리적 한계
        needed = min(mandatory_needs[idx], phys_limits[idx])
        
        if needed > TOLERANCE:
            # 줄 수 있는 만큼만 준다 (필요량 vs 남은 전력 중 작은 값)
            actual_give = min(needed, remaining_grid)
            
            allocation[idx] += actual_give
            remaining_grid -= actual_give

    # ---------------------------------------------------------
    # Step 2: Minimax Optimization (Distribute Surplus)
    # ---------------------------------------------------------
    # 중요: Step 1에서 전력을 다 썼으면(remaining_grid <= 0) 이 단계는 실행되지 않음
    
    residual_needs = remains - allocation
    residual_capacity = phys_limits - allocation
    
    # 남은 전력이 있고, 더 충전할 차량이 있을 때만 수행
    if remaining_grid > TOLERANCE and np.sum(residual_needs) > TOLERANCE:
        
        low, high = -1000.0, 1000.0
        best_extra = np.zeros_like(remains)
        
        for _ in range(MAX_ITER):
            stress_level = (low + high) / 2.0
            target_charge = residual_needs - (stress_level * durations)
            proposed_extra = np.clip(target_charge, 0, residual_capacity)
            
            if np.sum(proposed_extra) > remaining_grid:
                low = stress_level
            else:
                high = stress_level
                best_extra = proposed_extra

        allocation += best_extra
        remaining_grid -= np.sum(best_extra)

    # ---------------------------------------------------------
    # Step 3: Greedy Allocation (Fill the gaps)
    # ---------------------------------------------------------
    if remaining_grid > TOLERANCE:
        urgency_scores = remains / durations
        # 다시 urgency 순으로 정렬 (이미 allocation이 변했으므로 다시 계산 추천)
        # 여기서는 간단히 remains/durations가 큰 순서(밀도가 높은 순)로 처리
        greedy_indices = np.argsort(urgency_scores)[::-1]
        
        for idx in greedy_indices:
            if remaining_grid <= TOLERANCE:
                break
            
            room_to_charge = phys_limits[idx] - allocation[idx]
            if room_to_charge > TOLERANCE:
                top_up = min(remaining_grid, room_to_charge)
                allocation[idx] += top_up
                remaining_grid -= top_up

    # ---------------------------------------------------------
    # 4. Result Formatting
    # ---------------------------------------------------------
    results = []
    total_assigned = 0.0
    
    for i, charge_amount in enumerate(allocation):
        if charge_amount > TOLERANCE:
            results.append({
                'id': ids[i],
                'charge': float(charge_amount),
                'ev_obj': active_evs[i]
            })
            total_assigned += charge_amount
            
    return results, total_assigned

# --- [2] 시뮬레이션 및 데이터 변환 로직 ---
def simulate_new_algorithm(T, P_limit, evs_data):
    """
    저장된 데이터 형식을 'calculate_scheduling_priorities'가 
    요구하는 형식으로 변환하여 시뮬레이션 수행
    """
    
    # 1. 상태 관리를 위한 EV 객체 초기화 (데이터 매핑)
    # JSON의 'e' -> 'remaining_energy', 'd' -> 'departure'
    sim_evs = []
    for i, ev in enumerate(evs_data):
        sim_evs.append({
            'id': i,
            'arrival': ev['a'],
            'departure': ev['d'],
            'remaining_energy': float(ev['e']),
            'r_bar': float(ev.get('r_bar', 1.0)) # 기본값 1.0
        })

    # 2. Time Step Loop
    for t in range(T):
        # 활성화된 EV 필터링 (도착함 AND 떠나지 않음 AND 충전 필요)
        active_evs = [
            ev for ev in sim_evs 
            if ev['arrival'] <= t < ev['departure'] and ev['remaining_energy'] > 1e-6
        ]
        
        if not active_evs:
            continue
            
        # 제공된 알고리즘 호출
        # (주의: max_rate는 생성 시 1.0으로 고정했으므로 1.0 전달)
        results, _ = calculate_scheduling_priorities(
            current_time=t,
            active_evs=active_evs,
            grid_capacity=P_limit,
            max_rate=1.0 
        )
        
        # 결과 반영 (남은 에너지 차감)
        for res in results:
            ev_id = res['id']
            charged_amount = res['charge']
            
            # 해당 EV 찾아서 업데이트
            target_ev = sim_evs[ev_id]
            target_ev['remaining_energy'] -= charged_amount

    # 3. 결과 검증 (모든 EV의 남은 에너지가 거의 0이어야 성공)
    failures = [ev for ev in sim_evs if ev['remaining_energy'] > 1e-4]
    
    return len(failures) == 0, failures


# --- [3] 메인 실행부 ---
def run_comparison():
    filename = 'sllf_failure_cases.json'
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            cases = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return

    print(f"{'='*70}")
    print(f"Testing 'Mandatory+Minimax' Algo on {len(cases)} sLLF Failure Cases")
    print(f"{'='*70}")
    print(f"{'ID':<5} | {'Horizon':<8} | {'Power(P)':<10} | {'sLLF Result':<12} | {'New Algo Result'}")
    print(f"{'-'*70}")

    new_algo_success_count = 0

    for case in cases:
        c_id = case['case_id']
        T = case['T_horizon']
        P = case['optimal_P']
        evs = case['evs']
        
        # 새로운 알고리즘으로 시뮬레이션
        is_success, failures = simulate_new_algorithm(T, P, evs)
        
        result_str = "SUCCESS ✅" if is_success else "FAIL ❌"
        
        if is_success:
            new_algo_success_count += 1
            
        print(f"{c_id:<5} | {T:<8} | {P:<10.4f} | {'FAIL':<12} | {result_str}")

    print(f"{'-'*70}")
    print(f"Summary:")
    print(f"  - Total sLLF Failure Cases Tested: {len(cases)}")
    print(f"  - New Algorithm Success: {new_algo_success_count}")
    print(f"  - New Algorithm Fail: {len(cases) - new_algo_success_count}")
    print(f"{'='*70}")
    
    if new_algo_success_count > 0:
        print("Conclusion: The provided algorithm IS stronger than standard sLLF!")
    else:
        print("Conclusion: The provided algorithm performs similarly to sLLF on these hard cases.")

if __name__ == "__main__":
    run_comparison()