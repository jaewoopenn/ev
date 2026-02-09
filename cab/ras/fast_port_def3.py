import copy

EPSILON = 1e-6

# ---------------------------------------------------------
# S-RAS 알고리즘 구현 (Safe-Robust Allocation Strategy)
# Source: robust_r_min.docx
# ---------------------------------------------------------

def calculate_sras_power(current_time, ready_evs, grid_capacity, max_ev_power, min_ev_power, time_step):
    """
    S-RAS (Safe-Robust Allocation Strategy)
    - Phase 1: Backward Analysis for Must-Run Load (Mi)
    - Phase 2: Tier 1 Safety Allocation (Mi vs Imin)
    - Phase 3: Tier 2 Efficiency (Filling Surplus)
    """
    if not ready_evs: return []

    # -----------------------------------------------------
    # 1. 공통 분석 단계 (Common Analysis Phase)
    #    - 역방향 분석(Backward Analysis)을 통해 필수 부하(Mi) 계산
    # -----------------------------------------------------
    
    # 시간축 구간화 (Event-based Discretization)
    # 현재 시점(t)부터 가장 늦은 데드라인까지 구간 생성
    segments = []
    first_seg_end = current_time + time_step
    
    # 첫 번째 구간: [t, t + time_step] -> 여기서 발생하는 수요가 Mi가 됨
    segments.append({
        "start": current_time, 
        "end": first_seg_end, 
        "capacity": grid_capacity * time_step, 
        "index": 0 # Current Step
    })
    
    deadlines = sorted(list(set(ev.deadline for ev in ready_evs)))
    deadlines = [d for d in deadlines if d > first_seg_end]
    
    start_t = first_seg_end
    for d in deadlines:
        segments.append({
            "start": start_t, 
            "end": d, 
            "capacity": grid_capacity * (d - start_t), 
            "index": 1 # Future Step
        })
        start_t = d

    # 역방향 분석 (Step 2: Backward Analysis)
    # 미래 데드라인부터 현재로 역추적하여 필수 부하 계산
    must_run_load = {ev.ev_id: 0.0 for ev in ready_evs}
    
    # 데드라인이 늦은 순서대로 채워나감
    sorted_evs_backward = sorted(ready_evs, key=lambda x: x.deadline, reverse=True)
    temp_segments = copy.deepcopy(segments)
    
    for ev in sorted_evs_backward:
        energy_needed = ev.remaining
        # 가장 늦은 구간부터 채움
        for seg in reversed(temp_segments):
            if energy_needed <= EPSILON: break
            if seg["start"] >= ev.deadline: continue # 데드라인 이후 구간은 사용 불가
            
            # 해당 구간에서 처리 가능한 최대량 (차량 물리적 한계 vs 세그먼트 남은 용량)
            max_processable = max_ev_power * (seg["end"] - seg["start"])
            fill = min(energy_needed, seg["capacity"], max_processable)
            
            if fill > EPSILON:
                seg["capacity"] -= fill
                energy_needed -= fill
                
                # index 0 (현재 타임스텝)에 할당된 양이 곧 Mi
                if seg["index"] == 0: 
                    must_run_load[ev.ev_id] += fill

    # -----------------------------------------------------
    # 2. 실행 단계 (Execution Phase)
    # -----------------------------------------------------
    
    # [Tier 1] 이산적 안전 할당 (Discretized Safety Allocation)
    # 전략: Mi > 0 이면 max(Mi, Imin) 할당
    final_allocations = {ev.ev_id: 0.0 for ev in ready_evs}
    current_load_sum = 0.0
    
    tier1_allocated_ids = set()

    for ev in ready_evs:
        m_i = must_run_load[ev.ev_id] / time_step  # 전력 단위로 변환
        
        if m_i > EPSILON:
            # 필수 부하가 있으면 최소 전류 이상을 보장해야 함
            alloc_req = max(m_i, min_ev_power)
            
            # 물리적 최대 속도 캡 (남은 에너지 고려)
            phys_limit = min(max_ev_power, ev.remaining / time_step)
            alloc_req = min(alloc_req, phys_limit)
            
            final_allocations[ev.ev_id] = alloc_req
            current_load_sum += alloc_req
            tier1_allocated_ids.add(ev.ev_id)
            
    # [Tier 2] 계단식 자원 채우기 (Quantized Filling)
    # 남은 자원 S 계산
    surplus = max(0.0, grid_capacity - current_load_sum)
    
    # 대기 그룹(Waiting)과 활성 그룹(Active) 분류
    waiting_evs = [ev for ev in ready_evs if final_allocations[ev.ev_id] <= EPSILON]
    active_evs = [ev for ev in ready_evs if final_allocations[ev.ev_id] > EPSILON]
    
    # Priority 정렬: EDF (Earliest Deadline First)
    waiting_evs.sort(key=lambda x: x.deadline)
    active_evs.sort(key=lambda x: x.deadline)
    
    # Step 2-1: 신규 가동 (New Activation)
    # 조건: S >= Imin
    # 대기 중인 작업을 실행
    processed_waiting = []
    for ev in waiting_evs:
        if surplus >= min_ev_power - EPSILON:
            # 할당 가능 (최소 전류 할당)
            alloc = min(min_ev_power, max_ev_power, ev.remaining / time_step)
            final_allocations[ev.ev_id] = alloc
            surplus -= alloc
            # 활성 그룹으로 이동
            active_evs.append(ev) 
        else:
            # 더 이상 신규 가동 불가능 (Imin 부족)
            break
            
    # 다시 정렬 (새로 들어온 EV도 포함하여 EDF 적용)
    active_evs.sort(key=lambda x: x.deadline)

    # Step 2-2: 가속 (Speed Up)
    # 조건: S < Imin 또는 대기 작업 없음 (위 루프를 거치면 자연스럽게 만족)
    # 기존 활성화된 작업의 속도를 높임
    for ev in active_evs:
        if surplus <= EPSILON: break
        
        current_alloc = final_allocations[ev.ev_id]
        max_alloc = min(max_ev_power, ev.remaining / time_step)
        
        room = max_alloc - current_alloc
        if room > EPSILON:
            bonus = min(surplus, room)
            final_allocations[ev.ev_id] += bonus
            surplus -= bonus

    # 리스트 형태로 반환 (입력 순서 유지)
    return [final_allocations[ev.ev_id] for ev in ready_evs]