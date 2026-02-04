import copy

EPSILON = 1e-6

# ---------------------------------------------------------
# 2. 핵심 알고리즘 (Algorithm Logic)
# ---------------------------------------------------------

def calculate_sllf_power(current_time, active_evs, grid_capacity, max_ev_power, min_ev_power, time_step):
    """
    [대조군] sLLF (Smoothed Least Laxity First) - Min Rate Aware
    - 최소 충전 속도(min_ev_power)를 하한선으로 두고 Water-filling 수행
    """
    count = len(active_evs)
    if count == 0: return []

    current_laxities = []
    min_requirements = [] # 각 차량이 최소한 받아야 하는 전력 (제약조건)
    max_acceptables = []  # 각 차량이 물리적으로 받을 수 있는 최대 전력

    for ev in active_evs:
        # Laxity 계산
        remaining_time = ev.deadline - current_time
        time_to_charge = ev.remaining / max_ev_power
        l_t = remaining_time - time_to_charge
        current_laxities.append(l_t)

        # 물리적 한계 (남은 양 vs 최대 속도)
        phys_limit = min(max_ev_power, ev.remaining / time_step)
        max_acceptables.append(phys_limit)
        
        # 최소 충전 속도 제약 (단, 남은 양이 최소 속도보다 적으면 남은 만큼만)
        min_req = min(min_ev_power, phys_limit)
        min_requirements.append(min_req)

    # 1. 기본 보장 전력 할당
    total_min_req = sum(min_requirements)
    
    # (Admission Control에서 걸러지겠지만, 안전장치)
    if total_min_req > grid_capacity + EPSILON:
        # 전력 부족 시 비율대로 축소 (Fail-safe)
        scale = grid_capacity / total_min_req
        return [r * scale for r in min_requirements]

    # 2. 잉여 전력 Water-filling
    remaining_capacity = grid_capacity - total_min_req

    def get_extra_power_for_target_L(target_L):
        total_extra = 0.0
        extras = []
        for i in range(count):
            # 목표 Laxity를 맞추기 위한 요구 전력
            req_p = (max_ev_power / time_step) * (target_L - current_laxities[i] + time_step)
            # 이미 min_req만큼 받았으므로 추가로 받을 수 있는 양 계산
            # 범위: 0 ~ (max_acceptables - min_requirements)
            available_room = max_acceptables[i] - min_requirements[i]
            if available_room < EPSILON:
                alloc_extra = 0.0
            else:
                alloc_extra = max(0.0, min(req_p, available_room))
            
            extras.append(alloc_extra)
            total_extra += alloc_extra
        return total_extra, extras

    min_lax = min(current_laxities)
    max_lax = max(current_laxities)
    low_L, high_L = min_lax - 5.0, max_lax + 5.0
    best_extras = [0.0] * count
    
    # Binary Search for Water Level
    for _ in range(20): 
        mid_L = (low_L + high_L) / 2.0
        p_sum, p_extras = get_extra_power_for_target_L(mid_L)
        if p_sum > remaining_capacity: high_L = mid_L
        else:
            low_L = mid_L
            best_extras = p_extras
            
    # 최종 합산 (최소 요구량 + 추가 할당량)
    final_allocations = []
    for i in range(count):
        final_allocations.append(min_requirements[i] + best_extras[i])
        
    return final_allocations


def calculate_optimal_ras_power(current_time, active_evs, grid_capacity, max_ev_power, min_ev_power, time_step):
    """
    [제안] Optimal S-RAS (Min-Rate Aware)
    - Must Load 계산 시 '마감 기한 충족' AND '최소 속도 충족'을 동시에 고려
    """
    if not active_evs: return []

    # 1. Backward Analysis for Deadline Must Load
    # ---------------------------------------------------
    # 시간축 세그먼트 생성
    segments = []
    first_seg_end = current_time + time_step
    segments.append({"start": current_time, "end": first_seg_end, "capacity": grid_capacity * time_step, "index": 0})
    
    deadlines = sorted(list(set(ev.deadline for ev in active_evs)))
    deadlines = [d for d in deadlines if d > first_seg_end]
    start_t = first_seg_end
    for d in deadlines:
        segments.append({"start": start_t, "end": d, "capacity": grid_capacity * (d - start_t), "index": 1})
        start_t = d

    # 역방향 채우기 (Deadline Must Load 계산)
    deadline_must_energy = {ev.ev_id: 0.0 for ev in active_evs}
    sorted_evs_backward = sorted(active_evs, key=lambda x: x.deadline, reverse=True)
    
    # 세그먼트 복사본 사용
    temp_segments = copy.deepcopy(segments)
    
    for ev in sorted_evs_backward:
        energy_needed = ev.remaining
        for seg in reversed(temp_segments):
            if energy_needed <= EPSILON: break
            if seg["start"] >= ev.deadline: continue
            
            fill = min(energy_needed, seg["capacity"], max_ev_power * (seg["end"] - seg["start"]))
            if fill > EPSILON:
                seg["capacity"] -= fill
                energy_needed -= fill
                if seg["index"] == 0: 
                    deadline_must_energy[ev.ev_id] += fill

    # 2. Combine with Minimum Rate Constraint
    # ---------------------------------------------------
    final_allocations = {}
    total_mandatory = 0.0
    mandatory_map = {}

    for ev in active_evs:
        # A: 마감을 위해 당장 필요한 전력
        p_deadline = deadline_must_energy[ev.ev_id] / time_step
        # B: 최소 충전 속도 제약 (물리적 한계 고려)
        p_min_rate = min(min_ev_power, ev.remaining / time_step)
        
        # 필수 할당량 = Max(A, B)
        # S-RAS의 핵심: 에너지 제약과 정책 제약 중 더 강한 것을 따름
        must_p = max(p_deadline, p_min_rate)
        must_p = min(must_p, max_ev_power) # 최대 속도 캡
        
        mandatory_map[ev.ev_id] = must_p
        total_mandatory += must_p

    # 3. Distribute Surplus (Slot Clearing - EDF)
    # ---------------------------------------------------
    surplus = max(0.0, grid_capacity - total_mandatory)
    sorted_evs_edf = sorted(active_evs, key=lambda x: x.deadline)
    
    for ev in sorted_evs_edf:
        must_p = mandatory_map[ev.ev_id]
        # 더 받을 수 있는 여유 공간
        room = max(0.0, min(max_ev_power, ev.remaining/time_step) - must_p)
        bonus = min(surplus, room)
        
        final_allocations[ev.ev_id] = must_p + bonus
        surplus = max(0.0, surplus - bonus)

    return [final_allocations.get(ev.ev_id, 0.0) for ev in active_evs]


