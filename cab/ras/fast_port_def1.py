EPSILON = 1e-6

# ---------------------------------------------------------
# 2. 핵심 알고리즘
# ---------------------------------------------------------

def calculate_sllf_power(current_time, active_evs, grid_capacity, max_ev_power, time_step):
    """
    [대조군] sLLF (Smoothed Least Laxity First)
    - 입장: LLF (Optimal)
    - 배분: Equalizing (Sub-optimal in Session Limit)
      -> 자리를 빨리 비우지 않고 다 같이 조금씩 충전함.
    """
    count = len(active_evs)
    if count == 0: return []

    current_laxities = []
    phys_limits = []

    for ev in active_evs:
        remaining_time = ev.deadline - current_time
        time_to_charge = ev.remaining / max_ev_power
        l_t = remaining_time - time_to_charge
        current_laxities.append(l_t)
        p_limit = min(max_ev_power, ev.remaining / time_step)
        phys_limits.append(p_limit)

    # 전력이 충분하면 요구량대로 배분
    if sum(phys_limits) <= grid_capacity + EPSILON:
        return phys_limits

    # 부족하면 Water-filling (평준화)
    def get_total_power_for_target_L(target_L):
        total_p = 0.0
        allocs = []
        for i in range(count):
            req_p = (max_ev_power / time_step) * (target_L - current_laxities[i] + time_step)
            req_p = max(0.0, min(req_p, phys_limits[i]))
            allocs.append(req_p)
            total_p += req_p
        return total_p, allocs

    min_lax = min(current_laxities)
    max_lax = max(current_laxities)
    low_L, high_L = min_lax - 5.0, max_lax + 5.0
    best_allocations = [0.0] * count
    
    for _ in range(20): # Binary Search
        mid_L = (low_L + high_L) / 2.0
        p_sum, p_allocs = get_total_power_for_target_L(mid_L)
        if p_sum > grid_capacity: high_L = mid_L
        else:
            low_L = mid_L
            best_allocations = p_allocs
            
    # Scaling
    if sum(best_allocations) > grid_capacity:
        scale = grid_capacity / sum(best_allocations)
        best_allocations = [p * scale for p in best_allocations]
        
    return best_allocations


def calculate_optimal_ras_power(current_time, active_evs, grid_capacity, max_ev_power, time_step):
    """
    [제안] Optimal S-RAS (Session-Aware)
    - 입장: LLF (Optimal Admission - Caller에서 수행)
    - 배분: Must Load + Slot Clearing (Optimal Allocation)
      -> Must Load로 에너지 제약 방어
      -> EDF로 세션 제약 방어 (빨리 내보내기)
    """
    if not active_evs: return []

    # 1. Backward Analysis for Must Load (에너지 생존성)
    # ---------------------------------------------------
    # 구간 생성
    segments = []
    first_seg_end = current_time + time_step
    segments.append({"start": current_time, "end": first_seg_end, "capacity": grid_capacity * time_step, "index": 0})
    
    deadlines = sorted(list(set(ev.deadline for ev in active_evs)))
    deadlines = [d for d in deadlines if d > first_seg_end]
    start_t = first_seg_end
    for d in deadlines:
        segments.append({"start": start_t, "end": d, "capacity": grid_capacity * (d - start_t), "index": 1})
        start_t = d

    # 역방향 채우기
    must_load_energy = {ev.ev_id: 0.0 for ev in active_evs}
    sorted_evs_backward = sorted(active_evs, key=lambda x: x.deadline, reverse=True)
    
    for ev in sorted_evs_backward:
        energy_needed = ev.remaining
        for seg in reversed(segments):
            if energy_needed <= EPSILON: break
            if seg["start"] >= ev.deadline: continue
            
            fill = min(energy_needed, seg["capacity"], max_ev_power * (seg["end"] - seg["start"]))
            if fill > EPSILON:
                seg["capacity"] -= fill
                energy_needed -= fill
                if seg["index"] == 0: must_load_energy[ev.ev_id] += fill

    # 2. Slot-Clearing Allocation (세션 회전율)
    # ---------------------------------------------------
    total_must = sum(must_load_energy.values()) / time_step
    surplus = max(0.0, grid_capacity - total_must)
    
    final_allocations = {}
    
    # 잉여 전력은 '마감이 빠른 순(EDF)'으로 줘서 빨리 내보냄 -> 빈 세션 확보
    sorted_evs_edf = sorted(active_evs, key=lambda x: x.deadline)
    
    for ev in sorted_evs_edf:
        must_p = must_load_energy[ev.ev_id] / time_step
        room = max(0.0, min(max_ev_power, ev.remaining/time_step) - must_p)
        bonus = min(surplus, room)
        
        final_allocations[ev.ev_id] = must_p + bonus
        surplus = max(0.0, surplus - bonus)

    return [final_allocations.get(ev.ev_id, 0.0) for ev in active_evs]
