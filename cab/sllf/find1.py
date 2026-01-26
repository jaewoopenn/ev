import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

# --- 기본 클래스 ---
class EV:
    def __init__(self, ev_id, arrival, deadline, demand, max_rate):
        self.id = ev_id
        self.arrival = arrival
        self.deadline = deadline
        self.total_demand = demand
        self.max_rate = max_rate
        self.remaining_demand = demand

    def __repr__(self):
        return f"EV{self.id}([{self.arrival}-{self.deadline}], E={self.total_demand:.1f}, R={self.max_rate:.1f})"

# --- 1. Offline Solver & Profile Extractor ---
def get_tight_offline_profile(evs, T_max):
    """
    오프라인 최적해를 구하고, 그때 사용된 전력량을 'Grid Limit'으로 리턴합니다.
    """
    G = nx.DiGraph()
    source, sink = 'Source', 'Sink'
    G.add_node(source); G.add_node(sink)
    
    total_demand = sum(ev.total_demand for ev in evs)
    
    for ev in evs:
        ev_node = f'EV_{ev.id}'
        G.add_edge(source, ev_node, capacity=ev.total_demand)
        for t in range(ev.arrival, ev.deadline):
            # weight=t: 가능한 빠른 시간에 충전하도록 유도 (Jagged Profile 생성)
            G.add_edge(ev_node, f'Time_{t}', capacity=ev.max_rate, weight=t)
            
    for t in range(T_max):
        # 초기에는 무한대 용량으로 풀어서 최적의 필요량을 찾음
        G.add_edge(f'Time_{t}', sink, capacity=9999.0)
            
    try:
        # [수정됨] 정확한 함수명: max_flow_min_cost
        flow_dict = nx.max_flow_min_cost(G, source, sink)
        
        # 유효성 검사
        actual_flow = sum(flow_dict[source][f'EV_{ev.id}'] for ev in evs if f'EV_{ev.id}' in flow_dict[source])
        if not np.isclose(actual_flow, total_demand, atol=1e-3):
            return None # Offline도 실패
            
        # 시간대별 사용량(Profile) 추출
        profile = [0.0] * T_max
        for ev in evs:
            ev_node = f'EV_{ev.id}'
            if ev_node in flow_dict:
                for t in range(ev.arrival, ev.deadline):
                    time_node = f'Time_{t}'
                    amount = flow_dict[ev_node].get(time_node, 0.0)
                    profile[t] += amount
        return profile
        
    except nx.NetworkXUnfeasible:
        return None

# --- 2. sLLF Simulation ---
def calculate_laxity(ev, current_time):
    if current_time >= ev.deadline: return -10.0
    if ev.remaining_demand <= 1e-6: return 100.0
    return (ev.deadline - current_time) - (ev.remaining_demand / ev.max_rate)

def solve_sllf(evs_input, T_max, P_schedule):
    evs = copy.deepcopy(evs_input)
    schedule = {t: {} for t in range(T_max)}
    
    for t in range(T_max):
        current_P = P_schedule[t]
        active_evs = [ev for ev in evs if ev.arrival <= t < ev.deadline and ev.remaining_demand > 1e-6]
        if not active_evs: continue

        laxities = {ev.id: calculate_laxity(ev, t) for ev in active_evs}

        # Bisection Search
        def get_total(L_val):
            tot = 0
            for ev in active_evs:
                l_i = laxities[ev.id]
                r = ev.max_rate * (L_val - l_i + 1)
                r = max(0, min(r, min(ev.max_rate, ev.remaining_demand)))
                tot += r
            return tot

        target = min(current_P, sum([min(e.max_rate, e.remaining_demand) for e in active_evs]))
        L_min, L_max = -100.0, 100.0
        for _ in range(50):
            L_mid = (L_min + L_max) / 2
            if get_total(L_mid) < target: L_min = L_mid
            else: L_max = L_mid
        optimal_L = (L_min + L_max) / 2
        
        used = 0
        current_rates = {}
        for ev in sorted(active_evs, key=lambda e: e.id):
            l_i = laxities[ev.id]
            r = ev.max_rate * (optimal_L - l_i + 1)
            r = max(0, min(r, min(ev.max_rate, ev.remaining_demand)))
            current_rates[ev.id] = r
            used += r
            
        if used > current_P + 1e-5:
            scale = current_P / used
            for eid in current_rates: current_rates[eid] *= scale
            
        for ev in active_evs:
            r = current_rates[ev.id]
            ev.remaining_demand -= r
            schedule[t][ev.id] = r
            
    # 잔여량 검사 (1% 이상 남으면 실패)
    failed_evs = [ev for ev in evs if ev.remaining_demand > ev.total_demand * 0.01]
    return len(failed_evs) == 0, failed_evs, schedule

# --- 3. [그림자 밟기] 채굴기 ---
def mine_shadow_trap():
    print("🕵️ Mining for sLLF Failure Cases using 'Shadow Profiling'...")
    attempt = 0
    
    while True:
        attempt += 1
        T_max = 3
        
        # 1. 시나리오 생성 (Rate Bottleneck + Deadline Constraint 혼합)
        evs = []
        # EV0: Heavy & Slow (Requires full slots)
        evs.append(EV(0, 0, 3, 5.0, 2.0))
        # EV1: Fast & Flexible (Distractor)
        evs.append(EV(1, 0, 2, 3.0, 5.0))
        # EV2: Late & Urgent
        evs.append(EV(2, 1, 3, 4.0, 4.0))
        
        # 파라미터 랜덤 변조 (다양한 케이스 탐색)
        for ev in evs:
            ev.total_demand *= np.random.uniform(0.8, 1.2)
            # 물리적 불가능 방지
            max_possible = (ev.deadline - ev.arrival) * ev.max_rate
            if ev.total_demand > max_possible:
                ev.total_demand = max_possible * 0.95

        # 2. [Offline] 최적의 "빡빡한" Grid Limit 추출
        optimal_profile = get_tight_offline_profile(evs, T_max)
        if optimal_profile is None: continue # Offline 불가능하면 패스

        # 3. [Trap] Grid Limit을 Offline이 쓴 만큼만 줌 (여유=0)
        # sLLF는 Smoothing(평활화) 하느라 이 뾰족한 Profile을 못 맞출 확률이 큼
        P_schedule = optimal_profile
        
        # 4. [sLLF] 실행
        sllf_success, failed, schedule = solve_sllf(evs, T_max, P_schedule)
        
        # 5. 실패 시 결과 출력
        if not sllf_success:
            print(f"\n🔥 FOUND FAILURE CASE at Attempt {attempt}!")
            print("=" * 50)
            print(f"Tight Grid Limit: {[round(x,2) for x in P_schedule]}")
            for ev in evs:
                print(f" - {ev}")
            print("-" * 50)
            
            print("Why it failed:")
            print("The Grid Limit was set exactly to the Offline Optimal usage.")
            print("sLLF tried to smooth the charging rates, but the grid had ZERO slack.")
            
            # 그래프
            plt.figure(figsize=(9, 5))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            labels = [f'EV{ev.id}' for ev in evs]
            bottom = np.zeros(T_max)
            
            for i, ev in enumerate(evs):
                rates = [schedule[t].get(ev.id, 0) for t in range(T_max)]
                plt.bar(range(T_max), rates, bottom=bottom, label=labels[i], color=colors[i], width=0.6)
                bottom += np.array(rates)
                
            plt.plot(range(T_max), P_schedule, 'r--o', label='Tight Grid Limit', linewidth=2)
            plt.title("sLLF Failure under Shadow Constraints")
            plt.xlabel("Time Step")
            plt.ylabel("Power")
            plt.xticks(range(T_max))
            plt.legend()
            plt.show()
            
            break # 성공 종료

if __name__ == "__main__":
    mine_shadow_trap()