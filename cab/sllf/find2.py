import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy

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

# --- Offline Solver ---
def solve_offline_max_flow(evs, T_max, P_schedule):
    G = nx.DiGraph()
    source, sink = 'Source', 'Sink'
    G.add_node(source); G.add_node(sink)
    
    total_demand = sum(ev.total_demand for ev in evs)
    
    for ev in evs:
        ev_node = f'EV_{ev.id}'
        G.add_edge(source, ev_node, capacity=ev.total_demand)
        for t in range(ev.arrival, ev.deadline):
            G.add_edge(ev_node, f'Time_{t}', capacity=ev.max_rate)
            
    for t in range(T_max):
        if G.has_node(f'Time_{t}'):
            G.add_edge(f'Time_{t}', sink, capacity=P_schedule[t])
            
    flow_value = nx.maximum_flow_value(G, source, sink)
    return np.isclose(flow_value, total_demand, atol=1e-4)

# --- sLLF Solver ---
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
            
    failed_evs = [ev for ev in evs if ev.remaining_demand > 0.05]
    return len(failed_evs) == 0, failed_evs, schedule

# --- 🏢 현실적 시나리오 채굴기 ---
def mine_realistic_scenario():
    print("Searching for 'Realistic Peak-Shaving' Failure Case...")
    attempt = 0
    
    while True:
        attempt += 1
        
        # 1. 현실적 Grid Profile 설정 (Step Function)
        # T=0,1: 오전 (전력 풍부) -> 12.0
        # T=2,3: 오후 (피크 제어, 전력 감소) -> 4.0 (Grid Drop 발생)
        T_max = 4
        P_schedule = [12.0, 12.0, 4.0, 4.0]
        
        evs = []
        
        # 2. 차량 조합 생성 (장기 주차 vs 단기 주차)
        
        # [EV Group A] 장기 주차 (Long-Stay)
        # - 아침(0)에 와서 오후(4)에 나감.
        # - sLLF는 얘네가 시간이 많다고 생각해서 천천히 충전함.
        # - Offline은 오후에 전력이 줄어드니 오전에 미리 충전함.
        for i in range(2):
            dem = np.random.uniform(10, 14) # 꽤 많은 양
            rate = np.random.uniform(4, 6)  # 적당한 속도
            evs.append(EV(i, 0, 4, dem, rate))
            
        # [EV Group B] 단기 주차 (Short-Stay / Morning Rush)
        # - 아침(0)에 와서 점심(2)에 나감.
        # - 오전에 전력을 같이 씀으로써 sLLF가 장기 주차 차량에 전력을 몰아주지 못하게 방해함.
        for i in range(2, 4):
            dem = np.random.uniform(4, 6)
            rate = np.random.uniform(3, 5)
            evs.append(EV(i, 0, 2, dem, rate))
            
        # 3. 검증
        # Offline은 이 상황을 타개할 수 있어야 함 (오전에 Group A를 미리 충전)
        if not solve_offline_max_flow(evs, T_max, P_schedule):
            continue
            
        # sLLF는 실패해야 함 (오전에 Group A를 미루다가 오후 전력 부족에 걸림)
        sllf_ok, failed, schedule = solve_sllf(evs, T_max, P_schedule)
        
        if not sllf_ok:
            print(f"\n🏢 FOUND REALISTIC SCENARIO at Attempt {attempt}")
            print("=" * 60)
            print(f"Grid Profile (AM -> PM): {P_schedule}")
            print("-" * 60)
            print("EV List:")
            for ev in evs:
                tag = "Long-Stay (Victim)" if ev.deadline == 4 else "Short-Stay (Distractor)"
                print(f" - {tag}: {ev}")
            
            print("-" * 60)
            print("Why it works:")
            print("1. In the morning (t=0,1), Grid is abundant (12.0).")
            print("2. sLLF sees Long-Stay EVs have lots of time, so it shares power equally with Short-Stay EVs.")
            print("   (It creates a 'false sense of security' due to high Laxity).")
            print("3. In the afternoon (t=2,3), Grid drops to 4.0 (Peak Shaving).")
            print("4. Long-Stay EVs still have significant demand left but are choked by the low grid limit.")
            print("5. Offline algorithm would have blasted Long-Stay EVs with full power in the morning.")

            # 그래프 그리기
            plt.figure(figsize=(10, 6))
            colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']
            labels = [f'EV{ev.id} ({ev.deadline-ev.arrival}h)' for ev in evs]
            
            bottom = np.zeros(T_max)
            for i, ev in enumerate(evs):
                rates = [schedule[t].get(ev.id, 0) for t in range(T_max)]
                plt.bar(range(T_max), rates, bottom=bottom, label=labels[i], color=colors[i % 4], width=0.6)
                bottom += np.array(rates)
                
            plt.step(range(T_max), P_schedule, where='mid', color='red', linestyle='--', linewidth=3, label='Grid Limit (Peak Shaving)')
            # 시각적 보정
            plt.hlines(P_schedule, xmin=range(T_max), xmax=np.array(range(T_max))+0.6, colors='red', linestyles='--')

            plt.title("Realistic Peak-Shaving Scenario: sLLF Failure")
            plt.xlabel("Time Slots (0-1: Morning, 2-3: Afternoon)")
            plt.ylabel("Power Allocated (kW)")
            plt.xticks(range(T_max), ['AM 1', 'AM 2', 'PM 1 (Drop)', 'PM 2 (Drop)'])
            plt.legend()
            plt.grid(True, axis='y', alpha=0.3)
            plt.show()
            break

if __name__ == "__main__":
    mine_realistic_scenario()