import random
import copy

# 마이그레이트 테스트 s

# ============================================================
# 1. 스케줄러빌리티 수식 및 프로세서
# ============================================================
def compute_x_max(U_LC_A: float, U_LC_D: float, U_HC_H: float):
    denom = U_LC_A - U_LC_D
    if denom <= 0.0:
        if U_HC_H + U_LC_D <= 1.0: return 1.0
        return None
    numer = 1.0 - U_HC_H - U_LC_D
    if numer <= 0.0: return None
    x_max = numer / denom
    if x_max > 1.0: x_max = 1.0
    return x_max

def is_schedulable_new(U_LC_A, U_HC_L, U_LC_D, U_HC_H, hc_tasks, lc_tasks):
    x = compute_x_max(U_LC_A, U_LC_D, U_HC_H)
    if x is None: return False
    lo_sum = U_LC_A
    for (u_l, u_h) in hc_tasks:
        if u_l / x >= u_h:
            lo_sum += u_h 
        else:
            lo_sum += u_l / x
    return lo_sum <= 1.0

class Processor:
    def __init__(self, proc_id, sched_func):
        self.id = proc_id
        self._sched_func = sched_func
        self.U_LC_A = 0.0    
        self.U_HC_L = 0.0    
        self.U_LC_D = 0.0    
        self.U_HC_H = 0.0    
        self.hc_tasks = []   
        self.lc_tasks = []   
        self.tasks = []
        
        # 시뮬레이션 런타임 상태
        self.mode = "LO"          
        self.ready_queue = []     
        self.running_job = None   

    def try_add(self, task: dict) -> bool:
        if task["crit"] == "HC":
            new_U_LC_A, new_U_LC_D = self.U_LC_A, self.U_LC_D
            new_U_HC_L = self.U_HC_L + task["u_LO"]
            new_U_HC_H = self.U_HC_H + task["u_HI"]
            new_hc = self.hc_tasks + [(task["u_LO"], task["u_HI"])]
            new_lc = self.lc_tasks
        else:
            new_U_LC_A = self.U_LC_A + task["u_LO"]
            new_U_LC_D = self.U_LC_D + task["u_HI"]
            new_U_HC_L, new_U_HC_H = self.U_HC_L, self.U_HC_H
            new_hc = self.hc_tasks
            new_lc = self.lc_tasks + [(task["u_LO"], task["u_HI"])]
        return self._sched_func(new_U_LC_A, new_U_HC_L, new_U_LC_D, new_U_HC_H, new_hc, new_lc)

    def add(self, task: dict):
        if task["crit"] == "HC":
            self.U_HC_L += task["u_LO"]
            self.U_HC_H += task["u_HI"]
            self.hc_tasks.append((task["u_LO"], task["u_HI"]))
        else:
            self.U_LC_A += task["u_LO"]
            self.U_LC_D += task["u_HI"]
            self.lc_tasks.append((task["u_LO"], task["u_HI"]))
        self.tasks.append(task)

    def remove(self, task: dict):
        if task in self.tasks:
            self.tasks.remove(task)
            # 유틸리티 재계산
            saved_tasks = list(self.tasks)
            self.U_LC_A = self.U_HC_L = self.U_LC_D = self.U_HC_H = 0.0
            self.hc_tasks = []
            self.lc_tasks = []
            self.tasks = []
            for t in saved_tasks:
                self.add(t)

# ============================================================
# 2. 파티셔닝 알고리즘
# ============================================================
def partition_ffd_new(tasks, m):
    sorted_tasks = sorted(tasks, key=lambda t: max(t["u_LO"], t["u_HI"]), reverse=True)
    procs = [Processor(i, is_schedulable_new) for i in range(m)]
    
    for task in sorted_tasks:
        placed = False
        for p in procs:
            if p.try_add(task):
                p.add(task)
                placed = True
                break
        if not placed:
            return None
    return procs

# ============================================================
# 3. 마이그레이션 정책 비교형 시뮬레이터
# ============================================================
def run_simulation(base_tasks, procs, sim_ticks, allow_migration):
    # 런타임 상태 관리를 위해 태스크 객체에 소속 정보 주입
    runtime_tasks = copy.deepcopy(base_tasks)
    
    # 파티셔닝 결과를 런타임 객체에 매핑
    for rt_task in runtime_tasks:
        # base_tasks와 동일한 ID를 가진 원래 할당된 프로세서 찾기
        original_proc_id = next(p.id for p in procs if any(t["id"] == rt_task["id"] for t in p.tasks))
        proc = next(p for p in procs if p.id == original_proc_id)
        
        rt_task["home_proc"] = proc
        rt_task["current_proc"] = proc

    total_jobs_spawned = 0
    degraded_jobs_count = 0
    
    for tick in range(sim_ticks):
        # 1. Job Release
        for t in runtime_tasks:
            if tick % t["period"] == 0:
                total_jobs_spawned += 1
                new_job = {
                    "task": t,
                    "id": total_jobs_spawned,
                    "deadline": tick + t["period"],
                    "rem_LO": t["c_LO"],
                    "rem_HI": t["c_HI"],
                    "started": False
                }
                
                if t["crit"] == "LC" and t["current_proc"].mode == "HI":
                    degraded_jobs_count += 1
                else:
                    t["current_proc"].ready_queue.append(new_job)

        # 2. EDF 스케줄링 및 모드 로직
        for p in procs:
            if p.running_job:
                is_done = False
                if p.mode == "LO" and p.running_job["rem_LO"] <= 0: is_done = True
                if p.mode == "HI" and p.running_job["rem_HI"] <= 0: is_done = True
                if is_done:
                    p.running_job = None

            # Idle 복귀 로직
            if p.running_job is None and len(p.ready_queue) == 0:
                if p.mode == "HI":
                    p.mode = "LO"
                    if allow_migration:
                        for t in runtime_tasks:
                            if t["home_proc"] == p and t["current_proc"] != p:
                                t["current_proc"].remove(t)
                                p.add(t)
                                t["current_proc"] = p

            # 새 Job 스케줄링 및 20% 확률 모드 스위치
            if p.running_job is None and p.ready_queue:
                p.ready_queue.sort(key=lambda j: j["deadline"])
                p.running_job = p.ready_queue.pop(0)

                if not p.running_job["started"]:
                    p.running_job["started"] = True
                    if p.running_job["task"]["crit"] == "HC" and p.mode == "LO":
                        if random.random() < 0.20:
                            p.mode = "HI"
                            lc_tasks = [t for t in p.tasks if t["crit"] == "LC"]
                            
                            for lc_task in lc_tasks:
                                if allow_migration:
                                    migrated = False
                                    for target_p in procs:
                                        if target_p != p and target_p.mode == "LO" and target_p.try_add(lc_task):
                                            p.remove(lc_task)
                                            target_p.add(lc_task)
                                            lc_task["current_proc"] = target_p
                                            
                                            jobs_to_move = [j for j in p.ready_queue if j["task"]["id"] == lc_task["id"]]
                                            p.ready_queue = [j for j in p.ready_queue if j["task"]["id"] != lc_task["id"]]
                                            target_p.ready_queue.extend(jobs_to_move)
                                            migrated = True
                                            break
                                    
                                    if not migrated: # 빈 코어가 없으면 해당 태스크 Degrade
                                        jobs_to_degrade = [j for j in p.ready_queue if j["task"]["id"] == lc_task["id"]]
                                        p.ready_queue = [j for j in p.ready_queue if j["task"]["id"] != lc_task["id"]]
                                        degraded_jobs_count += len(jobs_to_degrade)
                                        if p.running_job and p.running_job["task"]["id"] == lc_task["id"]:
                                            degraded_jobs_count += 1
                                            p.running_job = None
                                else:
                                    # 마이그레이션 불가: 코어에 속한 모든 LC 즉각 Degrade
                                    jobs_to_degrade = [j for j in p.ready_queue if j["task"]["id"] == lc_task["id"]]
                                    p.ready_queue = [j for j in p.ready_queue if j["task"]["id"] != lc_task["id"]]
                                    degraded_jobs_count += len(jobs_to_degrade)
                                    if p.running_job and p.running_job["task"]["id"] == lc_task["id"]:
                                        degraded_jobs_count += 1
                                        p.running_job = None

            # 실행 진행
            if p.running_job:
                if p.mode == "LO": p.running_job["rem_LO"] -= 1
                else: p.running_job["rem_HI"] -= 1

    return total_jobs_spawned, degraded_jobs_count

# ============================================================
# 4. 종합 테스트 실행기
# ============================================================
def compare_migration_policies():
    m = 4 
    task_count = 16
    sim_ticks = 100000 # 10만 틱 진행
    periods = [20, 50, 100, 200]
    
    print("Generating Task Set & Initial Partitioning...")
    while True:
        base_tasks = []
        for i in range(task_count):
            is_hc = random.random() < 0.5
            period = random.choice(periods)
            u_base = random.uniform(0.05, 0.20)
            
            if is_hc:
                u_hi = u_base
                u_lo = u_hi / random.uniform(1.0, 3.0)
            else:
                u_lo = u_base
                u_hi = random.uniform(0.001, u_lo / 2.0)
                
            base_tasks.append({
                "id": i,
                "crit": "HC" if is_hc else "LC",
                "u_LO": u_lo, "u_HI": u_hi,
                "period": period,
                "c_LO": max(1, int(u_lo * period)),
                "c_HI": max(1, int(u_hi * period))
            })
            
        procs_for_init = partition_ffd_new(copy.deepcopy(base_tasks), m)
        if procs_for_init is not None:
            break

    print(f"Simulation Target: m={m}, tasks={task_count}, ticks={sim_ticks}\n")
    
    # --------------------------------------------------------
    # 시나리오 A: 마이그레이션 허용 (Migration Enabled)
    # --------------------------------------------------------
    random.seed(42) # 비교적 공정한 환경 구성을 위해 시드 고정
    procs_A = partition_ffd_new(copy.deepcopy(base_tasks), m)
    total_A, degraded_A = run_simulation(base_tasks, procs_A, sim_ticks, allow_migration=True)
    ratio_A = (degraded_A / total_A * 100) if total_A > 0 else 0
    
    # --------------------------------------------------------
    # 시나리오 B: 마이그레이션 불가 (Migration Disabled)
    # --------------------------------------------------------
    random.seed(42) # 동일한 난수 시작점 제공
    procs_B = partition_ffd_new(copy.deepcopy(base_tasks), m)
    total_B, degraded_B = run_simulation(base_tasks, procs_B, sim_ticks, allow_migration=False)
    ratio_B = (degraded_B / total_B * 100) if total_B > 0 else 0

    # 결과 출력
    print("=" * 60)
    print(f"{'Metric':<25} | {'Migration ON':<14} | {'Migration OFF':<14}")
    print("-" * 60)
    print(f"{'Total Jobs Executed':<25} | {total_A:<14} | {total_B:<14}")
    print(f"{'Degraded Jobs':<25} | {degraded_A:<14} | {degraded_B:<14}")
    print(f"{'Degradation Ratio (%)':<25} | {ratio_A:<13.2f}% | {ratio_B:<13.2f}%")
    print("=" * 60)
    
    diff = ratio_B - ratio_A
    print(f"\n=> 마이그레이션 허용 시, Degrade 비율이 절대값 기준 {diff:.2f}%p 감소했습니다.")

if __name__ == "__main__":
    compare_migration_policies()