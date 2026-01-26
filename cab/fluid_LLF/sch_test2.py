import pandas as pd
import matplotlib.pyplot as plt

import fluid_LLF.fluid_basic as fl



# ---------------------------------------------------------
# 4. 혼잡도 단계별 실험 수행 및 시각화
# ---------------------------------------------------------
P_val, R_max_val, T_val = 100, 40, 50
n_range = range(5, 41, 5) # 차량 대수 5대부터 40대까지 증가
iterations = 50
exp_results = []

print("실험 시작: 혼잡도 수준에 따른 스케줄 가능성 분석...")

for n in n_range:
    d_succ, s_succ = 0, 0
    for _ in range(iterations):
        jobs = fl.generate_random_jobs(n, T_val, P_val, R_max_val)
        if fl.density_test(jobs, P_val, R_max_val): d_succ += 1
        if fl.fluid_llf_simulation(jobs, P_val, R_max_val): s_succ += 1
    
    exp_results.append({
        'num_jobs': n,
        'density_rate': d_succ / iterations,
        'sim_rate': s_succ / iterations
    })

# 그래프 그리기
df = pd.DataFrame(exp_results)
plt.figure(figsize=(10, 6))
plt.plot(df['num_jobs'], df['sim_rate'], 'o-', label='Fluid-LLF (Actual Algorithm)', color='blue')
plt.plot(df['num_jobs'], df['density_rate'], 's--', label='Density Test (Sufficient Condition)', color='red')
plt.fill_between(df['num_jobs'], df['density_rate'], df['sim_rate'], color='gray', alpha=0.1, label='Conservativeness Gap')

plt.title('Schedulability Success Rate vs. System Congestion')
plt.xlabel('Number of EV Jobs (N)')
plt.ylabel('Success Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("분석 완료. 그래프를 확인해 주세요.")