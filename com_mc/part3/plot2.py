import os
import csv
import matplotlib.pyplot as plt

def main():
    result_dir = "/Users/jaewoo/data/com"
    # 주의: 실제 파일명에 맞게 아래 문자열을 수정해 주세요.
    csv_file_path = os.path.join(result_dir, "imc_simulation_3way_results.csv")
    
    if not os.path.exists(csv_file_path):
        print(f"CSV file not found: {csv_file_path}")
        return

    data_by_m = {}

    # CSV 읽어오기
    with open(csv_file_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m_val = int(row["m"])
            if m_val not in data_by_m:
                data_by_m[m_val] = {
                    "targets": [], 
                    "degrade_off": [], 
                    "degrade_single": [],
                    "degrade_chain": []
                }
            
            # x축은 Target, y축은 3가지(OFF, Single, Chain) 결과 라벨로 파싱
            data_by_m[m_val]["targets"].append(float(row["Target"]))
            data_by_m[m_val]["degrade_off"].append(float(row["Degrade_OFF"]))
            data_by_m[m_val]["degrade_single"].append(float(row["Degrade_Single"]))
            data_by_m[m_val]["degrade_chain"].append(float(row["Degrade_Chain"]))

    # m 값별로 PDF 그래프 생성 및 저장
    for m, data in data_by_m.items():
        plt.figure(figsize=(10, 6))
        
        # 3개의 선 플롯 (plot3.py와 동일한 색상, 마커, 선 스타일 적용)
        plt.plot(data["targets"], data["degrade_off"], marker='x', linestyle='--', color='red', linewidth=2, label='Migration OFF')
        plt.plot(data["targets"], data["degrade_single"], marker='o', linestyle='-', color='blue', linewidth=2, label='Migration Single')
        plt.plot(data["targets"], data["degrade_chain"], marker='^', linestyle='-.', color='green', linewidth=2, label='Migration Chain')

        plt.title(f'IMC Multiprocessor (m={m}) Degradation Ratio Comparison', fontsize=14)
        plt.xlabel('Target Utilization / Core', fontsize=12)
        plt.ylabel('Degraded Job Ratio (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(data["targets"])
        
        # y축 범위를 3개 데이터의 최댓값 기준으로 조절
        max_y = max(max(data["degrade_off"]), max(data["degrade_single"]), max(data["degrade_chain"]))
        plt.ylim(-0.5, max_y + 2.0)
        
        plt.tight_layout()
        
        # PDF 파일명도 3-way 결과임을 나타내도록 변경
        pdf_filename = f"imc_simulation_3way_m{m}.pdf"
        pdf_file_path = os.path.join(result_dir, pdf_filename)
        
        plt.savefig(pdf_file_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Plot for m={m} saved successfully to: {pdf_file_path}")

if __name__ == "__main__":
    main()