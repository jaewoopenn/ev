import os
import csv
import matplotlib.pyplot as plt

def main():
    result_dir = "/Users/jaewoo/data/com"
    csv_file_path = os.path.join(result_dir, "imc_simulation_results.csv")
    
    # 로컬에서 파일을 못 찾을 경우를 대비한 안전 장치(같은 폴더에 있을 경우)
    if not os.path.exists(csv_file_path):
        csv_file_path = "imc_simulation_results.csv"
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
                    "rates_off": [], 
                    "rates_single": [],
                    "rates_chain": []
                }
            
            # CSV의 실제 컬럼명에 맞추어 3가지 데이터 저장
            data_by_m[m_val]["targets"].append(float(row["Target"]))
            data_by_m[m_val]["rates_off"].append(float(row["Degrade_OFF"]))
            data_by_m[m_val]["rates_single"].append(float(row["Degrade_Single"]))
            data_by_m[m_val]["rates_chain"].append(float(row["Degrade_Chain"]))

    # m 값별로 PDF 그래프 생성
    for m, data in data_by_m.items():
        plt.figure(figsize=(10, 6))
        
        # 3가지 선 그래프 추가 및 마커, 색상 설정
        plt.plot(data["targets"], data["rates_off"], marker='x', linestyle='--', color='red', linewidth=2, label='Degrade OFF')
        plt.plot(data["targets"], data["rates_single"], marker='o', linestyle='-', color='blue', linewidth=2, label='Degrade Single')
        plt.plot(data["targets"], data["rates_chain"], marker='^', linestyle='-.', color='green', linewidth=2, label='Degrade Chain')

        plt.title(f'IMC Multiprocessor (m={m}) Degradation Ratio Comparison', fontsize=14)
        plt.xlabel('Target Utilization / Core', fontsize=12)
        plt.ylabel('Degraded Job Ratio (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(data["targets"])
        
        # y축 범위를 3가지 데이터의 최대값을 기준으로 맞게 조절
        max_y = max(max(data["rates_off"]), max(data["rates_single"]), max(data["rates_chain"]))
        plt.ylim(-0.5, max_y + 2.0)
        
        plt.tight_layout()
        # 화면에 띄우거나 파일로 저장
        # plt.show()
        pdf_file_path = os.path.join(result_dir, f"imc_simulation_plot_m_{m}.pdf")
        plt.savefig(pdf_file_path, format='pdf')

        # plt.savefig(f'plot_m_{m}.png') # 파일로 저장하고 싶다면 주석 해제

if __name__ == "__main__":
    main()