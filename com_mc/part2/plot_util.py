import os
import csv
import matplotlib.pyplot as plt

def main():
    result_dir = "/Users/jaewoo/data/com"
    csv_file_path = os.path.join(result_dir, "imc_simulation_results.csv")
    
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
                    "rates_on": [], 
                    "rates_off": []
                }
            
            data_by_m[m_val]["targets"].append(float(row["Target"]))
            data_by_m[m_val]["rates_on"].append(float(row["Degrade_ON_Ratio"]))
            data_by_m[m_val]["rates_off"].append(float(row["Degrade_OFF_Ratio"]))

    # m 값별로 PDF 그래프 생성
    for m, data in data_by_m.items():
        plt.figure(figsize=(10, 6))
        
        plt.plot(data["targets"], data["rates_off"], marker='x', linestyle='--', color='red', linewidth=2, label='Migration OFF')
        plt.plot(data["targets"], data["rates_on"], marker='o', linestyle='-', color='blue', linewidth=2, label='Migration ON')

        plt.title(f'IMC Multiprocessor (m={m}) Degradation Ratio Comparison', fontsize=14)
        plt.xlabel('Target Utilization / Core', fontsize=12)
        plt.ylabel('Degraded Job Ratio (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(data["targets"])
        
        # y축 범위를 데이터에 맞게 약간 여유 있게 조절
        max_y = max(max(data["rates_on"]), max(data["rates_off"]))
        plt.ylim(-0.5, max_y + 2.0)
        
        plt.tight_layout()
        
        pdf_file_path = os.path.join(result_dir, f"imc_simulation_plot_m_{m}.pdf")
        plt.savefig(pdf_file_path, format='pdf')
        plt.close()
        print(f"Plot for m={m} saved successfully to: {pdf_file_path}")

if __name__ == "__main__":
    main()