import os
import csv
import matplotlib.pyplot as plt

def main():
    result_dir = "/Users/jaewoo/data/com"
    csv_file_path = os.path.join(result_dir, "simulation_results.csv")
    
    if not os.path.exists(csv_file_path):
        print(f"CSV file not found: {csv_file_path}")
        return

    # m 값별로 데이터를 묶어서 저장할 딕셔너리
    data_by_m = {}

    # CSV 읽기
    with open(csv_file_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m_val = int(row["m"])
            if m_val not in data_by_m:
                data_by_m[m_val] = {
                    "targets": [], "rates_orig": [], "rates_ffd_new": [], 
                    "rates_mb_new": [], "rates_cu_udp_orig": []
                }
            
            data_by_m[m_val]["targets"].append(float(row["Target"]))
            data_by_m[m_val]["rates_orig"].append(float(row["FFD_Orig_Rate"]))
            data_by_m[m_val]["rates_ffd_new"].append(float(row["FFD_New_Rate"]))
            data_by_m[m_val]["rates_mb_new"].append(float(row["MB_New_Rate"]))
            data_by_m[m_val]["rates_cu_udp_orig"].append(float(row["CU_UDP_Orig_Rate"]))

    # 각 m 값에 대해 개별 PDF 생성
    for m, data in data_by_m.items():
        plt.figure(figsize=(10, 6))
        
        plt.plot(data["targets"], data["rates_orig"], marker='o', linestyle='-', color='gray', label='1. FFD (Original Math)')
        plt.plot(data["targets"], data["rates_ffd_new"], marker='x', linestyle=':', color='blue', label='2. FFD (New Math)')
        plt.plot(data["targets"], data["rates_mb_new"], marker='s', linestyle='--', color='red', linewidth=2, label='3. MB-FFD (New Math)')
        plt.plot(data["targets"], data["rates_cu_udp_orig"], marker='^', linestyle='-', color='green', linewidth=2, label='4. CU-UDP (Original Math)')

        plt.title(f'Multiprocessor (m={m}) Acceptance Ratio Simulation', fontsize=14)
        plt.xlabel('Target Utilization / Core', fontsize=12)
        plt.ylabel('Acceptance Ratio (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(data["targets"])
        plt.ylim(0, 105)
        plt.tight_layout()
        
        # 파일명에 m을 포함하여 저장
        pdf_file_path = os.path.join(result_dir, f"simulation_plot_m_{m}.pdf")
        plt.savefig(pdf_file_path, format='pdf')
        plt.close() # 다음 그래프를 위해 닫기
        print(f"Plot for m={m} saved successfully to: {pdf_file_path}")

if __name__ == "__main__":
    main()