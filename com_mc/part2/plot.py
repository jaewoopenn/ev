import os
import csv
import matplotlib.pyplot as plt

def main():
    result_dir = "/Users/jaewoo/data/com"
    
    # ============================================================
    # 1. 기존: Utilization variation plots
    # ============================================================
    csv_file_path = os.path.join(result_dir, "simulation_results.csv")
    
    if os.path.exists(csv_file_path):
        data_by_m = {}
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

        for m, data in data_by_m.items():
            plt.figure(figsize=(10, 6))
            plt.plot(data["targets"], data["rates_orig"], marker='o', linestyle='-', color='gray', label='FFD(base)')
            plt.plot(data["targets"], data["rates_ffd_new"], marker='x', linestyle=':', color='blue', label='IMC-PALM(v1)')
            plt.plot(data["targets"], data["rates_mb_new"], marker='s', linestyle='--', color='red', linewidth=2, label='IMC-PALM(v2)')
            plt.plot(data["targets"], data["rates_cu_udp_orig"], marker='^', linestyle='-', color='green', linewidth=2, label='CU-UDP')
            # plt.title(f'Multiprocessor (m={m}) Acceptance Ratio Simulation', fontsize=14)
            plt.xlabel('Target Utilization / Core', fontsize=14)
            plt.ylabel('Acceptance Ratio (%)', fontsize=14)
            plt.legend(fontsize=14)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.xticks(data["targets"])
            plt.ylim(0, 105)
            plt.tight_layout()
            pdf_file_path = os.path.join(result_dir, f"simulation_plot_m_{m}.pdf")
            plt.savefig(pdf_file_path, format='pdf')
            plt.close()
            print(f"[Util] Plot for m={m} saved to: {pdf_file_path}")
    else:
        print(f"CSV not found: {csv_file_path}")

    # ============================================================
    # 2. 추가: P_H variation plots
    # ============================================================
    csv_ph_path = os.path.join(result_dir, "simulation_ph_results.csv")
    
    if os.path.exists(csv_ph_path):
        data_by_m_ph = {}
        with open(csv_ph_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                m_val = int(row["m"])
                if m_val not in data_by_m_ph:
                    data_by_m_ph[m_val] = {
                        "p_h_values": [], "rates_orig": [], "rates_ffd_new": [], 
                        "rates_mb_new": [], "rates_cu_udp_orig": []
                    }
                data_by_m_ph[m_val]["p_h_values"].append(float(row["P_H"]))
                data_by_m_ph[m_val]["rates_orig"].append(float(row["FFD_Orig_Rate"]))
                data_by_m_ph[m_val]["rates_ffd_new"].append(float(row["FFD_New_Rate"]))
                data_by_m_ph[m_val]["rates_mb_new"].append(float(row["MB_New_Rate"]))
                data_by_m_ph[m_val]["rates_cu_udp_orig"].append(float(row["CU_UDP_Orig_Rate"]))

        for m, data in data_by_m_ph.items():
            plt.figure(figsize=(10, 6))
            plt.plot(data["p_h_values"], data["rates_orig"], marker='o', linestyle='-', color='gray', label='FFD(base)')
            plt.plot(data["p_h_values"], data["rates_ffd_new"], marker='x', linestyle=':', color='blue', label='IMC-PALM(v1)')
            plt.plot(data["p_h_values"], data["rates_mb_new"], marker='s', linestyle='--', color='red', linewidth=2, label='IMC-PALM(v2)')
            plt.plot(data["p_h_values"], data["rates_cu_udp_orig"], marker='^', linestyle='-', color='green', linewidth=2, label='CU-UDP')
            # plt.title(f'Multiprocessor (m={m}, Util=0.80) Acceptance Ratio vs P_H', fontsize=14)
            plt.xlabel('Probability of HC Task (P_H)', fontsize=14)
            plt.ylabel('Acceptance Ratio (%)', fontsize=14)
            plt.legend(fontsize=14)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.xticks(data["p_h_values"])
            plt.ylim(0, 105)
            plt.tight_layout()
            pdf_file_path = os.path.join(result_dir, f"simulation_ph_plot_m_{m}.pdf")
            plt.savefig(pdf_file_path, format='pdf')
            plt.close()
            print(f"[P_H] Plot for m={m} saved to: {pdf_file_path}")
    else:
        print(f"CSV not found: {csv_ph_path}")

if __name__ == "__main__":
    main()