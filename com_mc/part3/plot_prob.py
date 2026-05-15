import os
import csv
import matplotlib.pyplot as plt

def main():
    # 폴더 경로는 기존과 동일하게 유지
    result_dir = "/Users/jaewoo/data/com"
    # 읽어올 CSV 파일 이름 변경
    csv_file_path = os.path.join(result_dir, "imc_prob_3way_results.csv")
    
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
                    "probs": [], 
                    "degrade_off": [], 
                    "degrade_single": [],
                    "degrade_chain": []
                }
            
            # 3-way 결과 라벨에 맞게 데이터 파싱
            data_by_m[m_val]["probs"].append(float(row["Prob"]))
            data_by_m[m_val]["degrade_off"].append(float(row["Degrade_OFF"]))
            data_by_m[m_val]["degrade_single"].append(float(row["Degrade_Single"]))
            data_by_m[m_val]["degrade_chain"].append(float(row["Degrade_Chain"]))

    # m 값별로 PDF 그래프 생성 및 저장
    for m, data in data_by_m.items():
        plt.figure(figsize=(10, 6))
        
        # 3개의 선(Migration OFF, Single, Chain) 플롯
        plt.plot(data["probs"], data["degrade_off"], marker='x', linestyle='--', color='red', linewidth=2, label='Migration OFF')
        plt.plot(data["probs"], data["degrade_single"], marker='o', linestyle='-', color='blue', linewidth=2, label='Migration Single')
        plt.plot(data["probs"], data["degrade_chain"], marker='^', linestyle='-.', color='green', linewidth=2, label='Migration Chain')

        plt.title(f'IMC Multiprocessor (m={m}) - Mode Switch Prob vs Degradation', fontsize=14)
        plt.xlabel('Mode Switch Probability (HC Task Start)', fontsize=12)
        plt.ylabel('Degraded Job Ratio (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(data["probs"])
        
        # y축 범위 설정 (세 가지 데이터 중 최댓값 기준)
        max_y = max(max(data["degrade_off"]), max(data["degrade_single"]), max(data["degrade_chain"]))
        plt.ylim(-0.5, max_y + 2.0)

        # PDF로 저장
        pdf_filename = f"imc_prob_3way_m{m}.pdf"
        pdf_file_path = os.path.join(result_dir, pdf_filename)
        
        plt.savefig(pdf_file_path, format='pdf', bbox_inches='tight')
        print(f"Saved: {pdf_file_path}")
        
        # 메모리 누수 방지를 위해 figure 닫기
        plt.close()

if __name__ == "__main__":
    main()