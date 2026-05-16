import os
import csv
import matplotlib.pyplot as plt

def main():
    result_dir = "/Users/jaewoo/data/com"
    csv_file_path = os.path.join(result_dir, "imc_overhead_results.csv")
    
    if not os.path.exists(csv_file_path):
        print(f"CSV file not found: {csv_file_path}")
        return

    data_by_m = {}

    with open(csv_file_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m_val = int(row["m"])
            if m_val not in data_by_m:
                data_by_m[m_val] = {
                    "alphas": [], 
                    "degrade_off": [], 
                    "degrade_single": [],
                    "degrade_chain": []
                }
            
            data_by_m[m_val]["alphas"].append(float(row["Alpha"]))
            data_by_m[m_val]["degrade_off"].append(float(row["Degrade_OFF"]))
            data_by_m[m_val]["degrade_single"].append(float(row["Degrade_Single"]))
            data_by_m[m_val]["degrade_chain"].append(float(row["Degrade_Chain"]))

    for m, data in data_by_m.items():
        # 1. Broken axis를 위해 2개의 subplot 생성 (위, 아래)
        # 업로드한 이미지 비율에 맞춰 아래쪽 그래프를 약간 더 크게 설정(height_ratios)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 2]})
        fig.subplots_adjust(hspace=0.05)  # 두 그래프 사이의 간격을 좁힘
        
        alpha_pct = [a * 100 for a in data["alphas"]]
        
        # 2. 위쪽 그래프 (ax1): Migration OFF 데이터
        # 참고 이미지에 맞춰 마커를 's'(사각형), 색상을 'tab:red'로 변경
        line1 = ax1.plot(alpha_pct, data["degrade_off"], marker='s', linestyle='--', color='tab:red', linewidth=2, label='Migration OFF')
        off_min, off_max = min(data["degrade_off"]), max(data["degrade_off"])
        ax1.set_ylim(off_min - 0.2, off_max + 0.2)
        
        # 3. 아래쪽 그래프 (ax2): Migration Single, Chain 데이터
        # 참고 이미지에 맞춰 마커와 선 스타일, 색상 변경
        line2 = ax2.plot(alpha_pct, data["degrade_single"], marker='o', linestyle='-', color='tab:blue', linewidth=2, label='Single')
        line3 = ax2.plot(alpha_pct, data["degrade_chain"], marker='^', linestyle='-', color='tab:green', linewidth=2, label='Chain')
        bot_min = min(min(data["degrade_single"]), min(data["degrade_chain"]))
        bot_max = max(max(data["degrade_single"]), max(data["degrade_chain"]))
        ax2.set_ylim(bot_min - 0.1, bot_max + 0.1)

        # 4. Broken Axis 시각적 효과 (테두리 제거 및 사선 추가)
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.tick_params(labelbottom=False, bottom=False)  # 위쪽 그래프 x축 눈금 숨기기
        
        # 끊어진 표시(Cut marks) 그리기 파라미터
        d = 0.015
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)        # Top-left
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right
        
        kwargs.update(transform=ax2.transAxes)        # ax2 기준으로 변경
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right

        # 5. 축 레이블, 타이틀 및 범례 설정
        ax2.set_xlabel('Migration Overhead $\\alpha$ (%)', fontsize=12)
        ax2.set_xticks(alpha_pct)
        
        # y축 레이블을 피규어 전체의 중앙에 배치
        fig.text(0.04, 0.5, 'DJR (%)', va='center', rotation='vertical', fontsize=12)
        ax1.set_title(f'$m={m}$', fontsize=14)

        # 두 그래프의 범례를 합쳐서 아래쪽 그래프의 좌측 상단에 표시
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left', fontsize=10)
        
        ax1.grid(True, linestyle='-', alpha=0.3)
        ax2.grid(True, linestyle='-', alpha=0.3)

        # 파일 저장
        pdf_filename = f"imc_overhead_m{m}.pdf"
        pdf_file_path = os.path.join(result_dir, pdf_filename)
        
        plt.savefig(pdf_file_path, format='pdf', bbox_inches='tight')
        print(f"Saved: {pdf_file_path}")
        
        plt.close()

if __name__ == "__main__":
    main()