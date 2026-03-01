import pandas as pd
import matplotlib.pyplot as plt
import os

# 파일 경로 설정
DATA_SAVE_PATH = "/Users/jaewoo/data/com/data/"
INPUT_CSV = os.path.join(DATA_SAVE_PATH, "schedulability_results.csv")
OUTPUT_PNG = os.path.join(DATA_SAVE_PATH, "acceptance_ratio_plot.png")
OUTPUT_PDF = os.path.join(DATA_SAVE_PATH, "acceptance_ratio_plot.pdf")

def plot_acceptance_ratio():
    try:
        # CSV 파일 읽기
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} 파일을 찾을 수 없습니다. 평가 코드를 먼저 실행해주세요.")
        return

    # 그래프 스타일 설정 (논문용으로 깔끔하게)
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6))

    # X축: Utilization
    x = df['Utilization']
    
    # Y축 1: 일반 EDF (파란색, 원형 마커, 점선)
    y_edf = df['EDF_Acceptance_Ratio']
    ax.plot(x, y_edf, 
            marker='o', markersize=8, linestyle='--', linewidth=2, 
            color='blue', label='Standard EDF')

    # Y축 2: EDF-VD (빨간색, 사각형 마커, 실선)
    y_edf_vd = df['EDF_VD_Acceptance_Ratio']
    ax.plot(x, y_edf_vd, 
            marker='s', markersize=8, linestyle='-', linewidth=2, 
            color='red', label='EDF-VD')

    # Y축 3: AMC-max (초록색, 삼각형 마커, 점쇄선) - 새로 추가된 부분
    y_amc = df['AMC_Max_Acceptance_Ratio']
    ax.plot(x, y_amc, 
            marker='^', markersize=8, linestyle='-.', linewidth=2, 
            color='green', label='AMC-max')

    # 축 라벨 및 타이틀 설정
    ax.set_xlabel('Target Utilization ($U$)', fontsize=14)
    ax.set_ylabel('Acceptance Ratio', fontsize=14)
    ax.set_title('Schedulability of Mixed-Criticality Systems', fontsize=16, pad=15)

    # 축 범위 및 눈금 설정
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 그리드 추가 (가독성 향상)
    ax.grid(True, linestyle=':', alpha=0.7)

    # 범례 설정 (3개의 항목이 잘 보이도록 위치 및 크기 조정)
    ax.legend(loc='lower left', fontsize=12, framealpha=0.9)

    # 레이아웃 타이트하게 조정 (여백 최소화)
    plt.tight_layout()

    # 이미지 파일로 저장 (PNG와 고해상도 PDF)
    # plt.savefig(OUTPUT_PNG, dpi=300, format='png')
    plt.savefig(OUTPUT_PDF, format='pdf')
    print(f"세 알고리즘 비교 그래프가 다음 경로에 저장되었습니다:\n- {OUTPUT_PNG}\n- {OUTPUT_PDF}")

if __name__ == "__main__":
    plot_acceptance_ratio()