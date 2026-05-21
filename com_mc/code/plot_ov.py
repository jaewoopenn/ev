#!/usr/bin/env python3
# ===========================================================================
#  CONFIG  ---  여기만 바꾸세요. 그 외에는 그냥 실행 버튼만 누르면 됩니다.
# ===========================================================================
CONFIG = {
    # 입력 폴더 (results_overhead_perseed.csv 가 들어있는 곳)
    "INPUT_DIR":  "/Users/jaewoo/data/com",
    # 출력 폴더 (PDF 가 저장될 곳)
    "OUTPUT_DIR": "/Users/jaewoo/data/com",

    # 입력 CSV 파일명 (overhead_significance.py 가 생성한 per-seed 파일)
    "CSV_NAME":   "results_overhead_perseed.csv",

    # 박스플롯에 표시할 구성 (열 이름 -> 라벨/색상). plot_uni.py 와 동일 팔레트.
    #   Rec=blue, NoRec=green, OFF=red (기존 그래프와 100% 동일)
    "SERIES": [
        ("DJR_Rec",   "Migration Rec",   "blue"),
        ("DJR_NoRec", "Migration NoRec", "green"),
    ],
    # OFF 도 같이 그리려면 위 리스트에 ("DJR_OFF","Migration OFF","red") 추가.
    # 단 OFF 는 alpha 에 불변(상수)이라 박스가 거의 점이 되므로 기본 제외.

    "FIGSIZE":   (10, 6),
    "FONTSIZE":  14,
}
# ===========================================================================
#  이 아래로는 수정할 필요 없습니다.
# ===========================================================================
#
#  목적
#  ----
#  overhead_significance.py 가 만든 per-seed CSV
#  (m, Alpha, SeedIdx, MS_Seed, DJR_OFF, DJR_Rec, DJR_NoRec)
#  를 읽어, m 별로 "Migration Overhead alpha" 대비 degraded job
#  ratio 의 30-seed 분포를 박스플롯으로 그린다. 스타일(색/폰트/
#  그리드/저장)은 plot_uni.py 의 overhead 그림과 일치시킨다.
#  각 alpha 위치에 구성별 박스를 나란히(side-by-side) 배치하고,
#  seed 평균을 잇는 선을 겹쳐 그려 기존 라인 그래프와의 연속성을
#  유지한다.

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _box_style(color):
    """plot_uni.py 색을 그대로 쓰되 박스에 맞게 변환."""
    return dict(
        patch_artist=True,
        boxprops=dict(facecolor=color, alpha=0.30,
                      edgecolor=color, linewidth=1.6),
        medianprops=dict(color=color, linewidth=2.0),
        whiskerprops=dict(color=color, linewidth=1.4),
        capprops=dict(color=color, linewidth=1.4),
        flierprops=dict(marker="o", markersize=3,
                        markerfacecolor=color, markeredgecolor=color,
                        alpha=0.5),
    )


def plot_box(cfg):
    csv_path = os.path.join(cfg["INPUT_DIR"], cfg["CSV_NAME"])
    if not os.path.exists(csv_path):
        print(f"[알림] CSV 파일이 없습니다: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
    series = cfg["SERIES"]
    fs = cfg["FONTSIZE"]

    for m in sorted(df["m"].unique()):
        sub = df[df["m"] == m]
        alphas = sorted(sub["Alpha"].unique())
        alpha_pct = [a * 100 for a in alphas]

        plt.figure(figsize=cfg["FIGSIZE"])
        ax = plt.gca()

        n_series = len(series)
        # 각 alpha 슬롯의 폭과, 그 안에서 구성별 박스의 오프셋
        slot = 1.0
        box_w = slot / (n_series + 1)
        base_x = list(range(len(alphas)))  # 0,1,2,... 등간격 슬롯

        for si, (col, label, color) in enumerate(series):
            # 구성별 오프셋 (가운데 정렬)
            off = (si - (n_series - 1) / 2.0) * box_w
            positions = [x + off for x in base_x]
            data = [sub[sub["Alpha"] == a][col].values for a in alphas]

            ax.boxplot(data, positions=positions, widths=box_w * 0.9,
                       **_box_style(color))

            # seed 평균선 (기존 라인 그래프와 동일 마커/스타일 느낌)
            means = [s.mean() for s in data]
            ax.plot(positions, means, color=color, linewidth=2,
                    linestyle="-",
                    marker=("o" if "Rec" in col and "No" not in col
                            else "^"),
                    markersize=6, zorder=5)

        ax.set_xticks(base_x)
        ax.set_xticklabels([f"{p:g}" for p in alpha_pct])
        ax.set_xlabel(r"Migration Overhead $\alpha$ (%)", fontsize=fs)
        ax.set_ylabel("Degraded Job Ratio (%)", fontsize=fs)
        ax.tick_params(labelsize=fs - 1)
        ax.grid(True, linestyle=":", alpha=0.7)

        # 범례: 박스 색과 일치하는 프록시 핸들
        handles = [Line2D([0], [0], color=c, linewidth=3, label=lab)
                   for (_, lab, c) in series]
        ax.legend(handles=[h for h in handles], fontsize=fs,
                  loc="best")

        ax.set_xlim(base_x[0] - 0.5, base_x[-1] + 0.5)

        plt.tight_layout()
        out_name = f"imc_overhead_box_m{m}.pdf"
        out_path = os.path.join(cfg["OUTPUT_DIR"], out_name)
        plt.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"[Box] Plot for m={m} saved: {out_name}")


def main():
    plot_box(CONFIG)


if __name__ == "__main__":
    main()