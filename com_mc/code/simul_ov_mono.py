#!/usr/bin/env python3
# ===========================================================================
#  CONFIG  ---  여기만 바꾸세요. 그 외에는 그냥 실행 버튼만 누르면 됩니다.
# ===========================================================================
CONFIG = {
    # 원본 파일 경로 -- 본인 환경에 맞게 이 두 줄만 맞춰주세요.
    # (두 파일 모두 그대로 import 해서 로직 재사용. 원본 수정 안 함.
    #  이 스크립트와 같은 폴더에 두면 아래 기본값 그대로 실행 가능.)
    "SIM_PATH":  "simul_unified2.py",
    "GEN_PATH":  "stask_gen.py",

    # 실험 규모
    "M_VALUES":    [2, 4, 8],                       # 코어 수
    "ALPHAS":      [0.0, 0.01, 0.03, 0.05, 0.10],   # 마이그레이션 오버헤드 (Fig.7)
    "U_T":         0.70,                            # 고정 target util (Fig.7)
    "SWITCH_PROB": 0.20,                            # 고정 P^MS (Fig.7)
    "MAX_SETS":    100,                             # m당 태스크셋 수 (논문 1000; 줄이면 빠름)
    "SIM_TICKS":   10000,                           # 논문과 동일
    "N_SEEDS":     30,                              # 독립 mode-switch 시드 반복 (핵심)
    # 대략적 실행 시간(단일 코어 기준): MAX_SETS=100,N_SEEDS=30 일 때
    #   m=2 ~3분, m=4 ~5분, m=8 ~12분  => 합계 약 20분.
    #   더 빠르게: MAX_SETS=50, N_SEEDS=20 (합계 ~7분, 검정력은 다소 하락)
    #   논문 규모: MAX_SETS=1000 (수 시간) — 통상 불필요. seed 수가 핵심.

    # 재현용 고정 시드
    "WL_SEED":      20260518,   # 워크로드 생성 시드 (m마다 +m 됨)
    "MS_SEED_BASE": 1000,       # mode-switch 시드 시작값 (1000,1001,...)
    "BOOT_SEED":    12345,      # 부트스트랩 시드
    "N_BOOT":       20000,      # 부트스트랩 반복

    # 출력 (스크립트 실행 폴더에 생성됨)
    "OUT_SUMMARY": "results_overhead_sig.csv",
    "OUT_PERSEED": "results_overhead_perseed.csv",
}
# ===========================================================================
#  이 아래로는 수정할 필요 없습니다.
# ===========================================================================
#
#  목적
#  ----
#  원본 simul_unified2.py 는 태스크셋마다 random.seed(42) 로 고정한 뒤
#  태스크셋들을 SUM 하여 (m, alpha) 당 단일 비율 하나만 냅니다. 분산이
#  없으므로 Migration NoRec 의 alpha 에 대한 작은(~0.2%p) 비단조성이
#  통계적으로 유의한지 검정할 수 없습니다.
#
#  이 스크립트는:
#   * 원본 stask_gen.py 의 generate_valid_task_set 으로 워크로드 생성
#     (논문 4.1절 절차와 정확히 동일 — 임의 재구현이 아님),
#   * 원본 simul_unified2.py 의 스케줄링/마이그레이션/복구 로직을 그대로
#     import 해서 사용,
#   * 바깥 루프만 변경: mode-switch RNG 를 매 realization 마다 다른
#     시드로 재시드 → 시드별 degraded job ratio 를 표본으로 수집 →
#     동일 워크로드에 대한 paired bootstrap 95% CI 로 NoRec 의
#     alpha=0 -> alpha=1% 변화("the rise")가 유의한지 검정.
#
#  정직성 메모: 두 원본 파일을 그대로 쓰므로 워크로드 분포는 논문과
#  동일하다. 다만 원본 simul 은 set 별 seed(42) 고정·합산이라 "단일
#  실현"이고, 이 스크립트는 그 단일 실현을 시드 반복으로 분포화한 것이다.
#  따라서 특정 시드의 절대 DJR 은 논문 Fig.7 값과 다를 수 있다. 답하려는
#  질문(비단조성이 시드 변동보다 큰가)은 절대값과 무관하므로 결론은 유효.

import os
import sys
import csv
import copy
import random
import statistics
import importlib.util


def _load_module(path, modname):
    """원본 .py 를 경로로 import. 임포트 시 원본 main() 자동 실행만 차단."""
    src = open(path).read()
    src = src.replace('if __name__ == "__main__":\n    main()',
                       'if __name__ == "__main__":\n    pass')
    spec = importlib.util.spec_from_loader(modname, loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def build_prepared(gen, m, U_t, n_sets, wl_seed):
    """원본 stask_gen.generate_valid_task_set 로 워크로드 생성.
    generate_valid_task_set 은 전역 random 모듈을 쓰므로
    재현성을 위해 wl_seed 로 고정한 뒤 생성한다."""
    random.seed(wl_seed)
    prepared = []
    for _ in range(n_sets):
        ts = gen.generate_valid_task_set(m, U_t)   # id/period/c_LO/c_HI 주입 포함
        prepared.append(copy.deepcopy(ts))
    return prepared


def eval_point_seeded(sim, prepared, m, sim_ticks, switch_prob, alpha,
                      ms_seed):
    """원본 eval_point 과 동일한 집계. 단 mode-switch RNG 를 ms_seed 로
    시드(원본은 42 고정). 시스템 전체 DJR(%) 를 mode 별로 반환."""
    acc = {mode: {"total": 0, "degrade": 0}
           for mode in ["off", "mig_rec", "mig_norec"]}
    for task_set in prepared:
        for mode in ["off", "mig_rec", "mig_norec"]:
            random.seed(ms_seed)            # <-- 원본 대비 유일한 변경점
            t_total, d_total = sim.run_simulation(
                task_set, m, sim_ticks,
                mig_mode=mode, switch_prob=switch_prob, mig_alpha=alpha)
            acc[mode]["total"] += t_total
            acc[mode]["degrade"] += d_total
    if acc["off"]["total"] == 0:
        return None
    return {
        "off":   100.0 * acc["off"]["degrade"]       / acc["off"]["total"],
        "rec":   100.0 * acc["mig_rec"]["degrade"]   / acc["mig_rec"]["total"],
        "norec": 100.0 * acc["mig_norec"]["degrade"] / acc["mig_norec"]["total"],
    }


def paired_bootstrap_ci(diffs, n_boot, ci, seed):
    """paired 차이 표본 평균의 percentile bootstrap CI."""
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot)]
    return (statistics.fmean(diffs), lo, hi)


def main():
    C = CONFIG
    sim = _load_module(C["SIM_PATH"], "simul_unified2")
    gen = _load_module(C["GEN_PATH"], "stask_gen")
    _outdir = os.path.dirname(C["OUT_SUMMARY"])
    if _outdir:
        os.makedirs(_outdir, exist_ok=True)

    alphas = C["ALPHAS"]
    per_seed_rows = []
    summary_rows = []

    for m in C["M_VALUES"]:
        # 같은 m 에서는 모든 (alpha, seed) 가 동일 워크로드 공유 →
        # alpha 만 변하는 paired 설계.
        prepared = build_prepared(gen, m, C["U_T"],
                                  C["MAX_SETS"], C["WL_SEED"] + m)
        if not prepared:
            print(f"[warn] m={m}: 워크로드 생성 실패", file=sys.stderr)
            continue
        print(f"[info] m={m}: {len(prepared)}개 태스크셋 준비 완료",
              file=sys.stderr)

        norec = {a: [] for a in alphas}
        rec = {a: [] for a in alphas}
        off = {a: [] for a in alphas}

        for si in range(C["N_SEEDS"]):
            ms_seed = C["MS_SEED_BASE"] + si
            for a in alphas:
                r = eval_point_seeded(sim, prepared, m, C["SIM_TICKS"],
                                      C["SWITCH_PROB"], a, ms_seed)
                if r is None:
                    continue
                norec[a].append(r["norec"])
                rec[a].append(r["rec"])
                off[a].append(r["off"])
                per_seed_rows.append([m, a, si, ms_seed,
                                      r["off"], r["rec"], r["norec"]])
            print(f"[info] m={m} seed#{si+1}/{C['N_SEEDS']} 완료",
                  file=sys.stderr)

        for a in alphas:
            if not norec[a]:
                continue
            summary_rows.append([
                m, a, len(norec[a]),
                statistics.fmean(off[a]),
                statistics.fmean(rec[a]),
                statistics.pstdev(rec[a]) if len(rec[a]) > 1 else 0.0,
                statistics.fmean(norec[a]),
                statistics.pstdev(norec[a]) if len(norec[a]) > 1 else 0.0,
            ])

        # ---- 핵심 검정: NoRec 비단조성 ----
        if len(alphas) >= 2 and norec[alphas[0]] and norec[alphas[1]]:
            a0, a1, amax = alphas[0], alphas[1], alphas[-1]
            n = min(len(norec[a0]), len(norec[a1]))
            d_rise = [norec[a1][i] - norec[a0][i] for i in range(n)]
            mean_r, lo_r, hi_r = paired_bootstrap_ci(
                d_rise, C["N_BOOT"], 0.95, C["BOOT_SEED"])
            n2 = min(len(norec[a1]), len(norec[amax]))
            d_fall = [norec[a1][i] - norec[amax][i] for i in range(n2)]
            mean_f, lo_f, hi_f = paired_bootstrap_ci(
                d_fall, C["N_BOOT"], 0.95, C["BOOT_SEED"] + 1)
            pos = sum(1 for d in d_rise if d > 0)
            sig_r = "유의함" if (lo_r > 0 or hi_r < 0) else "유의하지 않음 (CI가 0 포함)"
            sig_f = "유의함" if (lo_f > 0 or hi_f < 0) else "유의하지 않음 (CI가 0 포함)"
            print(f"\n=== m={m}: NoRec 비단조성 검정 ===")
            print(f"  RISE  NoRec(a={a1}) - NoRec(a={a0}): "
                  f"mean={mean_r:+.4f}%p  95%CI=[{lo_r:+.4f}, {hi_r:+.4f}]  "
                  f"양수 {pos}/{n} 시드  -> {sig_r}")
            print(f"  FALL  NoRec(a={a1}) - NoRec(a={amax}): "
                  f"mean={mean_f:+.4f}%p  95%CI=[{lo_f:+.4f}, {hi_f:+.4f}]  "
                  f"-> {sig_f}")

    with open(C["OUT_SUMMARY"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "Alpha", "N_seeds", "OFF_mean",
                    "Rec_mean", "Rec_std", "NoRec_mean", "NoRec_std"])
        for r in summary_rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.6f}",
                        f"{r[4]:.6f}", f"{r[5]:.6f}",
                        f"{r[6]:.6f}", f"{r[7]:.6f}"])

    with open(C["OUT_PERSEED"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "Alpha", "SeedIdx", "MS_Seed",
                    "DJR_OFF", "DJR_Rec", "DJR_NoRec"])
        for r in per_seed_rows:
            w.writerow([r[0], r[1], r[2], r[3],
                        f"{r[4]:.6f}", f"{r[5]:.6f}", f"{r[6]:.6f}"])

    print(f"\n[완료] 요약  -> {C['OUT_SUMMARY']}")
    print(f"[완료] 시드별 -> {C['OUT_PERSEED']}")


if __name__ == "__main__":
    main()