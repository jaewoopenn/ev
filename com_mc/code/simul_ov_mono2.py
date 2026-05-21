#!/usr/bin/env python3
# ===========================================================================
#  CONFIG  ---  여기만 바꾸세요. 그 외에는 그냥 실행 버튼만 누르면 됩니다.
# ===========================================================================
CONFIG = {
    # 원본 파일 경로
    "SIM_PATH":  "simul_unified4.py",
    "GEN_PATH":  "stask_gen.py",

    # 실험 규모
    "M_VALUES":    [2, 4, 8],
    "ALPHAS":      [0.0, 0.01, 0.03, 0.05, 0.10],
    "U_T":         0.70,
    "SWITCH_PROB": 0.20,
    "MAX_SETS":    100,
    "SIM_TICKS":   10000,
    "N_SEEDS":     30,

    # 재현용 고정 시드
    "WL_SEED":      20260518,
    "MS_SEED_BASE": 1000,
    "BOOT_SEED":    12345,
    "N_BOOT":       20000,

    # 출력
    "OUT_SUMMARY": "results_overhead_sig.csv",
    "OUT_PERSEED": "results_overhead_perseed.csv",
    
    # 병렬 처리 설정 (None으로 두면 Mac Mini의 모든 코어 자동 활용)
    "MAX_WORKERS": None,
}
# ===========================================================================
#  이 아래로는 수정할 필요 없습니다.
# ===========================================================================

import os
import sys
import csv
import copy
import random
import statistics
import importlib.util
import multiprocessing
import concurrent.futures

def _load_module(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    src = src.replace('if __name__ == "__main__":\n    main()',
                      'if __name__ == "__main__":\n    pass')
    exec(compile(src, path, "exec"), mod.__dict__)
    sys.modules[modname] = mod
    return mod

def build_prepared(gen, m, U_t, n_sets, wl_seed):
    random.seed(wl_seed)
    prepared = []
    for _ in range(n_sets):
        ts = gen.generate_valid_task_set(m, U_t)
        prepared.append(copy.deepcopy(ts))
        
    csv_filename = f"tasksets_m{m}.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Set_ID", "Task_ID", "Period", "C_LO", "C_HI"])
        for set_id, ts_data in enumerate(prepared):
            for task_id, task in enumerate(ts_data):
                if hasattr(task, 'get'):
                    writer.writerow([set_id, task_id, task.get('period', ''), task.get('c_LO', ''), task.get('c_HI', '')])
                else:
                    writer.writerow([set_id, task_id, getattr(task, 'period', ''), getattr(task, 'c_LO', ''), getattr(task, 'c_HI', '')])
    return prepared


# ===========================================================================
# 워커(Worker) 전용 글로벌 캐시 메모리 공간
# ===========================================================================
worker_sim_cache = None
worker_gen_cache = None
worker_prepared_cache = {}

def worker_eval_task(args):
    """병렬로 할당되는 개별 시뮬레이션 워커"""
    global worker_sim_cache, worker_gen_cache, worker_prepared_cache
    m, alpha, si, ms_seed, wl_seed, u_t, max_sets, sim_path, gen_path, sim_ticks, switch_prob = args
    
    # 1. 모듈 로드 캐싱 (워커당 1번만)
    if worker_sim_cache is None:
        worker_sim_cache = _load_module(sim_path, "simul_unified2")
        worker_gen_cache = _load_module(gen_path, "stask_gen")
        
    sim = worker_sim_cache
    gen = worker_gen_cache
    
    # 2. 태스크셋(Prepared) 캐싱 (직렬화 에러를 막기 위해 워커 내부에서 재생성 후 재사용)
    cache_key = (m, wl_seed)
    if cache_key not in worker_prepared_cache:
        random.seed(wl_seed)
        prep = []
        for _ in range(max_sets):
            prep.append(copy.deepcopy(gen.generate_valid_task_set(m, u_t)))
        worker_prepared_cache[cache_key] = prep
        
    prepared = worker_prepared_cache[cache_key]

    # 3. 메인 시뮬레이션 로직 실행
    acc = {mode: {"total": 0, "degrade": 0} for mode in ["off", "mig_rec", "mig_norec"]}
    for task_set in prepared:
        for mode in ["off", "mig_rec", "mig_norec"]:
            random.seed(ms_seed)
            t_total, d_total = sim.run_simulation(
                task_set, m, sim_ticks,
                mig_mode=mode, switch_prob=switch_prob, mig_alpha=alpha)
            acc[mode]["total"] += t_total
            acc[mode]["degrade"] += d_total
            
    if acc["off"]["total"] == 0:
        return None
        
    return {
        "m": m, "alpha": alpha, "si": si, "ms_seed": ms_seed,
        "off":   100.0 * acc["off"]["degrade"]       / acc["off"]["total"],
        "rec":   100.0 * acc["mig_rec"]["degrade"]   / acc["mig_rec"]["total"],
        "norec": 100.0 * acc["mig_norec"]["degrade"] / acc["mig_norec"]["total"],
    }


def paired_bootstrap_ci(diffs, n_boot, ci, seed):
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
    # macOS 환경에서 안정적인 멀티프로세싱을 위한 필수 설정
    multiprocessing.set_start_method('spawn', force=True)
    
    C = CONFIG
    gen = _load_module(C["GEN_PATH"], "stask_gen")
    _outdir = os.path.dirname(C["OUT_SUMMARY"])
    if _outdir:
        os.makedirs(_outdir, exist_ok=True)

    alphas = C["ALPHAS"]
    per_seed_rows = []
    summary_rows = []
    
    max_workers = C.get("MAX_WORKERS") or multiprocessing.cpu_count()
    print(f"=== 🚀 병렬 시뮬레이션 시작 (활성 코어: {max_workers}개) ===", file=sys.stderr)

    for m in C["M_VALUES"]:
        wl_seed_m = C["WL_SEED"] + m
        
        # 메인 프로세스에서는 CSV 기록 목적으로만 1번 생성
        prepared = build_prepared(gen, m, C["U_T"], C["MAX_SETS"], wl_seed_m)
        if not prepared:
            print(f"[warn] m={m}: 워크로드 생성 실패", file=sys.stderr)
            continue
        print(f"[info] m={m}: {len(prepared)}개 태스크셋 준비 및 CSV 저장 완료", file=sys.stderr)

        norec = {a: [] for a in alphas}
        rec = {a: [] for a in alphas}
        off = {a: [] for a in alphas}

        # 태스크 준비 (m마다 모든 시드와 alpha 조합을 쪼갬)
        tasks = []
        for si in range(C["N_SEEDS"]):
            ms_seed = C["MS_SEED_BASE"] + si
            for a in alphas:
                tasks.append((
                    m, a, si, ms_seed, wl_seed_m, C["U_T"], C["MAX_SETS"],
                    C["SIM_PATH"], C["GEN_PATH"], C["SIM_TICKS"], C["SWITCH_PROB"]
                ))

        print(f"[info] m={m} 시뮬레이션 병렬 처리 중... (총 태스크 {len(tasks)}개)", file=sys.stderr)

        results = []
        # 병렬 실행 (ProcessPoolExecutor)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker_eval_task, t): t for t in tasks}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                try:
                    res = future.result()
                    if res is not None:
                        results.append(res)
                except Exception as e:
                    print(f"[error] 태스크 실행 실패: {e}", file=sys.stderr)
                    
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"       진행률: {completed}/{len(tasks)} 완료", file=sys.stderr)

        # 결과를 정렬하여 기존과 동일한 순서(시드, 오버헤드 순) 보장
        results.sort(key=lambda x: (x["si"], x["alpha"]))

        for r in results:
            a = r["alpha"]
            norec[a].append(r["norec"])
            rec[a].append(r["rec"])
            off[a].append(r["off"])
            per_seed_rows.append([m, a, r["si"], r["ms_seed"], r["off"], r["rec"], r["norec"]])

        # 기존 요약 및 검정 로직 동일
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