import csv
import json
import os
from dataclasses import dataclass

import numpy as np
import pyopencl as cl


N = 4096

TSK_VALUES = [8, 16, 24, 32]
TS_VALUES = [64, 96, 128]
WPT_VALUES = [4, 6, 8, 10]

ASYMMETRIC = [
    (128, 64, 16, 8, 4),
    (64, 128, 16, 4, 8),
    (128, 128, 32, 8, 8),
    (96, 96, 16, 6, 6),
    (128, 128, 8, 8, 8),
]


K2_SOURCE = r"""
__kernel void matmul_coalesced(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    float sum = 0.0f;
    for (int k = 0; k < N; k++)
        sum += A[row*N + k] * B[k*N + col];
    C[row*N + col] = sum;
}
"""


def make_k_tune_source(pad_b=False, unroll=False, volatile_id=False):
    b_decl = "__local float Bsub[2][TSN][TSK + 2];" if pad_b else "__local float Bsub[2][TSN][TSK];"

    if volatile_id:
        a_load = (
            "        for (int wm = 0; wm < WPTM; wm++) {\n"
            "            int row = offsetM + tidm + wm*RTSM;\n"
            "            volatile int idm = tidm + wm*RTSM;\n"
            "            Asub[cur][tidn][idm] = A[row*N + tidn];\n"
            "        }"
        )
        a_load_next = (
            "        for (int wm = 0; wm < WPTM; wm++) {\n"
            "            int row = offsetM + tidm + wm*RTSM;\n"
            "            volatile int idm = tidm + wm*RTSM;\n"
            "            Asub[nxt][tidn][idm] = A[row*N + t*TSK + tidn];\n"
            "        }"
        )
        b_load = (
            "        for (int wn = 0; wn < WPTN; wn++) {\n"
            "            int col = offsetN + tidn + wn*RTSN;\n"
            "            volatile int idn = tidn + wn*RTSN;\n"
            "            Bsub[cur][idn][tidm] = B[tidm*N + col];\n"
            "        }"
        )
        b_load_next = (
            "        for (int wn = 0; wn < WPTN; wn++) {\n"
            "            int col = offsetN + tidn + wn*RTSN;\n"
            "            volatile int idn = tidn + wn*RTSN;\n"
            "            Bsub[nxt][idn][tidm] = B[(t*TSK + tidm)*N + col];\n"
            "        }"
        )
    else:
        a_load = (
            "        for (int wm = 0; wm < WPTM; wm++) {\n"
            "            int row = offsetM + tidm + wm*RTSM;\n"
            "            Asub[cur][tidn][tidm + wm*RTSM] = A[row*N + tidn];\n"
            "        }"
        )
        a_load_next = (
            "        for (int wm = 0; wm < WPTM; wm++) {\n"
            "            int row = offsetM + tidm + wm*RTSM;\n"
            "            Asub[nxt][tidn][tidm + wm*RTSM] = A[row*N + t*TSK + tidn];\n"
            "        }"
        )
        b_load = (
            "        for (int wn = 0; wn < WPTN; wn++) {\n"
            "            int col = offsetN + tidn + wn*RTSN;\n"
            "            Bsub[cur][tidn + wn*RTSN][tidm] = B[tidm*N + col];\n"
            "        }"
        )
        b_load_next = (
            "        for (int wn = 0; wn < WPTN; wn++) {\n"
            "            int col = offsetN + tidn + wn*RTSN;\n"
            "            Bsub[nxt][tidn + wn*RTSN][tidm] = B[(t*TSK + tidm)*N + col];\n"
            "        }"
        )

    k_head = "#pragma unroll\n        " if unroll else ""
    wm_head = "#pragma unroll\n            " if unroll else ""
    wn_head = "#pragma unroll\n                " if unroll else ""

    return f"""
__kernel void matmul_k_tune(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{{
    __local float Asub[2][TSK][TSM];
    {b_decl}

    int tidm    = get_local_id(0);
    int tidn    = get_local_id(1);
    int offsetM = TSM * get_group_id(0);
    int offsetN = TSN * get_group_id(1);

    float acc[WPTM][WPTN];
    for (int wm = 0; wm < WPTM; wm++)
        for (int wn = 0; wn < WPTN; wn++)
            acc[wm][wn] = 0.0f;

    float aReg[WPTM];
    float bReg[WPTN];

    int cur = 0, nxt = 1;
    int numTiles = N / TSK;

    // Prefetch tile 0
{a_load}
{b_load}
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = 1; t < numTiles; t++) {{

        // Load next tile while computing current
{a_load_next}
{b_load_next}

        // Compute tile t-1
        {k_head}for (int k = 0; k < TSK; k++) {{
            {wm_head}for (int wm = 0; wm < WPTM; wm++)
                aReg[wm] = Asub[cur][k][tidm + wm*RTSM];
            {wn_head}for (int wn = 0; wn < WPTN; wn++)
                bReg[wn] = Bsub[cur][tidn + wn*RTSN][k];
            {wm_head}for (int wm = 0; wm < WPTM; wm++)
                {wn_head}for (int wn = 0; wn < WPTN; wn++)
                    acc[wm][wn] += aReg[wm] * bReg[wn];
        }}

        barrier(CLK_LOCAL_MEM_FENCE);
        cur ^= 1; nxt ^= 1;
    }}

    // Compute last tile
    {k_head}for (int k = 0; k < TSK; k++) {{
        {wm_head}for (int wm = 0; wm < WPTM; wm++)
            aReg[wm] = Asub[cur][k][tidm + wm*RTSM];
        {wn_head}for (int wn = 0; wn < WPTN; wn++)
            bReg[wn] = Bsub[cur][tidn + wn*RTSN][k];
        {wm_head}for (int wm = 0; wm < WPTM; wm++)
            {wn_head}for (int wn = 0; wn < WPTN; wn++)
                acc[wm][wn] += aReg[wm] * bReg[wn];
    }}

    for (int wm = 0; wm < WPTM; wm++)
        for (int wn = 0; wn < WPTN; wn++)
            C[(offsetM + tidm + wm*RTSM)*N
              + (offsetN + tidn + wn*RTSN)] = acc[wm][wn];
}}
"""


@dataclass(frozen=True)
class Cfg:
    tsm: int
    tsn: int
    tsk: int
    wptm: int
    wptn: int

    @property
    def rtsm(self):
        return self.tsm // self.wptm

    @property
    def rtsn(self):
        return self.tsn // self.wptn


@dataclass
class RunResult:
    cfg: Cfg
    label: str
    source_variant: str
    fast_math: bool
    gflops: float
    mean_ms: float
    local_bytes: int
    threads: int
    max_abs_diff: float
    exact_correct: bool


def build_options(cfg: Cfg, fast_math=False):
    opts = [
        f"-DTSM={cfg.tsm}",
        f"-DTSN={cfg.tsn}",
        f"-DTSK={cfg.tsk}",
        f"-DWPTM={cfg.wptm}",
        f"-DWPTN={cfg.wptn}",
        f"-DRTSM={cfg.rtsm}",
        f"-DRTSN={cfg.rtsn}",
    ]
    if fast_math:
        opts.append("-cl-fast-relaxed-math")
    return opts


def local_mem_bytes(cfg: Cfg):
    return 2 * (cfg.tsk * cfg.tsm + cfg.tsn * cfg.tsk) * 4


def validate_cfg(cfg: Cfg):
    reasons = []
    if cfg.tsm % cfg.wptm != 0:
        reasons.append("TSM%WPTM!=0")
    if cfg.tsn % cfg.wptn != 0:
        reasons.append("TSN%WPTN!=0")
    if reasons:
        return False, reasons

    threads = cfg.rtsm * cfg.rtsn
    if threads < 64 or threads > 512:
        reasons.append("threads outside [64,512]")

    if local_mem_bytes(cfg) > 45000:
        reasons.append("local memory > 45000")

    if N % cfg.tsk != 0:
        reasons.append("N%TSK!=0")
    if N % cfg.wptm != 0:
        reasons.append("N%WPTM!=0")
    if N % cfg.wptn != 0:
        reasons.append("N%WPTN!=0")

    den = cfg.rtsm * cfg.rtsn
    lpta_num = cfg.tsk * cfg.tsm
    lptb_num = cfg.tsk * cfg.tsn
    if lpta_num % den != 0:
        reasons.append("LPTA not integer")
    if lptb_num % den != 0:
        reasons.append("LPTB not integer")

    return len(reasons) == 0, reasons


def bench_kernel(queue, kernel, global_s, local_s, args, warmup=3, timed=5):
    for _ in range(warmup):
        kernel(queue, global_s, local_s, *args).wait()

    times = []
    for _ in range(timed):
        ev = kernel(queue, global_s, local_s, *args)
        ev.wait()
        times.append((ev.profile.end - ev.profile.start) * 1e-9)

    mean_t = float(np.mean(times))
    gf = (2.0 * (N ** 3)) / (mean_t * 1e9)
    return gf, mean_t * 1e3


def run_cfg(ctx, queue, a_buf, b_buf, c_buf, cfg: Cfg, source, variant_name, c_ref, fast_math=False):
    options = build_options(cfg, fast_math=fast_math)
    program = cl.Program(ctx, source).build(options=options)
    kernel = program.matmul_k_tune

    global_s = (N * cfg.rtsm // cfg.tsm, N * cfg.rtsn // cfg.tsn)
    local_s = (cfg.rtsm, cfg.rtsn)
    args = (a_buf, b_buf, c_buf, np.int32(N))

    gf, ms = bench_kernel(queue, kernel, global_s, local_s, args)

    kernel(queue, global_s, local_s, *args).wait()
    c_out = np.empty_like(c_ref)
    cl.enqueue_copy(queue, c_out, c_buf).wait()
    max_abs = float(np.max(np.abs(c_out - c_ref)))
    exact_ok = (max_abs == 0.0)

    return RunResult(
        cfg=cfg,
        label=f"TSM={cfg.tsm},TSN={cfg.tsn},TSK={cfg.tsk},WPTM={cfg.wptm},WPTN={cfg.wptn}",
        source_variant=variant_name,
        fast_math=fast_math,
        gflops=gf,
        mean_ms=ms,
        local_bytes=local_mem_bytes(cfg),
        threads=cfg.rtsm * cfg.rtsn,
        max_abs_diff=max_abs,
        exact_correct=exact_ok,
    )


def fmt_result(r: RunResult):
    fm = "yes" if r.fast_math else "no"
    ok = "yes" if r.exact_correct else "no"
    return (
        f"{r.label:48s} | var={r.source_variant:14s} | fast={fm:3s} | "
        f"WG={r.cfg.rtsm:2d}x{r.cfg.rtsn:<2d} ({r.threads:3d}) | "
        f"LMEM={r.local_bytes:5d} B | {r.mean_ms:8.2f} ms | {r.gflops:8.2f} GFLOPS | "
        f"max_abs={r.max_abs_diff:.3f} | exact={ok}"
    )


def main():
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    print(f"Device: {device.name}")

    a = np.random.rand(N, N).astype(np.float32)
    b = np.random.rand(N, N).astype(np.float32)
    c = np.zeros((N, N), dtype=np.float32)

    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

    # K2 reference (3 warmup + 5 timed)
    ref_prog = cl.Program(ctx, K2_SOURCE).build()
    ref_kernel = ref_prog.matmul_coalesced
    k2_gf, k2_ms = bench_kernel(
        queue,
        ref_kernel,
        (N, N),
        (16, 16),
        (a_buf, b_buf, c_buf, np.int32(N)),
    )
    c_ref = np.empty_like(c)
    ref_kernel(queue, (N, N), (16, 16), a_buf, b_buf, c_buf, np.int32(N)).wait()
    cl.enqueue_copy(queue, c_ref, c_buf).wait()
    print(f"K2 reference: {k2_ms:.2f} ms, {k2_gf:.2f} GFLOPS")

    # Build candidate set
    configs = []
    for tsk in TSK_VALUES:
        for ts in TS_VALUES:
            for wpt in WPT_VALUES:
                configs.append(Cfg(ts, ts, tsk, wpt, wpt))

    for tsm, tsn, tsk, wptm, wptn in ASYMMETRIC:
        configs.append(Cfg(tsm, tsn, tsk, wptm, wptn))

    unique = {}
    for cobj in configs:
        unique[(cobj.tsm, cobj.tsn, cobj.tsk, cobj.wptm, cobj.wptn)] = cobj
    configs = list(unique.values())

    valid = []
    invalid = []
    for cfg in configs:
        ok, reasons = validate_cfg(cfg)
        if ok:
            valid.append(cfg)
        else:
            invalid.append((cfg, reasons))

    print(f"Total candidate configs: {len(configs)}")
    print(f"Valid configs: {len(valid)}")
    print(f"Invalid configs: {len(invalid)}")

    base_source = make_k_tune_source(pad_b=False, unroll=False, volatile_id=False)

    # Step B: sweep all valid configs
    step_b = []
    print("\nSTEP B - valid config sweep")
    for cfg in valid:
        r = run_cfg(ctx, queue, a_buf, b_buf, c_buf, cfg, base_source, "base", c_ref, fast_math=False)
        step_b.append(r)
        print(fmt_result(r))

    step_b_sorted = sorted(step_b, key=lambda x: x.gflops, reverse=True)
    step_b_correct = [r for r in step_b_sorted if r.exact_correct]

    # Step C: top 3 + independent tweaks
    top3 = step_b_correct[:3]
    if len(top3) < 3:
        print("\nWARNING: fewer than 3 exact-correct configs found in Step B.")
    print("\nTop 3 from Step B")
    for r in top3:
        print(fmt_result(r))

    tweak_results = []
    print("\nSTEP C - independent tweaks on top 3")
    for base in top3:
        cfg = base.cfg

        r_fast = run_cfg(
            ctx, queue, a_buf, b_buf, c_buf, cfg,
            make_k_tune_source(pad_b=False, unroll=False, volatile_id=False),
            "fast_math",
            c_ref,
            fast_math=True,
        )
        tweak_results.append(r_fast)
        print(fmt_result(r_fast))

        r_pad = run_cfg(
            ctx, queue, a_buf, b_buf, c_buf, cfg,
            make_k_tune_source(pad_b=True, unroll=False, volatile_id=False),
            "pad_b_tsk2",
            c_ref,
            fast_math=False,
        )
        tweak_results.append(r_pad)
        print(fmt_result(r_pad))

        r_unroll = run_cfg(
            ctx, queue, a_buf, b_buf, c_buf, cfg,
            make_k_tune_source(pad_b=False, unroll=True, volatile_id=False),
            "pragma_unroll",
            c_ref,
            fast_math=False,
        )
        tweak_results.append(r_unroll)
        print(fmt_result(r_unroll))

    # Best from B + C
    pool_bc = [r for r in (step_b + tweak_results) if r.exact_correct]
    best_bc = max(pool_bc, key=lambda x: x.gflops)

    # Step D: volatile test on best B+C config
    print("\nSTEP D - volatile id test on best B+C config")
    r_vol = run_cfg(
        ctx,
        queue,
        a_buf,
        b_buf,
        c_buf,
        best_bc.cfg,
        make_k_tune_source(
            pad_b=(best_bc.source_variant == "pad_b_tsk2"),
            unroll=(best_bc.source_variant == "pragma_unroll"),
            volatile_id=True,
        ),
        "volatile_id",
        c_ref,
        fast_math=best_bc.fast_math,
    )
    print(fmt_result(r_vol))

    # Overall best among B + C + D
    all_tuned = pool_bc + ([r_vol] if r_vol.exact_correct else [])
    best_all = max(all_tuned, key=lambda x: x.gflops)

    print("\nSummary")
    print(f"Best Step B: {fmt_result(step_b_sorted[0])}")
    print(f"Best B+C : {fmt_result(best_bc)}")
    print(f"Best B+C+D: {fmt_result(best_all)}")
    print(f"Gain vs K2 (best B+C+D): {best_all.gflops / k2_gf:.2f}x")

    # Save full tables
    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "k9_autotune_results.csv")
    json_path = os.path.join(out_dir, "k9_autotune_results.json")

    def to_row(r: RunResult, phase):
        return {
            "phase": phase,
            "tsm": r.cfg.tsm,
            "tsn": r.cfg.tsn,
            "tsk": r.cfg.tsk,
            "wptm": r.cfg.wptm,
            "wptn": r.cfg.wptn,
            "rtsm": r.cfg.rtsm,
            "rtsn": r.cfg.rtsn,
            "threads": r.threads,
            "local_bytes": r.local_bytes,
            "source_variant": r.source_variant,
            "fast_math": r.fast_math,
            "mean_ms": r.mean_ms,
            "gflops": r.gflops,
            "gain_vs_k2": r.gflops / k2_gf,
            "max_abs_diff": r.max_abs_diff,
            "exact_correct": r.exact_correct,
            "label": r.label,
        }

    rows = []
    rows.extend([to_row(r, "step_b") for r in sorted(step_b, key=lambda x: x.gflops, reverse=True)])
    rows.extend([to_row(r, "step_c") for r in sorted(tweak_results, key=lambda x: x.gflops, reverse=True)])
    rows.append(to_row(r_vol, "step_d"))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "k2": {"mean_ms": k2_ms, "gflops": k2_gf},
        "valid_count": len(valid),
        "invalid": [
            {
                "cfg": {
                    "tsm": c.tsm,
                    "tsn": c.tsn,
                    "tsk": c.tsk,
                    "wptm": c.wptm,
                    "wptn": c.wptn,
                },
                "reasons": reasons,
            }
            for c, reasons in invalid
        ],
        "step_b_sorted": [to_row(r, "step_b") for r in sorted(step_b, key=lambda x: x.gflops, reverse=True)],
        "step_c_sorted": [to_row(r, "step_c") for r in sorted(tweak_results, key=lambda x: x.gflops, reverse=True)],
        "step_d": to_row(r_vol, "step_d"),
        "best_step_b": to_row(step_b_sorted[0], "step_b"),
        "best_bc": to_row(best_bc, "step_c"),
        "best_all": to_row(best_all, "step_d"),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
