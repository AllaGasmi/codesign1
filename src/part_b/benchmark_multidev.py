import gc
import os
import time
import warnings

import numpy as np

# Prevent Windows cp1252 cache encoding errors in PyOpenCL.
os.environ.setdefault("PYOPENCL_NO_CACHE", "1")

import pyopencl as cl

try:
    from pytools.py_codegen import ExistingLineCacheWarning
    warnings.filterwarnings("ignore", category=ExistingLineCacheWarning)
except Exception:
    pass


N = int(os.getenv("LAB_N", "8192"))
TILE_SIZE = 16
TSM = 128
TSN = 128
TSK = 16
WPTM = 8
WPTN = 8
WIDTH = 4
WPT = 4
RTSM = TSM // WPTM
RTSN = TSN // WPTN


def align_rows(rows: int) -> int:
    rows = max(TSM, rows)
    rows = (rows // TSM) * TSM
    return min(max(rows, TSM), N - TSM)


def load_kernel_source() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    kernel_src = os.path.join(here, "..", "kernels", "matmul_kernels.cl")
    with open(kernel_src, "r", encoding="utf-8") as f:
        body = f.read()
    defines = f"""
#define TILE_SIZE {TILE_SIZE}
#define TSM       {TSM}
#define TSN       {TSN}
#define TSK       {TSK}
#define WPTM      {WPTM}
#define WPTN      {WPTN}
#define RTSM      {RTSM}
#define RTSN      {RTSN}
#define WIDTH     {WIDTH}
#define WPT       {WPT}
"""
    return defines + body


def find_devices():
    nvidia = None
    intel = None
    for platform in cl.get_platforms():
        for dev in platform.get_devices(cl.device_type.GPU):
            if "NVIDIA" in dev.name and nvidia is None:
                nvidia = dev
            elif ("Intel" in dev.name or "Iris" in dev.name) and intel is None:
                intel = dev
    if nvidia is None:
        raise RuntimeError("NVIDIA GPU not found")
    if intel is None:
        raise RuntimeError("Intel integrated GPU not found")
    return nvidia, intel


def bench_kernel(queue, kernel, global_s, local_s, args, repeats=3):
    kernel(queue, global_s, local_s, *args).wait()
    times = []
    for _ in range(repeats):
        ev = kernel(queue, global_s, local_s, *args)
        ev.wait()
        times.append((ev.profile.end - ev.profile.start) * 1e-9)
    return min(times)


def build_program(device, src):
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    prog = cl.Program(ctx, src).build()
    return ctx, queue, prog


def run_parallel_candidate(queue0, prog0, b0, a0_chunk, queue1, prog1, b1, a1_chunk, repeats=2):
    m0 = a0_chunk.shape[0]
    m1 = a1_chunk.shape[0]

    mf = cl.mem_flags

    a0 = cl.Buffer(queue0.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a0_chunk)
    c0 = cl.Buffer(queue0.context, mf.WRITE_ONLY, a0_chunk.nbytes)

    a1 = cl.Buffer(queue1.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a1_chunk)
    c1 = cl.Buffer(queue1.context, mf.WRITE_ONLY, a1_chunk.nbytes)

    g0 = (m0, N)
    l0 = (TILE_SIZE, TILE_SIZE)
    args0 = (a0, b0, c0, np.int32(N))

    g1 = (m1 * RTSM // TSM, N * RTSN // TSN)
    l1 = (RTSM, RTSN)
    args1 = (a1, b1, c1, np.int32(N))

    prog0.matmul_naive(queue0, g0, l0, *args0).wait()
    prog1.matmul_best(queue1, g1, l1, *args1).wait()

    wall = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ev0 = prog0.matmul_naive(queue0, g0, l0, *args0)
        ev1 = prog1.matmul_best(queue1, g1, l1, *args1)
        ev0.wait()
        ev1.wait()
        wall.append(time.perf_counter() - t0)

    return min(wall)


def main():
    print("=" * 64)
    print("Part B - Multi-device OpenCL (robust split tuning)")
    print(f"N={N}, naive local=16x16, best local={RTSM}x{RTSN}")
    print("=" * 64)

    dev0, dev1 = find_devices()
    print(f"Device 0 (NVIDIA / NAIVE): {dev0.name}")
    print(f"Device 1 (Intel  / BEST ) : {dev1.name}")

    print(f"\nAllocating host matrices {N}x{N}...")
    a = np.random.rand(N, N).astype(np.float32)
    b = np.random.rand(N, N).astype(np.float32)

    src = load_kernel_source()
    ctx0, queue0, prog0 = build_program(dev0, src)
    ctx1, queue1, prog1 = build_program(dev1, src)

    mf = cl.mem_flags
    b0 = cl.Buffer(ctx0, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    b1 = cl.Buffer(ctx1, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

    # Step 1: full-matrix single-device reference throughput.
    print("\n" + "=" * 64)
    print("STEP 1 - Full matrix benchmarks")
    print("=" * 64)

    a0_full = cl.Buffer(ctx0, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    c0_full = cl.Buffer(ctx0, mf.WRITE_ONLY, a.nbytes)
    t_n = bench_kernel(
        queue0,
        prog0.matmul_naive,
        (N, N),
        (TILE_SIZE, TILE_SIZE),
        (a0_full, b0, c0_full, np.int32(N)),
    )
    gf_nvidia = 2 * N**3 / (t_n * 1e9)
    print(f"RTX 3050 NAIVE      : {t_n*1000:8.2f} ms -> {gf_nvidia:8.2f} GFLOPS")

    a1_full = cl.Buffer(ctx1, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    c1_full = cl.Buffer(ctx1, mf.WRITE_ONLY, a.nbytes)
    t_i = bench_kernel(
        queue1,
        prog1.matmul_best,
        (N * RTSM // TSM, N * RTSN // TSN),
        (RTSM, RTSN),
        (a1_full, b1, c1_full, np.int32(N)),
    )
    gf_intel = 2 * N**3 / (t_i * 1e9)
    print(f"Intel BEST (K9)     : {t_i*1000:8.2f} ms -> {gf_intel:8.2f} GFLOPS")

    # Step 2: tune split.
    print("\n" + "=" * 64)
    print("STEP 2 - Split search")
    print("=" * 64)

    predicted = align_rows(int(round(gf_nvidia / (gf_nvidia + gf_intel) * N)))
    coarse = [512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072]
    local = [predicted + d * TSM for d in (-2, -1, 0, 1, 2)]
    candidates = sorted({align_rows(v) for v in (coarse + local)})

    print(f"Predicted rows on NVIDIA: {predicted}")
    print(f"Candidates tested       : {candidates}")

    best = None
    for m0 in candidates:
        m1 = N - m0
        a0_chunk = np.ascontiguousarray(a[:m0, :])
        a1_chunk = np.ascontiguousarray(a[m0:, :])

        t_wall = run_parallel_candidate(
            queue0,
            prog0,
            b0,
            a0_chunk,
            queue1,
            prog1,
            b1,
            a1_chunk,
            repeats=2,
        )

        gf_parallel = 2 * N**3 / (t_wall * 1e9)
        speedup = gf_parallel / gf_nvidia
        print(
            f"rows_NVIDIA={m0:4d}, rows_Intel={m1:4d} "
            f"-> {t_wall*1000:8.2f} ms, {gf_parallel:8.2f} GFLOPS, {speedup:5.2f}x"
        )

        if best is None or gf_parallel > best[2]:
            best = (m0, t_wall, gf_parallel, speedup)

    best_m0, best_t, best_gf, best_speedup = best

    print("\n" + "=" * 64)
    print("FINAL BEST RESULT")
    print("=" * 64)
    print(f"RTX 3050 only (NAIVE) : {gf_nvidia:8.2f} GFLOPS")
    print(f"Intel only (BEST)     : {gf_intel:8.2f} GFLOPS")
    print(f"Best rows on NVIDIA   : {best_m0}")
    print(f"Best rows on Intel    : {N - best_m0}")
    print(f"Parallel wall time    : {best_t*1000:8.2f} ms")
    print(f"Parallel throughput   : {best_gf:8.2f} GFLOPS")
    print(f"Speedup vs NVIDIA     : {best_speedup:8.2f}x")
    print("=" * 64)

    del a, b
    gc.collect()


if __name__ == "__main__":
    main()
