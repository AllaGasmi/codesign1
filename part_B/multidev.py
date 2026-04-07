"""
LAB 1 – Part B : Multi-device matrix multiplication
=====================================================
Splits an 8192×8192 matrix multiplication across:
  • RTX 3050  – NAIVE  (uncoalesced, intentionally slow)
  • Intel Iris Xe – BEST (K9: prefetch + 2-D register)

The row-split is computed so both GPUs finish simultaneously,
maximising hardware utilisation.

Speedup = GFLOPS(2 GPUs parallel) / GFLOPS(RTX 3050 NAIVE alone)

File layout
-----------
lab1/
  kernels/kernels.cl   ← all OpenCL kernel source code
  part_A/benchmark.py
  part_B/multidev.py   ← this file
"""

import pyopencl as cl
import numpy as np
import time
import os

# ── Configuration ────────────────────────────────────────────
N         = 8192
TILE_SIZE = 16
TSM       = 128
TSN       = 128
TSK       = 16
WPTM      = 8
WPTN      = 8
WIDTH     = 4
WPT       = 4
RTSM      = TSM // WPTM   # 16
RTSN      = TSN // WPTN   # 16

# ── Load & inject defines into kernel source ─────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_KERNEL_SRC = os.path.join(_HERE, "..", "kernels", "kernels.cl")

with open(_KERNEL_SRC) as f:
    _raw = f.read()

DEFINES = f"""
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

# Part B uses partial kernels with a row_offset parameter
PARTIAL_KERNELS = DEFINES + """
// ── NAIVE partial (RTX 3050) ──────────────────────────────
__kernel void matmul_naive_partial(
    __global float* A,
    __global float* B,
    __global float* C,
    int N_full,
    int row_offset)
{
    int row = get_global_id(1) + row_offset;
    int col = get_global_id(0);
    float sum = 0.0f;
    for (int k = 0; k < N_full; k++)
        sum += A[row*N_full + k] * B[k*N_full + col];
    C[row*N_full + col] = sum;
}

// ── BEST partial (Intel Iris Xe) ──────────────────────────
__kernel void matmul_best_partial(
    __global float* A,
    __global float* B,
    __global float* C,
    int N_full,
    int row_offset)
{
    __local float Asub[2][TSK][TSM];
    __local float Bsub[2][TSN][TSK];

    int tidm    = get_local_id(0);
    int tidn    = get_local_id(1);
    int offsetM = TSM * get_group_id(0) + row_offset;
    int offsetN = TSN * get_group_id(1);

    float acc[WPTM][WPTN];
    for (int wm = 0; wm < WPTM; wm++)
        for (int wn = 0; wn < WPTN; wn++)
            acc[wm][wn] = 0.0f;

    float aReg[WPTM];
    float bReg[WPTN];

    int cur = 0, nxt = 1;
    int numTiles = N_full / TSK;

    // Prefetch tile 0
    for (int wm = 0; wm < WPTM; wm++) {
        int row = offsetM + tidm + wm*RTSM;
        Asub[cur][tidn][tidm + wm*RTSM] = A[row*N_full + tidn];
    }
    for (int wn = 0; wn < WPTN; wn++) {
        int col = offsetN + tidn + wn*RTSN;
        Bsub[cur][tidn + wn*RTSN][tidm] = B[tidm*N_full + col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = 1; t < numTiles; t++) {
        for (int wm = 0; wm < WPTM; wm++) {
            int row = offsetM + tidm + wm*RTSM;
            Asub[nxt][tidn][tidm + wm*RTSM] = A[row*N_full + t*TSK + tidn];
        }
        for (int wn = 0; wn < WPTN; wn++) {
            int col = offsetN + tidn + wn*RTSN;
            Bsub[nxt][tidn + wn*RTSN][tidm] = B[(t*TSK + tidm)*N_full + col];
        }
        for (int k = 0; k < TSK; k++) {
            for (int wm = 0; wm < WPTM; wm++)
                aReg[wm] = Asub[cur][k][tidm + wm*RTSM];
            for (int wn = 0; wn < WPTN; wn++)
                bReg[wn] = Bsub[cur][tidn + wn*RTSN][k];
            for (int wm = 0; wm < WPTM; wm++)
                for (int wn = 0; wn < WPTN; wn++)
                    acc[wm][wn] += aReg[wm] * bReg[wn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        cur ^= 1; nxt ^= 1;
    }

    // Last tile
    for (int k = 0; k < TSK; k++) {
        for (int wm = 0; wm < WPTM; wm++)
            aReg[wm] = Asub[cur][k][tidm + wm*RTSM];
        for (int wn = 0; wn < WPTN; wn++)
            bReg[wn] = Bsub[cur][tidn + wn*RTSN][k];
        for (int wm = 0; wm < WPTM; wm++)
            for (int wn = 0; wn < WPTN; wn++)
                acc[wm][wn] += aReg[wm] * bReg[wn];
    }

    for (int wm = 0; wm < WPTM; wm++)
        for (int wn = 0; wn < WPTN; wn++)
            C[(offsetM + tidm + wm*RTSM)*N_full
              + (offsetN + tidn + wn*RTSN)] = acc[wm][wn];
}
"""

# ── Device discovery ─────────────────────────────────────────
nvidia_device = None
intel_gpu     = None

for platform in cl.get_platforms():
    for device in platform.get_devices(cl.device_type.GPU):
        if "NVIDIA" in device.name and nvidia_device is None:
            nvidia_device = (platform, device)
        elif ("Iris" in device.name or "Intel" in device.name) \
                and intel_gpu is None:
            intel_gpu = (platform, device)

assert nvidia_device is not None, "NVIDIA GPU not found"
assert intel_gpu     is not None, "Intel GPU not found"

p0, dev0 = nvidia_device   # RTX 3050  – NAIVE
p1, dev1 = intel_gpu       # Iris Xe   – BEST

print(f"Device 0 (NAIVE) : {dev0.name}")
print(f"Device 1 (BEST)  : {dev1.name}")

# ── Allocate matrices ────────────────────────────────────────
print(f"\nAllocating {N}×{N} matrices …")
A     = np.random.rand(N, N).astype(np.float32)
B     = np.random.rand(N, N).astype(np.float32)
C_out = np.zeros((N, N), dtype=np.float32)

# ── Step 1 : individual benchmarks (full N×N) ────────────────
print("\n" + "="*60)
print("  STEP 1 : Individual benchmarks  (N=8192, full matrix)")
print("="*60)

def bench_full(p, dev, kernel_name, local_s, global_s, label):
    ctx   = cl.Context([dev])
    queue = cl.CommandQueue(ctx,
                properties=cl.command_queue_properties.PROFILING_ENABLE)
    prog  = cl.Program(ctx, PARTIAL_KERNELS).build()
    mf    = cl.mem_flags
    Ab = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=A)
    Bb = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=B)
    Cb = cl.Buffer(ctx, mf.WRITE_ONLY, C_out.nbytes)

    kern = getattr(prog, kernel_name)
    kern(queue, global_s, local_s,
         Ab, Bb, Cb, np.int32(N), np.int32(0)).wait()   # warmup

    times = []
    for _ in range(3):
        ev = kern(queue, global_s, local_s,
                  Ab, Bb, Cb, np.int32(N), np.int32(0))
        ev.wait()
        times.append((ev.profile.end - ev.profile.start) * 1e-9)
    t  = min(times)
    gf = 2 * N**3 / (t * 1e9)
    print(f"  {label:38s}: {t*1000:8.2f} ms  →  {gf:8.2f} GFLOPS")
    return gf

gf_nvidia = bench_full(p0, dev0,
    "matmul_naive_partial",
    (TILE_SIZE, TILE_SIZE), (N, N),
    "RTX 3050   – NAIVE")

gf_intel = bench_full(p1, dev1,
    "matmul_best_partial",
    (RTSM, RTSN), (N * RTSM // TSM, N * RTSN // TSN),
    "Iris Xe    – BEST (prefetch+2Dreg)")

# ── Step 2 : compute optimal row split ───────────────────────
print("\n" + "="*60)
print("  STEP 2 : Optimal row split")
print("="*60)

#   Each device gets a fraction of rows proportional to its speed
#   → both finish at the same wall-clock time
M0_raw = int(round(gf_nvidia / (gf_nvidia + gf_intel) * N))
M0 = max(TSM, (M0_raw // TSM) * TSM)   # align to tile boundary (TSM=128)
if M0 >= N:
    M0 = N - TSM
M1 = N - M0

frac_n = gf_nvidia / (gf_nvidia + gf_intel) * 100
frac_i = gf_intel  / (gf_nvidia + gf_intel) * 100
print(f"  RTX 3050  speed fraction : {frac_n:.1f}%"
      f"  →  rows [0 .. {M0-1}]      ({M0} rows)")
print(f"  Iris Xe   speed fraction : {frac_i:.1f}%"
      f"  →  rows [{M0} .. {N-1}]  ({M1} rows)")

# ── Step 3 : parallel execution ──────────────────────────────
print("\n" + "="*60)
print("  STEP 3 : Parallel execution on 2 GPUs")
print("="*60)

mf = cl.mem_flags

# RTX 3050
ctx0   = cl.Context([dev0])
queue0 = cl.CommandQueue(ctx0,
             properties=cl.command_queue_properties.PROFILING_ENABLE)
prog0  = cl.Program(ctx0, PARTIAL_KERNELS).build()
A_buf0 = cl.Buffer(ctx0, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=A)
B_buf0 = cl.Buffer(ctx0, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=B)
C_buf0 = cl.Buffer(ctx0, mf.WRITE_ONLY, C_out.nbytes)

# Intel Iris Xe
ctx1   = cl.Context([dev1])
queue1 = cl.CommandQueue(ctx1,
             properties=cl.command_queue_properties.PROFILING_ENABLE)
prog1  = cl.Program(ctx1, PARTIAL_KERNELS).build()
A_buf1 = cl.Buffer(ctx1, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=A)
B_buf1 = cl.Buffer(ctx1, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=B)
C_buf1 = cl.Buffer(ctx1, mf.WRITE_ONLY, C_out.nbytes)

# Work sizes
local0  = (TILE_SIZE, TILE_SIZE)
global0 = (N, M0)                             # NVIDIA  : M0 rows

local1  = (RTSM, RTSN)
global1 = (M1 * RTSM // TSM, N * RTSN // TSN)  # Iris Xe : M1 rows

# Warmup
prog0.matmul_naive_partial(queue0, global0, local0,
    A_buf0, B_buf0, C_buf0, np.int32(N), np.int32(0)).wait()
prog1.matmul_best_partial(queue1, global1, local1,
    A_buf1, B_buf1, C_buf1, np.int32(N), np.int32(M0)).wait()

# Timed parallel runs
wall_times = []
for _ in range(3):
    t0  = time.perf_counter()
    ev0 = prog0.matmul_naive_partial(queue0, global0, local0,
              A_buf0, B_buf0, C_buf0, np.int32(N), np.int32(0))
    ev1 = prog1.matmul_best_partial(queue1, global1, local1,
              A_buf1, B_buf1, C_buf1, np.int32(N), np.int32(M0))
    cl.wait_for_events([ev0, ev1])           # both GPUs done
    wall_times.append(time.perf_counter() - t0)

t_parallel = min(wall_times)

# Read back and assemble C
C_tmp0 = np.empty_like(C_out)
C_tmp1 = np.empty_like(C_out)
cl.enqueue_copy(queue0, C_tmp0, C_buf0);  queue0.finish()
cl.enqueue_copy(queue1, C_tmp1, C_buf1);  queue1.finish()
C_out[:M0, :] = C_tmp0[:M0, :]
C_out[M0:, :] = C_tmp1[M0:, :]

# ── Results ───────────────────────────────────────────────────
gf_parallel = 2 * N**3 / (t_parallel * 1e9)
speedup     = gf_parallel / gf_nvidia

print(f"\n  RTX 3050 alone (NAIVE)     : {gf_nvidia:8.2f} GFLOPS  ← reference")
print(f"  Iris Xe  alone (BEST)      : {gf_intel:8.2f} GFLOPS")
print(f"  Parallel wall time         : {t_parallel*1000:8.2f} ms")
print(f"  Parallel throughput        : {gf_parallel:8.2f} GFLOPS")
print(f"\n  Speedup = {gf_parallel:.2f} / {gf_nvidia:.2f} = {speedup:.2f}×")

print("\n" + "="*60)
print(f"  {'GPU':<22} {'Kernel':<22} {'GFLOPS':>10}")
print(f"  {'-'*56}")
print(f"  {'RTX 3050':<22} {'NAIVE':<22} {gf_nvidia:>10.2f}")
print(f"  {'Iris Xe':<22} {'BEST (K9)':<22} {gf_intel:>10.2f}")
print(f"  {'2 GPUs combined':<22} {'parallel':<22} {gf_parallel:>10.2f}")
print(f"  {'Speedup':<44} {speedup:>10.2f}×")
print("="*60)
