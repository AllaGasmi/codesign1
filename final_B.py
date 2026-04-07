import pyopencl as cl
import numpy as np
import time

# ============================================
# CONFIGURATION
# ============================================
N     = 8192
TSM   = 128
TSN   = 128
TSK   = 16
WPTM  = 8
WPTN  = 8
RTSM  = TSM // WPTM   # 16
RTSN  = TSN // WPTN   # 16
TILE_SIZE = 16

# ============================================
# SELECT DEVICES MANUALLY
#   Device 0 : NVIDIA RTX 3050  → NAIVE (uncoalesced)
#   Device 1 : Intel Iris Xe    → BEST kernel (K5 transposed)
# ============================================
nvidia_device = None
intel_gpu     = None

for platform in cl.get_platforms():
    for device in platform.get_devices(cl.device_type.GPU):
        if "NVIDIA" in device.name:
            nvidia_device = (platform, device)
        elif "Iris" in device.name or "Intel" in device.name:
            intel_gpu = (platform, device)

assert nvidia_device is not None, "NVIDIA GPU not found"
assert intel_gpu     is not None, "Intel GPU not found"

p0, dev0 = nvidia_device   # RTX 3050  – NAIVE
p1, dev1 = intel_gpu       # Iris Xe   – BEST

print(f"Device 0 (NAIVE) : {dev0.name}")
print(f"Device 1 (BEST)  : {dev1.name}")

# ============================================
# KERNEL SOURCE
# ============================================
kernel_src = f"""
#define TSM       {TSM}
#define TSN       {TSN}
#define TSK       {TSK}
#define WPTM      {WPTM}
#define WPTN      {WPTN}
#define RTSM      {RTSM}
#define RTSN      {RTSN}
#define TILE_SIZE {TILE_SIZE}

// --- NAIVE (uncoalesced) : RTX 3050 ---
__kernel void matmul_naive(
    __global float* A,
    __global float* B,
    __global float* C,
    int N_full,
    int row_offset)
{{
    int row = get_global_id(1) + row_offset;
    int col = get_global_id(0);
    float sum = 0.0f;
    for(int k = 0; k < N_full; k++)
        sum += A[row*N_full + k] * B[k*N_full + col];
    C[row*N_full + col] = sum;
}}

// --- BEST (K5: transposed + 2D register) : Intel Iris Xe ---
__kernel void matmul_best(
    __global float* A,
    __global float* B,
    __global float* C,
    int N_full,
    int row_offset)
{{
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK];

    int tidm    = get_local_id(0);
    int tidn    = get_local_id(1);
    int offsetM = TSM * get_group_id(0) + row_offset;
    int offsetN = TSN * get_group_id(1);

    float acc[WPTM][WPTN];
    for(int wm = 0; wm < WPTM; wm++)
        for(int wn = 0; wn < WPTN; wn++)
            acc[wm][wn] = 0.0f;

    float aReg[WPTM], bReg[WPTN];
    int numTiles = N_full / TSK;

    for(int t = 0; t < numTiles; t++) {{
        for(int wm = 0; wm < WPTM; wm++) {{
            int row = offsetM + tidm + wm*RTSM;
            Asub[tidn][tidm + wm*RTSM] = A[row*N_full + t*TSK + tidn];
        }}
        for(int wn = 0; wn < WPTN; wn++) {{
            int col = offsetN + tidn + wn*RTSN;
            Bsub[tidn + wn*RTSN][tidm] = B[(t*TSK + tidm)*N_full + col];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < TSK; k++) {{
            for(int wm = 0; wm < WPTM; wm++)
                aReg[wm] = Asub[k][tidm + wm*RTSM];
            for(int wn = 0; wn < WPTN; wn++)
                bReg[wn] = Bsub[tidn + wn*RTSN][k];
            for(int wm = 0; wm < WPTM; wm++)
                for(int wn = 0; wn < WPTN; wn++)
                    acc[wm][wn] += aReg[wm] * bReg[wn];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    for(int wm = 0; wm < WPTM; wm++)
        for(int wn = 0; wn < WPTN; wn++)
            C[(offsetM + tidm + wm*RTSM)*N_full
              + (offsetN + tidn + wn*RTSN)] = acc[wm][wn];
}}
"""

# ============================================
# MATRICES  (N=8192)
# ============================================
print(f"\nAllocating {N}x{N} matrices...")
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C_out = np.zeros((N, N), dtype=np.float32)

# ============================================
# STEP 1 — INDIVIDUAL BENCHMARKS (full matrix)
# ============================================
print("\n" + "="*60)
print("  STEP 1 : Individual benchmarks  (N=8192, full matrix)")
print("="*60)

def bench_full(platform, device, kernel_name, local_s, global_s, label):
    ctx   = cl.Context([device])
    queue = cl.CommandQueue(ctx,
              properties=cl.command_queue_properties.PROFILING_ENABLE)
    prog  = cl.Program(ctx, kernel_src).build()
    mf    = cl.mem_flags
    A_buf = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C_out.nbytes)

    kern = getattr(prog, kernel_name)
    # warmup
    kern(queue, global_s, local_s,
         A_buf, B_buf, C_buf, np.int32(N), np.int32(0)).wait()
    # measure
    times = []
    for _ in range(3):
        ev = kern(queue, global_s, local_s,
                  A_buf, B_buf, C_buf, np.int32(N), np.int32(0))
        ev.wait()
        times.append((ev.profile.end - ev.profile.start) * 1e-9)
    t = min(times)
    gf = 2 * N**3 / (t * 1e9)
    print(f"  {label:35s}: {t*1000:8.2f} ms  →  {gf:8.2f} GFLOPS")
    return gf

gf_nvidia = bench_full(p0, dev0,
    "matmul_naive",
    (TILE_SIZE, TILE_SIZE), (N, N),
    f"RTX 3050  – NAIVE")

gf_intel = bench_full(p1, dev1,
    "matmul_best",
    (RTSM, RTSN), (N * RTSM // TSM, N * RTSN // TSN),
    f"Iris Xe   – BEST (K5)")

# ============================================
# STEP 2 — OPTIMAL SPLIT
# ============================================
print("\n" + "="*60)
print("  STEP 2 : Optimal row split")
print("="*60)

# Rows proportional to speed → both finish at the same time
M0_raw = int(round(gf_nvidia / (gf_nvidia + gf_intel) * N))
M0 = max(TSM, (M0_raw // TSM) * TSM)   # align to TSM=128
if M0 >= N: M0 = N - TSM
M1 = N - M0

print(f"  RTX 3050 fraction : {gf_nvidia/(gf_nvidia+gf_intel)*100:.1f}%"
      f"  →  rows [0 .. {M0-1}]  ({M0} rows)")
print(f"  Iris Xe  fraction : {gf_intel/(gf_nvidia+gf_intel)*100:.1f}%"
      f"  →  rows [{M0} .. {N-1}]  ({M1} rows)")

# ============================================
# STEP 3 — PARALLEL EXECUTION
# ============================================
print("\n" + "="*60)
print("  STEP 3 : Parallel execution on 2 GPUs")
print("="*60)

mf = cl.mem_flags

# --- RTX 3050 context ---
ctx0   = cl.Context([dev0])
queue0 = cl.CommandQueue(ctx0,
           properties=cl.command_queue_properties.PROFILING_ENABLE)
prog0  = cl.Program(ctx0, kernel_src).build()
A_buf0 = cl.Buffer(ctx0, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=A)
B_buf0 = cl.Buffer(ctx0, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=B)
C_buf0 = cl.Buffer(ctx0, mf.WRITE_ONLY, C_out.nbytes)

# --- Intel Iris Xe context ---
ctx1   = cl.Context([dev1])
queue1 = cl.CommandQueue(ctx1,
           properties=cl.command_queue_properties.PROFILING_ENABLE)
prog1  = cl.Program(ctx1, kernel_src).build()
A_buf1 = cl.Buffer(ctx1, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=A)
B_buf1 = cl.Buffer(ctx1, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=B)
C_buf1 = cl.Buffer(ctx1, mf.WRITE_ONLY, C_out.nbytes)

local0  = (TILE_SIZE, TILE_SIZE)
global0 = (N, M0)                                   # NVIDIA: M0 rows

local1  = (RTSM, RTSN)
global1 = (M1 * RTSM // TSM, N * RTSN // TSN)      # Intel:  M1 rows

# warmup
prog0.matmul_naive(queue0, global0, local0,
    A_buf0, B_buf0, C_buf0, np.int32(N), np.int32(0)).wait()
prog1.matmul_best(queue1, global1, local1,
    A_buf1, B_buf1, C_buf1, np.int32(N), np.int32(M0)).wait()

# ---- timed parallel runs ----
wall_times = []
for _ in range(3):
    t0 = time.perf_counter()

    ev0 = prog0.matmul_naive(queue0, global0, local0,
            A_buf0, B_buf0, C_buf0, np.int32(N), np.int32(0))
    ev1 = prog1.matmul_best(queue1, global1, local1,
            A_buf1, B_buf1, C_buf1, np.int32(N), np.int32(M0))

    cl.wait_for_events([ev0, ev1])      # both finish before we stop the clock
    wall_times.append(time.perf_counter() - t0)

t_parallel = min(wall_times)

# read back partial results
C_tmp0 = np.empty_like(C_out)
C_tmp1 = np.empty_like(C_out)
cl.enqueue_copy(queue0, C_tmp0, C_buf0); queue0.finish()
cl.enqueue_copy(queue1, C_tmp1, C_buf1); queue1.finish()
C_out[:M0, :] = C_tmp0[:M0, :]
C_out[M0:, :] = C_tmp1[M0:, :]

# ============================================
# RESULTS
# ============================================
gf_parallel = 2 * N**3 / (t_parallel * 1e9)
speedup     = gf_parallel / gf_nvidia

print(f"\n  RTX 3050 alone  (NAIVE) : {gf_nvidia:8.2f} GFLOPS  (reference)")
print(f"  Iris Xe  alone  (BEST)  : {gf_intel:8.2f} GFLOPS")
print(f"  Parallel wall time      : {t_parallel*1000:8.2f} ms")
print(f"  Parallel throughput     : {gf_parallel:8.2f} GFLOPS")
print(f"\n  Speedup = {gf_parallel:.2f} / {gf_nvidia:.2f} = {speedup:.2f}x")

print("\n" + "="*60)
print(f"  {'GPU':<20} {'Method':<15} {'GFLOPS':>10}")
print(f"  {'-'*47}")
print(f"  {'RTX 3050':<20} {'NAIVE':<15} {gf_nvidia:>10.2f}")
print(f"  {'Iris Xe':<20} {'BEST (K5)':<15} {gf_intel:>10.2f}")
print(f"  {'2 GPUs combined':<20} {'parallel':<15} {gf_parallel:>10.2f}")
print(f"  {'Speedup':<35} {speedup:>10.2f}x")
print("="*60)