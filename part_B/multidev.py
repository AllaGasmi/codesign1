"""
LAB 1 – Part B : Multi-device matrix multiplication
"""

import pyopencl as cl
import numpy as np
import time
import os

N = 1024  
TILE_SIZE = 16
TSM = 128
TSN = 128
TSK = 16
WPTM = 8
WPTN = 8
WIDTH = 4
WPT = 4
RTSM = TSM // WPTM   # 16
RTSN = TSN // WPTN   # 16

_HERE = os.path.dirname(os.path.abspath(__file__))
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
// ── NAIVE partial (GPU 0) ──────────────────────────────
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

// ── BEST partial (GPU 1) - VOTRE K9 DE LA PARTIE A ──────────
__kernel void matmul_best_partial(
    __global float4* A,
    __global float4* B,
    __global float*  C,
    int N_full,
    int row_offset)
{
    int tidm = get_local_id(0);
    int tidn = get_local_id(1);
    int groupm = get_group_id(0);
    int groupn = get_group_id(1);
    
    int offsetM = TSM * groupm + row_offset;
    int offsetN = TSN * groupn;
    int N4 = N_full / 4;
    
    // Sortie immédiate si hors limites
    if (offsetM + tidm >= N_full || offsetN + tidn >= N_full) {
        return;
    }
    
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK];
    
    float acc = 0.0f;
    
    // Boucle sur les tiles K
    for (int tile = 0; tile < N_full / TSK; tile++) {
        int k_start = tile * TSK;
        
        // Charger A dans shared memory
        for (int k = 0; k < TSK; k++) {
            int row = offsetM + tidm;
            int col = k_start + k;
            if (row < N_full && col < N_full) {
                float4 a_vec = A[row * N4 + col/4];
                if (col % 4 == 0) Asub[k][tidm] = a_vec.x;
                else if (col % 4 == 1) Asub[k][tidm] = a_vec.y;
                else if (col % 4 == 2) Asub[k][tidm] = a_vec.z;
                else Asub[k][tidm] = a_vec.w;
            }
        }
        
        // Charger B dans shared memory
        for (int k = 0; k < TSK; k++) {
            int row = k_start + k;
            int col = offsetN + tidn;
            if (row < N_full && col < N_full) {
                float4 b_vec = B[row * N4 + col/4];
                if (col % 4 == 0) Bsub[tidn][k] = b_vec.x;
                else if (col % 4 == 1) Bsub[tidn][k] = b_vec.y;
                else if (col % 4 == 2) Bsub[tidn][k] = b_vec.z;
                else Bsub[tidn][k] = b_vec.w;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Calcul
        for (int k = 0; k < TSK; k++) {
            acc += Asub[k][tidm] * Bsub[tidn][k];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Écriture résultat
    int row = offsetM + tidm;
    int col = offsetN + tidn;
    if (row < N_full && col < N_full) {
        C[row * N_full + col] = acc;
    }
}
"""

# ── Device discovery (version simple) ────────────────────────
print("\n" + "="*60)
print("  DETECTION DES GPUS")
print("="*60)

all_devices = []
for platform in cl.get_platforms():
    for device in platform.get_devices(cl.device_type.GPU):
        all_devices.append((platform, device))
        print(f"  Found: {device.name}")

assert len(all_devices) >= 2, f"Need at least 2 GPUs, found {len(all_devices)}"

p0, dev0 = all_devices[0]  # GPU 0: NAIVE
p1, dev1 = all_devices[1]  # GPU 1: BEST

print(f"\n  GPU 0 (NAIVE kernel) : {dev0.name}")
print(f"  GPU 1 (BEST kernel)  : {dev1.name}")

# ── Allocate matrices ────────────────────────────────────────
print(f"\nAllocating {N}×{N} matrices …")
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C_out = np.zeros((N, N), dtype=np.float32)

# ── Step 1 : individual benchmarks ──────────────────────────
print("\n" + "="*60)
print(f"  STEP 1 : Individual benchmarks (N={N})")
print("="*60)

def bench_full(device, kernel_name, local_s, global_s, label, use_float4=False):
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    prog = cl.Program(ctx, PARTIAL_KERNELS).build()
    
    mf = cl.mem_flags
    
    if use_float4:
        A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C_out.nbytes)
        kern = getattr(prog, kernel_name)
        kern(queue, global_s, local_s, A_buf, B_buf, C_buf, np.int32(N), np.int32(0)).wait()
        
        times = []
        for _ in range(3):
            ev = kern(queue, global_s, local_s, A_buf, B_buf, C_buf, np.int32(N), np.int32(0))
            ev.wait()
            times.append((ev.profile.end - ev.profile.start) * 1e-9)
    else:
        A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C_out.nbytes)
        kern = getattr(prog, kernel_name)
        kern(queue, global_s, local_s, A_buf, B_buf, C_buf, np.int32(N), np.int32(0)).wait()
        
        times = []
        for _ in range(3):
            ev = kern(queue, global_s, local_s, A_buf, B_buf, C_buf, np.int32(N), np.int32(0))
            ev.wait()
            times.append((ev.profile.end - ev.profile.start) * 1e-9)
    
    t = min(times)
    gf = 2 * N**3 / (t * 1e9)
    print(f"  {label:35s}: {t*1000:8.2f} ms  →  {gf:8.2f} GFLOPS")
    return gf

gf_gpu0 = bench_full(dev0, "matmul_naive_partial",
                     (TILE_SIZE, TILE_SIZE), (N, N),
                     f"{dev0.name[:20]} - NAIVE", use_float4=False)

gf_gpu1 = bench_full(dev1, "matmul_best_partial",
                     (RTSM, RTSN), (N * RTSM // TSM, N * RTSN // TSN),
                     f"{dev1.name[:20]} - BEST", use_float4=True)

# ── Step 2 : optimal row split ───────────────────────────────
print("\n" + "="*60)
print("  STEP 2 : Optimal row split")
print("="*60)

total_gf = gf_gpu0 + gf_gpu1
rows0 = int(round(gf_gpu0 / total_gf * N))
rows0 = max(TSM, (rows0 // TSM) * TSM)
if rows0 >= N:
    rows0 = N - TSM
rows1 = N - rows0

print(f"  GPU 0 ({gf_gpu0:.1f} GFLOPS, {gf_gpu0/total_gf*100:.1f}%): {rows0} rows")
print(f"  GPU 1 ({gf_gpu1:.1f} GFLOPS, {gf_gpu1/total_gf*100:.1f}%): {rows1} rows")

# ── Step 3 : parallel execution ──────────────────────────────
print("\n" + "="*60)
print("  STEP 3 : Parallel execution on 2 GPUs")
print("="*60)

# GPU 0 context
ctx0 = cl.Context([dev0])
queue0 = cl.CommandQueue(ctx0)
prog0 = cl.Program(ctx0, PARTIAL_KERNELS).build()
A_buf0 = cl.Buffer(ctx0, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
B_buf0 = cl.Buffer(ctx0, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
C_buf0 = cl.Buffer(ctx0, cl.mem_flags.WRITE_ONLY, C_out.nbytes)

# GPU 1 context
ctx1 = cl.Context([dev1])
queue1 = cl.CommandQueue(ctx1)
prog1 = cl.Program(ctx1, PARTIAL_KERNELS).build()
A_buf1 = cl.Buffer(ctx1, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
B_buf1 = cl.Buffer(ctx1, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
C_buf1 = cl.Buffer(ctx1, cl.mem_flags.WRITE_ONLY, C_out.nbytes)

# Work sizes
local0 = (TILE_SIZE, TILE_SIZE)
global0 = (N, rows0)

local1 = (RTSM, RTSN)
global1 = (rows1 * RTSM // TSM, N * RTSN // TSN)

print("  Running parallel computation...")

# Warmup
prog0.matmul_naive_partial(queue0, global0, local0, A_buf0, B_buf0, C_buf0, np.int32(N), np.int32(0)).wait()
prog1.matmul_best_partial(queue1, global1, local1, A_buf1, B_buf1, C_buf1, np.int32(N), np.int32(rows0)).wait()

# Timed run
start = time.perf_counter()
ev0 = prog0.matmul_naive_partial(queue0, global0, local0, A_buf0, B_buf0, C_buf0, np.int32(N), np.int32(0))
ev1 = prog1.matmul_best_partial(queue1, global1, local1, A_buf1, B_buf1, C_buf1, np.int32(N), np.int32(rows0))
cl.wait_for_events([ev0, ev1])
end = time.perf_counter()

t_parallel = end - start

# Read back results
C_tmp0 = np.empty((rows0, N), dtype=np.float32)
C_tmp1 = np.empty((rows1, N), dtype=np.float32)
cl.enqueue_copy(queue0, C_tmp0, C_buf0).wait()
cl.enqueue_copy(queue1, C_tmp1, C_buf1).wait()

C_out[:rows0, :] = C_tmp0
C_out[rows0:, :] = C_tmp1

# Results
gf_parallel = 2 * N**3 / (t_parallel * 1e9)
speedup = gf_parallel / gf_gpu0

print("\n" + "="*60)
print("  RESULTS")
print("="*60)
print(f"\n  GPU 0 alone (NAIVE)     : {gf_gpu0:8.2f} GFLOPS")
print(f"  GPU 1 alone (BEST)      : {gf_gpu1:8.2f} GFLOPS")
print(f"  Parallel wall time      : {t_parallel*1000:8.2f} ms")
print(f"  Parallel throughput     : {gf_parallel:8.2f} GFLOPS")
print(f"\n  SPEEDUP = {speedup:.2f}x")

print("\n" + "="*60)
print("="*60)


import sys
sys.stdout.flush()