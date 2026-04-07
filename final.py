import pyopencl as cl
import numpy as np
import time

# ============================================
# CONFIGURATION
# ============================================
N         = 4096
TILE_SIZE = 16   # TSK (depth tile)
TSM       = 128  # Tile rows
TSN       = 128  # Tile cols
TSK       = 16   # Tile depth
WPTM      = 8    # Work per thread (rows)
WPTN      = 8    # Work per thread (cols)
WIDTH     = 4    # float4
WPT       = 4    # for register kernel

RTSM = TSM // WPTM   # = 16
RTSN = TSN // WPTN   # = 16

# ============================================
# MATRICES
# ============================================
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# ============================================
# OPENCL SETUP
# ============================================
platform = cl.get_platforms()[0]
device   = platform.get_devices()[0]
ctx      = cl.Context([device])
queue    = cl.CommandQueue(ctx,
           properties=cl.command_queue_properties.PROFILING_ENABLE)

print(f"Device : {device.name}")
print(f"Max local memory : {device.local_mem_size} bytes")
print(f"Max work group size : {device.max_work_group_size}")

# ============================================
# KERNELS
# ============================================
kernel_code = f"""
#define TILE_SIZE {TILE_SIZE}
#define TSM  {TSM}
#define TSN  {TSN}
#define TSK  {TSK}
#define WPTM {WPTM}
#define WPTN {WPTN}
#define RTSM {RTSM}
#define RTSN {RTSN}
#define WIDTH {WIDTH}
#define WPT  {WPT}

// ============================================
// KERNEL 1 : NAIVE (baseline)
// ============================================
__kernel void matmul_naive(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{{
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;
    for(int k = 0; k < N; k++)
        sum += A[row*N + k] * B[k*N + col];
    C[row*N + col] = sum;
}}

// ============================================
// KERNEL 2 : COALESCED (reference)
// ============================================
__kernel void matmul_coalesced(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{{
    int col = get_global_id(0);
    int row = get_global_id(1);
    float sum = 0.0f;
    for(int k = 0; k < N; k++)
        sum += A[row*N + k] * B[k*N + col];
    C[row*N + col] = sum;
}}

// ============================================
// KERNEL 3 : TILING (local memory)
// ============================================
__kernel void matmul_tiling(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{{
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    int row      = get_global_id(1);
    int col      = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    float sum = 0.0f;
    int numTiles = N / TILE_SIZE;
    for(int t = 0; t < numTiles; t++) {{
        Asub[localRow][localCol] = A[row*N + t*TILE_SIZE + localCol];
        Bsub[localRow][localCol] = B[(t*TILE_SIZE + localRow)*N + col];
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int k = 0; k < TILE_SIZE; k++)
            sum += Asub[localRow][k] * Bsub[k][localCol];
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    C[row*N + col] = sum;
}}

// ============================================
// KERNEL 4 : REGISTER TILING 1D (WPT)
// ============================================
__kernel void matmul_register(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{{
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    int row      = get_global_id(1);
    int col      = get_global_id(0) * WPT;
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    float results[WPT];
    for(int w = 0; w < WPT; w++) results[w] = 0.0f;

    int numTiles = N / TILE_SIZE;
    for(int t = 0; t < numTiles; t++) {{
        Asub[localRow][localCol] = A[row*N + t*TILE_SIZE + localCol];
        Bsub[localRow][localCol] = B[(t*TILE_SIZE + localRow)*N + col];
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int k = 0; k < TILE_SIZE; k++) {{
            float a = Asub[localRow][k];
            for(int w = 0; w < WPT; w++)
                results[w] += a * Bsub[k][localCol + w];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    for(int w = 0; w < WPT; w++)
        C[row*N + col + w] = results[w];
}}

// ============================================
// KERNEL 5 : TRANSPOSED + RECTANGULAR TILES
// ============================================
__kernel void matmul_transposed(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{{
    __local float Asub[TSK][TSM];   // A stored transposed → no bank conflicts
    __local float Bsub[TSN][TSK];

    int tidm    = get_local_id(0);
    int tidn    = get_local_id(1);
    int offsetM = TSM * get_group_id(0);
    int offsetN = TSN * get_group_id(1);

    float acc[WPTM][WPTN];
    for(int wm = 0; wm < WPTM; wm++)
        for(int wn = 0; wn < WPTN; wn++)
            acc[wm][wn] = 0.0f;

    int numTiles = N / TSK;
    for(int t = 0; t < numTiles; t++) {{

        // Load A transposed into local memory
        for(int wm = 0; wm < WPTM; wm++) {{
            int row = offsetM + tidm + wm*RTSM;
            int col = t*TSK + tidn;
            Asub[tidn][tidm + wm*RTSM] = A[row*N + col];
        }}
        // Load B into local memory
        for(int wn = 0; wn < WPTN; wn++) {{
            int row = t*TSK + tidm;
            int col = offsetN + tidn + wn*RTSN;
            Bsub[tidn + wn*RTSN][tidm] = B[row*N + col];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < TSK; k++) {{
            for(int wm = 0; wm < WPTM; wm++) {{
                for(int wn = 0; wn < WPTN; wn++) {{
                    acc[wm][wn] += Asub[k][tidm + wm*RTSM]
                                 * Bsub[tidn + wn*RTSN][k];
                }}
            }}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    for(int wm = 0; wm < WPTM; wm++)
        for(int wn = 0; wn < WPTN; wn++)
            C[(offsetM + tidm + wm*RTSM)*N
              + (offsetN + tidn + wn*RTSN)] = acc[wm][wn];
}}

// ============================================
// KERNEL 6 : 2D REGISTER BLOCKING
// ============================================
__kernel void matmul_2d_register(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{{
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK];

    int tidm    = get_local_id(0);
    int tidn    = get_local_id(1);
    int offsetM = TSM * get_group_id(0);
    int offsetN = TSN * get_group_id(1);

    float acc[WPTM][WPTN];
    for(int wm = 0; wm < WPTM; wm++)
        for(int wn = 0; wn < WPTN; wn++)
            acc[wm][wn] = 0.0f;

    // Cache registers for A and B columns
    float aReg[WPTM];
    float bReg[WPTN];

    int numTiles = N / TSK;
    for(int t = 0; t < numTiles; t++) {{

        for(int wm = 0; wm < WPTM; wm++) {{
            int row = offsetM + tidm + wm*RTSM;
            Asub[tidn][tidm + wm*RTSM] = A[row*N + t*TSK + tidn];
        }}
        for(int wn = 0; wn < WPTN; wn++) {{
            int col = offsetN + tidn + wn*RTSN;
            Bsub[tidn + wn*RTSN][tidm] = B[(t*TSK + tidm)*N + col];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);

        // 2D register blocking: cache into registers before inner loop
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
            C[(offsetM + tidm + wm*RTSM)*N
              + (offsetN + tidn + wn*RTSN)] = acc[wm][wn];
}}

// ============================================
// KERNEL 7 : FLOAT4 + 2D REGISTER BLOCKING
// ============================================
__kernel void matmul_float4_2d(
    __global float4* A,
    __global float4* B,
    __global float4* C,
    int N)
{{
    __local float4 Asub[TSK][TSM/WIDTH];
    __local float4 Bsub[TSN][TSK/WIDTH];

    int tidm    = get_local_id(0);
    int tidn    = get_local_id(1);
    int offsetM = TSM * get_group_id(0);
    int offsetN = TSN * get_group_id(1);

    float4 acc[WPTM][WPTN/WIDTH];
    for(int wm = 0; wm < WPTM; wm++)
        for(int wn = 0; wn < WPTN/WIDTH; wn++)
            acc[wm][wn] = (float4)(0.0f);

    int numTiles = N / TSK;
    for(int t = 0; t < numTiles; t++) {{

        // Vectorised loads
        for(int wm = 0; wm < WPTM; wm++) {{
            int row = offsetM + tidm + wm*RTSM;
            int col = t*(TSK/WIDTH) + tidn;
            Asub[tidn][tidm + wm*RTSM] = A[row*(N/WIDTH) + col];  // NOTE: reinterpret as float4 row
        }}
        for(int wn = 0; wn < WPTN/WIDTH; wn++) {{
            int row = t*TSK + tidm;
            int col = (offsetN/WIDTH) + tidn + wn*RTSN;
            Bsub[tidn + wn*RTSN][tidm] = B[row*(N/WIDTH) + col];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute with float4 MAD
        for(int k = 0; k < TSK; k++) {{
            float4 aVec[WPTM];
            for(int wm = 0; wm < WPTM; wm++)
                aVec[wm] = Asub[k][tidm + wm*RTSM];

            for(int wn = 0; wn < WPTN/WIDTH; wn++) {{
                float4 b = Bsub[tidn + wn*RTSN][k];
                for(int wm = 0; wm < WPTM; wm++) {{
                    // dot-product-style accumulation
                    acc[wm][wn] += aVec[wm].x * b
                                 + aVec[wm].y * b
                                 + aVec[wm].z * b
                                 + aVec[wm].w * b;
                }}
            }}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    for(int wm = 0; wm < WPTM; wm++)
        for(int wn = 0; wn < WPTN/WIDTH; wn++) {{
            int row = offsetM + tidm + wm*RTSM;
            int col = (offsetN/WIDTH) + tidn + wn*RTSN;
            C[row*(N/WIDTH) + col] = acc[wm][wn];
        }}
}}

// ============================================
// KERNEL 8 : PREFETCH + 2D REGISTER (BEST)
// ============================================
__kernel void matmul_best(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{{
    __local float Asub[2][TSK][TSM];   // double buffer
    __local float Bsub[2][TSN][TSK];

    int tidm    = get_local_id(0);
    int tidn    = get_local_id(1);
    int offsetM = TSM * get_group_id(0);
    int offsetN = TSN * get_group_id(1);

    float acc[WPTM][WPTN];
    for(int wm = 0; wm < WPTM; wm++)
        for(int wn = 0; wn < WPTN; wn++)
            acc[wm][wn] = 0.0f;

    float aReg[WPTM];
    float bReg[WPTN];

    int cur = 0, nxt = 1;
    int numTiles = N / TSK;

    // Prefetch tile 0
    for(int wm = 0; wm < WPTM; wm++) {{
        int row = offsetM + tidm + wm*RTSM;
        Asub[cur][tidn][tidm + wm*RTSM] = A[row*N + tidn];
    }}
    for(int wn = 0; wn < WPTN; wn++) {{
        int col = offsetN + tidn + wn*RTSN;
        Bsub[cur][tidn + wn*RTSN][tidm] = B[tidm*N + col];
    }}
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int t = 1; t < numTiles; t++) {{

        // Prefetch tile t into next buffer
        for(int wm = 0; wm < WPTM; wm++) {{
            int row = offsetM + tidm + wm*RTSM;
            Asub[nxt][tidn][tidm + wm*RTSM] = A[row*N + t*TSK + tidn];
        }}
        for(int wn = 0; wn < WPTN; wn++) {{
            int col = offsetN + tidn + wn*RTSN;
            Bsub[nxt][tidn + wn*RTSN][tidm] = B[(t*TSK + tidm)*N + col];
        }}

        // Compute tile t-1 from current buffer
        for(int k = 0; k < TSK; k++) {{
            for(int wm = 0; wm < WPTM; wm++)
                aReg[wm] = Asub[cur][k][tidm + wm*RTSM];
            for(int wn = 0; wn < WPTN; wn++)
                bReg[wn] = Bsub[cur][tidn + wn*RTSN][k];
            for(int wm = 0; wm < WPTM; wm++)
                for(int wn = 0; wn < WPTN; wn++)
                    acc[wm][wn] += aReg[wm] * bReg[wn];
        }}

        barrier(CLK_LOCAL_MEM_FENCE);
        cur ^= 1; nxt ^= 1;
    }}

    // Compute last tile
    for(int k = 0; k < TSK; k++) {{
        for(int wm = 0; wm < WPTM; wm++)
            aReg[wm] = Asub[cur][k][tidm + wm*RTSM];
        for(int wn = 0; wn < WPTN; wn++)
            bReg[wn] = Bsub[cur][tidn + wn*RTSN][k];
        for(int wm = 0; wm < WPTM; wm++)
            for(int wn = 0; wn < WPTN; wn++)
                acc[wm][wn] += aReg[wm] * bReg[wn];
    }}

    // Write results
    for(int wm = 0; wm < WPTM; wm++)
        for(int wn = 0; wn < WPTN; wn++)
            C[(offsetM + tidm + wm*RTSM)*N
              + (offsetN + tidn + wn*RTSN)] = acc[wm][wn];
}}

// ============================================
// KERNEL 9 : ARBITRARY SIZE (padding support)
// ============================================
__kernel void matmul_arbitrary(
    __global float* A,
    __global float* B,
    __global float* C,
    int M, int K, int Nsize)
{{
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK];

    int tidm    = get_local_id(0);
    int tidn    = get_local_id(1);
    int offsetM = TSM * get_group_id(0);
    int offsetN = TSN * get_group_id(1);

    float acc[WPTM][WPTN];
    for(int wm = 0; wm < WPTM; wm++)
        for(int wn = 0; wn < WPTN; wn++)
            acc[wm][wn] = 0.0f;

    float aReg[WPTM];
    float bReg[WPTN];

    int numTiles = (K + TSK - 1) / TSK;
    for(int t = 0; t < numTiles; t++) {{

        // Bounds-checked load of A
        for(int wm = 0; wm < WPTM; wm++) {{
            int row = offsetM + tidm + wm*RTSM;
            int col = t*TSK + tidn;
            Asub[tidn][tidm + wm*RTSM] =
                (row < M && col < K) ? A[row*K + col] : 0.0f;
        }}
        // Bounds-checked load of B
        for(int wn = 0; wn < WPTN; wn++) {{
            int row = t*TSK + tidm;
            int col = offsetN + tidn + wn*RTSN;
            Bsub[tidn + wn*RTSN][tidm] =
                (row < K && col < Nsize) ? B[row*Nsize + col] : 0.0f;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);

        int tileK = min(TSK, K - t*TSK);
        for(int k = 0; k < tileK; k++) {{
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

    // Bounds-checked write
    for(int wm = 0; wm < WPTM; wm++)
        for(int wn = 0; wn < WPTN; wn++) {{
            int row = offsetM + tidm + wm*RTSM;
            int col = offsetN + tidn + wn*RTSN;
            if(row < M && col < Nsize)
                C[row*Nsize + col] = acc[wm][wn];
        }}
}}
"""

# ============================================
# COMPILE
# ============================================
program = cl.Program(ctx, kernel_code).build()

# ============================================
# BUFFERS
# ============================================
mf    = cl.mem_flags
A_buf = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=A)
B_buf = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=B)
C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

A4 = A.view(np.float32)
B4 = B.view(np.float32)
A4_buf = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=A4)
B4_buf = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=B4)
C4_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

# ============================================
# WORK SIZES
# ============================================
local_naive  = (TILE_SIZE, TILE_SIZE)
global_naive = (N, N)

local_reg    = (TILE_SIZE, TILE_SIZE)
global_reg   = (N // WPT, N)

local_rect   = (RTSM, RTSN)
global_rect  = (N * RTSM // TSM, N * RTSN // TSN)

local_f4     = (RTSM, RTSN)
global_f4    = (N * RTSM // TSM, N * RTSN // TSN)

# ============================================
# BENCHMARK FUNCTION
# ============================================
def benchmark(kernel_func, global_s, local_s, args, label, repeats=3):
    # warmup
    ev = kernel_func(queue, global_s, local_s, *args)
    ev.wait()
    # measure
    times = []
    for _ in range(repeats):
        ev = kernel_func(queue, global_s, local_s, *args)
        ev.wait()
        t = (ev.profile.end - ev.profile.start) * 1e-9
        times.append(t)
    t_best = min(times)
    gflops = (2 * N**3) / (t_best * 1e9)
    print(f"{label:40s} : {t_best*1000:8.2f} ms  →  {gflops:7.2f} GFLOPS")
    return gflops

# ============================================
# RUN ALL BENCHMARKS
# ============================================
print("\n" + "=" * 70)
print(f"  Matrix: {N}x{N}   TSM={TSM} TSN={TSN} TSK={TSK}   WPTM={WPTM} WPTN={WPTN}")
print("=" * 70)

args_std  = (A_buf, B_buf, C_buf, np.int32(N))
args_rect = (A_buf, B_buf, C_buf, np.int32(N))
args_f4   = (A4_buf, B4_buf, C4_buf, np.int32(N))

g_naive  = benchmark(program.matmul_naive,
                     global_naive, local_naive, args_std,
                     "K1 NAIVE")

g_coal   = benchmark(program.matmul_coalesced,
                     global_naive, local_naive, args_std,
                     "K2 COALESCED")

g_tile   = benchmark(program.matmul_tiling,
                     global_naive, local_naive, args_std,
                     "K3 TILING (local mem)")

g_reg    = benchmark(program.matmul_register,
                     global_reg, local_naive, args_std,
                     "K4 REGISTER 1D (WPT=4)")

g_trans  = benchmark(program.matmul_transposed,
                     global_rect, local_rect, args_rect,
                     "K5 TRANSPOSED + RECT TILES")

g_2dreg  = benchmark(program.matmul_2d_register,
                     global_rect, local_rect, args_rect,
                     "K6 2D REGISTER BLOCKING")

g_best   = benchmark(program.matmul_best,
                     global_rect, local_rect, args_rect,
                     "K8 PREFETCH + 2D REG (BEST)")

print("=" * 70)
print(f"\n Gains vs COALESCED :")
print(f"   K3 TILING           : {g_tile/g_coal:.2f}x")
print(f"   K4 REGISTER 1D      : {g_reg/g_coal:.2f}x")
print(f"   K5 TRANSPOSED       : {g_trans/g_coal:.2f}x")
print(f"   K6 2D REGISTER      : {g_2dreg/g_coal:.2f}x")
print(f"   K8 BEST (prefetch)  : {g_best/g_coal:.2f}x")
print(f"\n Gains vs NAIVE :")
print(f"   K8 BEST (prefetch)  : {g_best/g_naive:.2f}x")