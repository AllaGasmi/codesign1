import pyopencl as cl
import numpy as np

N = 4096
WPT = 4

A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

kernel_code = """
#define TILE_16 16
#define TILE_32 32
#define WPT 4

// ============================================
// KERNEL 1 : COALESCED (référence)
// ============================================
__kernel void matmul_coalesced(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    float sum = 0.0f;
    for(int k = 0; k < N; k++)
        sum += A[row*N + k] * B[k*N + col];
    C[row*N + col] = sum;
}

// ============================================
// KERNEL 2 : TILING 16x16 (A+B en local)
// ============================================
__kernel void matmul_tiling16(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    __local float Asub[TILE_16][TILE_16];
    __local float Bsub[TILE_16][TILE_16];

    int row = get_global_id(1);
    int col = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    float sum = 0.0f;
    int numTiles = N / TILE_16;

    for(int t = 0; t < numTiles; t++) {
        Asub[localRow][localCol] = A[row*N + t*TILE_16 + localCol];
        Bsub[localRow][localCol] = B[(t*TILE_16 + localRow)*N + col];
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < TILE_16; k++)
            sum += Asub[localRow][k] * Bsub[k][localCol];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[row*N + col] = sum;
}

// ============================================
// KERNEL 3 : TILING 32x32 (A+B en local)
// ============================================
__kernel void matmul_tiling32(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    __local float Asub[TILE_32][TILE_32];
    __local float Bsub[TILE_32][TILE_32];

    int row = get_global_id(1);
    int col = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    float sum = 0.0f;
    int numTiles = N / TILE_32;

    for(int t = 0; t < numTiles; t++) {
        Asub[localRow][localCol] = A[row*N + t*TILE_32 + localCol];
        Bsub[localRow][localCol] = B[(t*TILE_32 + localRow)*N + col];
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < TILE_32; k++)
            sum += Asub[localRow][k] * Bsub[k][localCol];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[row*N + col] = sum;
}

// ============================================
// KERNEL 4 : TILING 16x16 (A seulement en local)
// ============================================
__kernel void matmul_tiling_Aonly(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    __local float Asub[TILE_16][TILE_16];

    int row = get_global_id(1);
    int col = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    float sum = 0.0f;
    int numTiles = N / TILE_16;

    for(int t = 0; t < numTiles; t++) {
        Asub[localRow][localCol] = A[row*N + t*TILE_16 + localCol];
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < TILE_16; k++)
            sum += Asub[localRow][k] * B[(t*TILE_16 + k)*N + col];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[row*N + col] = sum;
}

// ============================================
// KERNEL 5 : TILING 16x16 + LOOP UNROLLING
// ============================================
__kernel void matmul_unroll(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    __local float Asub[TILE_16][TILE_16];
    __local float Bsub[TILE_16][TILE_16];

    int row = get_global_id(1);
    int col = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    float sum = 0.0f;
    int numTiles = N / TILE_16;

    for(int t = 0; t < numTiles; t++) {
        Asub[localRow][localCol] = A[row*N + t*TILE_16 + localCol];
        Bsub[localRow][localCol] = B[(t*TILE_16 + localRow)*N + col];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Boucle déroulée manuellement (unroll x4)
        for(int k = 0; k < TILE_16; k += 4) {
            sum += Asub[localRow][k]   * Bsub[k][localCol];
            sum += Asub[localRow][k+1] * Bsub[k+1][localCol];
            sum += Asub[localRow][k+2] * Bsub[k+2][localCol];
            sum += Asub[localRow][k+3] * Bsub[k+3][localCol];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[row*N + col] = sum;
}

// ============================================
// KERNEL 6 : TILING 16x16 + WPT=4 (combiné)
// ============================================
__kernel void matmul_tiling_wpt(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    __local float Asub[TILE_16][TILE_16];

    int row    = get_global_id(1);
    int col    = get_global_id(0) * WPT;
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    float results[WPT];
    for(int w = 0; w < WPT; w++)
        results[w] = 0.0f;

    int numTiles = N / TILE_16;
    for(int t = 0; t < numTiles; t++) {
        Asub[localRow][localCol] = A[row*N + t*TILE_16 + localCol];
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < TILE_16; k++) {
            float a = Asub[localRow][k];
            for(int w = 0; w < WPT; w++)
                results[w] += a * B[(t*TILE_16 + k)*N + col + w];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for(int w = 0; w < WPT; w++)
        C[row*N + col + w] = results[w];
}
"""

program = cl.Program(ctx, kernel_code).build()

mf = cl.mem_flags
A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

local16 = (16, 16)
local32 = (32, 32)
global16 = (N, N)
global32 = (N, N)
global_wpt = (N//WPT, N)

def benchmark(kernel_func, global_s, local_s, label):
    ev = kernel_func(queue, global_s, local_s, A_buf, B_buf, C_buf, np.int32(N))
    ev.wait()
    ev = kernel_func(queue, global_s, local_s, A_buf, B_buf, C_buf, np.int32(N))
    ev.wait()
    t = (ev.profile.end - ev.profile.start) * 1e-9
    gflops = (2 * N**3) / (t * 1e9)
    print(f"{label:38s} : {t*1000:8.2f} ms  →  {gflops:7.2f} GFLOPS")
    return gflops

print("=" * 70)
print(f"  Matrix: {N}x{N}    WPT={WPT}")
print("=" * 70)

ref   = benchmark(program.matmul_coalesced,   global16,  local16, "COALESCED (référence)")
t16   = benchmark(program.matmul_tiling16,    global16,  local16, "TILING 16x16 (A+B local)")
t32   = benchmark(program.matmul_tiling32,    global32,  local32, "TILING 32x32 (A+B local)")
tAonly= benchmark(program.matmul_tiling_Aonly,global16,  local16, "TILING 16x16 (A only)")
unroll= benchmark(program.matmul_unroll,      global16,  local16, "TILING 16x16 + UNROLL x4")
twpt  = benchmark(program.matmul_tiling_wpt,  global_wpt,local16, "TILING 16x16 + WPT=4")

print("=" * 70)
print(f"\n📊 Gains vs COALESCED :")
print(f"   Tiling 16x16 (A+B)   : {t16/ref:.2f}x")
print(f"   Tiling 32x32 (A+B)   : {t32/ref:.2f}x")
print(f"   Tiling 16x16 (A only): {tAonly/ref:.2f}x")
print(f"   Tiling + Unroll x4   : {unroll/ref:.2f}x")
print(f"   Tiling + WPT=4       : {twpt/ref:.2f}x")
print(f"\n🏆 Meilleure technique  : ", end="")
best = max([t16,t32,tAonly,unroll,twpt])
names = ["Tiling 16x16","Tiling 32x32","Tiling A only","Tiling+Unroll","Tiling+WPT"]
print(names[[t16,t32,tAonly,unroll,twpt].index(best)], f"→ {best/ref:.2f}x")