import pyopencl as cl
import numpy as np

N = 4096

A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

def make_kernel(tile, wpt):
    return f"""
#define TILE {tile}
#define WPT {wpt}

__kernel void matmul_twpt(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{{
    __local float Asub[TILE][TILE];

    int row      = get_global_id(1);
    int col      = get_global_id(0) * WPT;
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    float results[WPT];
    for(int w = 0; w < WPT; w++)
        results[w] = 0.0f;

    int numTiles = N / TILE;
    for(int t = 0; t < numTiles; t++) {{
        Asub[localRow][localCol] = A[row*N + t*TILE + localCol];
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < TILE; k++) {{
            float a = Asub[localRow][k];
            for(int w = 0; w < WPT; w++)
                results[w] += a * B[(t*TILE + k)*N + col + w];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    for(int w = 0; w < WPT; w++)
        C[row*N + col + w] = results[w];
}}

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

__kernel void matmul_tiling32(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{{
    __local float Asub[32][32];
    __local float Bsub[32][32];

    int row      = get_global_id(1);
    int col      = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    float sum = 0.0f;
    int numTiles = N / 32;

    for(int t = 0; t < numTiles; t++) {{
        Asub[localRow][localCol] = A[row*N + t*32 + localCol];
        Bsub[localRow][localCol] = B[(t*32 + localRow)*N + col];
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < 32; k++)
            sum += Asub[localRow][k] * Bsub[k][localCol];
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    C[row*N + col] = sum;
}}
"""

mf = cl.mem_flags
A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

def benchmark(kernel_func, global_s, local_s):
    ev = kernel_func(queue, global_s, local_s, A_buf, B_buf, C_buf, np.int32(N))
    ev.wait()
    ev = kernel_func(queue, global_s, local_s, A_buf, B_buf, C_buf, np.int32(N))
    ev.wait()
    t = (ev.profile.end - ev.profile.start) * 1e-9
    return (2 * N**3) / (t * 1e9), t * 1000

print("=" * 65)
print(f"  Matrix: {N}x{N}")
print("=" * 65)

# Référence coalesced
prog_ref = cl.Program(ctx, make_kernel(16, 1)).build()
gflops_ref, t_ref = benchmark(prog_ref.matmul_coalesced, (N, N), (16, 16))
print(f"{'COALESCED (référence)':40s}: {t_ref:8.2f} ms → {gflops_ref:7.2f} GFLOPS  (1.00x)")

# Tiling 32x32 sans WPT
prog_t32 = cl.Program(ctx, make_kernel(32, 1)).build()
gflops_t32, t_t32 = benchmark(prog_t32.matmul_tiling32, (N, N), (32, 32))
print(f"{'Tiling 32x32 (sans WPT)':40s}: {t_t32:8.2f} ms → {gflops_t32:7.2f} GFLOPS  ({gflops_t32/gflops_ref:.2f}x)")

print("-" * 65)

# Tester différentes combinaisons TILE + WPT
configs = [
    (16, 2),
    (16, 4),
    (16, 8),
    (32, 2),
    (32, 4),
]

results = []
for tile, wpt in configs:
    label = f"Tiling {tile}x{tile} + WPT={wpt}"
    try:
        prog = cl.Program(ctx, make_kernel(tile, wpt)).build()
        local_s  = (tile, tile)
        global_s = (N // wpt, N)
        gflops, t = benchmark(prog.matmul_twpt, global_s, local_s)
        gain = gflops / gflops_ref
        print(f"{label:40s}: {t:8.2f} ms → {gflops:7.2f} GFLOPS  ({gain:.2f}x)")
        results.append((label, gflops, gain))
    except Exception as e:
        print(f"{label:40s}: ERREUR → {e}")

print("=" * 65)

all_results = [("Tiling 32x32", gflops_t32, gflops_t32/gflops_ref)] + results
valid = [(l, g, gain) for l, g, gain in all_results if gain > 0]

if valid:
    best = max(valid, key=lambda x: x[1])
    print(f"\n🏆 Meilleure combinaison : {best[0]}")
    print(f"   GFLOPS : {best[1]:.2f}")
    print(f"   Gain   : {best[2]:.2f}x vs COALESCED")