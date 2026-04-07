import pyopencl as cl
import numpy as np
import time

# Taille de la matrice
N = 8192

# Initialisation des matrices
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# Choisir le device NVIDIA
platform = cl.get_platforms()[0]  # NVIDIA
device = platform.get_devices()[0]  # MX550
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Kernels OpenCL
kernel_code = """
// ============================================
// KERNEL 1 : NAIVE
// ============================================
__kernel void matmul_naive(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    float sum = 0.0f;
    for(int k = 0; k < N; k++) {
        sum += A[row*N + k] * B[k*N + col];
    }
    C[row*N + col] = sum;
}

// ============================================
// KERNEL 2 : COALESCED
// ============================================
__kernel void matmul_coalesced(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    float sum = 0.0f;
    for(int k = 0; k < N; k++) {
        sum += A[row*N + k] * B[k*N + col];
    }
    C[row*N + col] = sum;
}
"""


program = cl.Program(ctx, kernel_code).build()
mf = cl.mem_flags
A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

# Taille des work-groups
local_size = (16, 16)
global_size = (N, N)

# ============================================
# Benchmark NAIVE
# ============================================
event = program.matmul_naive(queue, global_size, local_size, 
                              A_buf, B_buf, C_buf, np.int32(N))
event.wait()
time_naive = (event.profile.end - event.profile.start) * 1e-9  # en secondes
gflops_naive = (2 * N**3) / (time_naive * 1e9)

print(f"NAIVE      : {time_naive*1000:.2f} ms  →  {gflops_naive:.2f} GFLOPS")

# ============================================
# Benchmark COALESCED
# ============================================
event = program.matmul_coalesced(queue, global_size, local_size,
                                  A_buf, B_buf, C_buf, np.int32(N))
event.wait()
time_coal = (event.profile.end - event.profile.start) * 1e-9
gflops_coal = (2 * N**3) / (time_coal * 1e9)

print(f"COALESCED  : {time_coal*1000:.2f} ms  →  {gflops_coal:.2f} GFLOPS")