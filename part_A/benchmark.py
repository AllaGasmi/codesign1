"""
LAB 1 – Part A : Matrix Multiplication Kernel Benchmarks
=========================================================
Runs all 9 kernels on the first available GPU and prints
GFLOPS + gains relative to K2 (coalesced reference).

File layout
-----------
lab1/
  kernels/kernels.cl   ← all OpenCL kernel source code
  part_A/benchmark.py  ← this file
  part_B/multidev.py   ← multi-device runner
"""

import pyopencl as cl
import numpy as np
import os

# ── Configuration ────────────────────────────────────────────
N         = 4096
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

# ── Load kernel source ────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_KERNEL_SRC = os.path.join(_HERE, "..", "kernels", "kernels.cl")

with open(_KERNEL_SRC) as f:
    _raw = f.read()

kernel_code = f"""
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
""" + _raw

# ── OpenCL setup ──────────────────────────────────────────────
platform = cl.get_platforms()[0]
device   = platform.get_devices()[0]
ctx      = cl.Context([device])
queue    = cl.CommandQueue(ctx,
               properties=cl.command_queue_properties.PROFILING_ENABLE)

print(f"Device            : {device.name}")
print(f"Max local memory  : {device.local_mem_size} bytes")
print(f"Max WG size       : {device.max_work_group_size}")
print(f"Compute units     : {device.max_compute_units}")

# ── Matrices ─────────────────────────────────────────────────
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# ── Compile ───────────────────────────────────────────────────
program = cl.Program(ctx, kernel_code).build()

# ── Buffers ───────────────────────────────────────────────────
mf     = cl.mem_flags
A_buf  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=A)
B_buf  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=B)
C_buf  = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

# float4 views (same data, reinterpreted)
A4_buf = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=A)
B4_buf = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=B)
C4_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

# ── Work sizes ────────────────────────────────────────────────
local_naive  = (TILE_SIZE, TILE_SIZE)
global_naive = (N, N)

local_reg    = (TILE_SIZE, TILE_SIZE)
global_reg   = (N // WPT, N)

local_rect   = (RTSM, RTSN)
global_rect  = (N * RTSM // TSM, N * RTSN // TSN)

# ── Benchmark helper ──────────────────────────────────────────
def benchmark(kern, global_s, local_s, args, label, repeats=3):
    kern(queue, global_s, local_s, *args).wait()          # warmup
    times = []
    for _ in range(repeats):
        ev = kern(queue, global_s, local_s, *args)
        ev.wait()
        times.append((ev.profile.end - ev.profile.start) * 1e-9)
    t   = min(times)
    gf  = 2 * N**3 / (t * 1e9)
    print(f"{label:45s}: {t*1000:8.2f} ms  →  {gf:7.2f} GFLOPS")
    return gf

# ── Run ───────────────────────────────────────────────────────
std  = (A_buf,  B_buf,  C_buf,  np.int32(N))

print("\n" + "="*70)
print(f"  Matrix {N}×{N}   TSM={TSM} TSN={TSN} TSK={TSK}   WPTM={WPTM} WPTN={WPTN}")
print("="*70)

g_naive  = benchmark(program.matmul_naive,
                     global_naive, local_naive, std,
                     "K1  NAIVE (baseline)")

g_coal   = benchmark(program.matmul_coalesced,
                     global_naive, local_naive, std,
                     "K2  COALESCED (reference)")

g_tile   = benchmark(program.matmul_tiling,
                     global_naive, local_naive, std,
                     "K3  TILING – local memory")

g_wpt    = benchmark(program.matmul_work_per_thread,
                     global_reg,   local_reg,   std,
                     "K4  INCREASED WORK PER THREAD (WPT=4)")

g_trans  = benchmark(program.matmul_transposed,
                     global_rect,  local_rect,  std,
                     "K5  TRANSPOSED + RECT TILES")

g_2dreg  = benchmark(program.matmul_2d_register,
                     global_rect,  local_rect,  std,
                     "K6  2-D REGISTER BLOCKING")

g_f4     = benchmark(program.matmul_float4,
                     global_rect,  local_rect,  std,
                     "K7  WIDER DATA-TYPES (float4)")

g_widereg = benchmark(program.matmul_wider_register,
                      global_rect, local_rect,  std,
                      "K8  WIDER LOADS + 2-D REGISTER (NEW)")

g_best   = benchmark(program.matmul_best,
                     global_rect,  local_rect,  std,
                     "K9  BEST – prefetch + 2-D register")

print("="*70)
print("\n  Gains vs K2 COALESCED:")
for label, g in [("K3  Tiling",             g_tile),
                 ("K4  Work per thread",     g_wpt),
                 ("K5  Transposed",          g_trans),
                 ("K6  2-D register",        g_2dreg),
                 ("K7  float4",              g_f4),
                 ("K8  Wider + register",    g_widereg),
                 ("K9  BEST (prefetch)",     g_best)]:
    print(f"   {label:30s}: {g/g_coal:.2f}x")

print(f"\n  Overall gain K9 vs K1: {g_best/g_naive:.2f}x")
print("="*70)