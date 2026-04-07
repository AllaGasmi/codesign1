import pyopencl as cl
import numpy as np
import time
import threading
from typing import Tuple, Dict, List

# ============================================
# CONFIGURATION - N=4096
# ============================================
N         = 4096
TILE_SIZE = 32   # Taille du bloc pour mémoire locale
TSM       = 128  # Tile rows
TSN       = 128  # Tile cols
TSK       = 32   # Tile depth
WPTM      = 8    # Work per thread (rows)
WPTN      = 8    # Work per thread (cols)
WIDTH     = 4    # float4
WPT       = 4    # for register kernel

RTSM = TSM // WPTM   # = 16
RTSN = TSN // WPTN   # = 16

print("=" * 70)
print(f"MULTIPLICATION DE MATRICES - Taille {N}x{N}")
print(f"Paramètres: TSM={TSM}, TSN={TSN}, TSK={TSK}")
print(f"WPTM={WPTM}, WPTN={WPTN}, RTSM={RTSM}, RTSN={RTSN}")
print("=" * 70)

# ============================================
# GÉNÉRATION DES MATRICES
# ============================================
print("\nGénération des matrices...")
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# Version transposée de B (optimisation)
B_T = B.T.copy()
print(f"Matrices générées: {A.nbytes / 1024**2:.1f} MB chacune")

# Version float4 pour vectorisation
if N % 4 == 0:
    A_float4 = A.view(np.float32).reshape(-1, 4)
    B_float4 = B_T.view(np.float32).reshape(-1, 4)
    print("Mode float4 activé")
else:
    A_float4 = A
    B_float4 = B_T
    print("Mode float4 désactivé (taille non multiple de 4)")

# ============================================
# DÉTECTION DES PÉRIPHÉRIQUES
# ============================================
def get_all_gpus():
    """Détecte tous les GPUs disponibles"""
    gpus = []
    platforms = cl.get_platforms()
    
    for platform in platforms:
        devices = platform.get_devices()
        for device in devices:
            if device.type == cl.device_type.GPU:
                gpus.append({
                    'platform': platform,
                    'device': device,
                    'name': device.name,
                    'type': 'GPU'
                })
                print(f"  ✓ {device.name}")
    
    return gpus

print("\nDétection des GPUs:")
gpus = get_all_gpus()
if len(gpus) == 0:
    print("  Aucun GPU trouvé!")
    exit(1)

# ============================================
# KERNELS OPENCL COMPLETS
# ============================================
KERNEL_CODE = f"""
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
// KERNEL 1 : NAIVE
// ============================================
__kernel void matmul_naive(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row >= N || col >= N) return;
    float sum = 0.0f;
    for(int k = 0; k < N; k++)
        sum += A[row*N + k] * B[k*N + col];
    C[row*N + col] = sum;
}}

// ============================================
// KERNEL 2 : COALESCED (référence)
// ============================================
__kernel void matmul_coalesced(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= N || col >= N) return;
    float sum = 0.0f;
    for(int k = 0; k < N; k++)
        sum += A[row*N + k] * B[k*N + col];
    C[row*N + col] = sum;
}}

// ============================================
// KERNEL 3 : TILING (mémoire locale)
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
    
    if (row >= N || col >= N) return;

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
// KERNEL 4 : REGISTER TILING 1D
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
    
    if (row >= N) return;

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
                if (col + w < N)
                    results[w] += a * Bsub[k][localCol + w];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    for(int w = 0; w < WPT; w++)
        if (col + w < N)
            C[row*N + col + w] = results[w];
}}

// ============================================
// KERNEL 5 : TRANSPOSED + RECTANGULAR TILES
// ============================================
__kernel void matmul_transposed(
    __global float* A,
    __global float* B_transposed,
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

    int numTiles = N / TSK;
    for(int t = 0; t < numTiles; t++) {{
        for(int wm = 0; wm < WPTM; wm++) {{
            int row = offsetM + tidm + wm*RTSM;
            int col = t*TSK + tidn;
            if (row < N && col < N)
                Asub[tidn][tidm + wm*RTSM] = A[row*N + col];
        }}
        for(int wn = 0; wn < WPTN; wn++) {{
            int row = t*TSK + tidm;
            int col = offsetN + tidn + wn*RTSN;
            if (row < N && col < N)
                Bsub[tidn + wn*RTSN][tidm] = B_transposed[row*N + col];
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
        for(int wn = 0; wn < WPTN; wn++) {{
            int row = offsetM + tidm + wm*RTSM;
            int col = offsetN + tidn + wn*RTSN;
            if (row < N && col < N)
                C[row*N + col] = acc[wm][wn];
        }}
}}

// ============================================
// KERNEL 6 : 2D REGISTER BLOCKING (MEILLEUR)
// ============================================
__kernel void matmul_2d_register(
    __global float* A,
    __global float* B_transposed,
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
            Bsub[tidn + wn*RTSN][tidm] = B_transposed[(t*TSK + tidm)*N + col];
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
        for(int wn = 0; wn < WPTN; wn++) {{
            int row = offsetM + tidm + wm*RTSM;
            int col = offsetN + tidn + wn*RTSN;
            if (row < N && col < N)
                C[row*N + col] = acc[wm][wn];
        }}
}}

// ============================================
// KERNEL 7 : UNCOALSCED (baseline NVIDIA)
// ============================================
__kernel void matmul_uncoalesced(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row >= N || col >= N) return;
    float sum = 0.0f;
    for(int k = 0; k < N; k++)
        sum += A[row*N + k] * B[k*N + col];
    C[row*N + col] = sum;
}}
"""

# ============================================
# CLASSE DE BENCHMARK
# ============================================
class MatrixBenchmark:
    def __init__(self, device_info: dict, N: int):
        self.device_info = device_info
        self.N = N
        self.ctx = cl.Context([device_info['device']])
        self.queue = cl.CommandQueue(self.ctx, 
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        # Compilation du programme
        print(f"  Compilation pour {device_info['name']}...")
        self.program = cl.Program(self.ctx, KERNEL_CODE)
        
        # Options de compilation
        opts = []
        if 'NVIDIA' in device_info['name']:
            opts = ['-cl-nv-maxrregcount=128', '-cl-mad-enable', '-cl-fast-relaxed-math']
        elif 'Intel' in device_info['name']:
            opts = ['-cl-fast-relaxed-math', '-cl-mad-enable']
        
        try:
            # CORRECTION: build avec les options comme arguments séparés
            if opts:
                self.program.build(options=' '.join(opts))
            else:
                self.program.build()
        except cl.LogicError as e:
            print(f"    Erreur compilation: {e}")
            print(self.program.get_build_info(device_info['device'], cl.program_build_info.LOG))
            raise
        
        # Buffers
        mf = cl.mem_flags
        self.A_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        self.B_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        self.BT_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_T)
        self.C_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, C.nbytes)
    
    def benchmark_kernel(self, kernel_name: str, global_size: tuple, local_size: tuple, 
                         args: list) -> Tuple[float, float]:
        """Exécute et mesure un kernel"""
        kernel = getattr(self.program, kernel_name)
        
        # Warmup
        ev = kernel(self.queue, global_size, local_size, *args)
        ev.wait()
        
        # Mesure (3 répétitions)
        times = []
        for _ in range(3):
            ev = kernel(self.queue, global_size, local_size, *args)
            ev.wait()
            t = (ev.profile.end - ev.profile.start) * 1e-9
            times.append(t)
        
        t_min = min(times)
        gflops = (2.0 * self.N ** 3) / (t_min * 1e9)
        
        return gflops, t_min * 1000
    
    def run_all_benchmarks(self) -> Dict[str, float]:
        """Exécute tous les benchmarks sur ce périphérique"""
        results = {}
        
        local_naive = (TILE_SIZE, TILE_SIZE)
        global_naive = (self.N, self.N)
        
        local_reg = (TILE_SIZE, TILE_SIZE)
        global_reg = (self.N // WPT, self.N)
        
        local_rect = (RTSM, RTSN)
        global_rect = (self.N * RTSM // TSM, self.N * RTSN // TSN)
        
        print(f"\n--- Benchmark sur {self.device_info['name']} ---")
        
        # Kernel 1: Naive
        g, t = self.benchmark_kernel('matmul_naive', global_naive, local_naive,
                                      [self.A_buf, self.B_buf, self.C_buf, np.int32(self.N)])
        results['NAIVE'] = g
        print(f"  NAIVE            : {t:.2f} ms -> {g:.2f} GFLOPS")
        
        # Kernel 2: Coalesced
        g, t = self.benchmark_kernel('matmul_coalesced', global_naive, local_naive,
                                      [self.A_buf, self.B_buf, self.C_buf, np.int32(self.N)])
        results['COALESCED'] = g
        print(f"  COALESCED        : {t:.2f} ms -> {g:.2f} GFLOPS")
        
        # Kernel 3: Tiling
        g, t = self.benchmark_kernel('matmul_tiling', global_naive, local_naive,
                                      [self.A_buf, self.B_buf, self.C_buf, np.int32(self.N)])
        results['TILING'] = g
        print(f"  TILING           : {t:.2f} ms -> {g:.2f} GFLOPS")
        
        # Kernel 4: Register 1D
        g, t = self.benchmark_kernel('matmul_register', global_reg, local_reg,
                                      [self.A_buf, self.B_buf, self.C_buf, np.int32(self.N)])
        results['REGISTER_1D'] = g
        print(f"  REGISTER 1D      : {t:.2f} ms -> {g:.2f} GFLOPS")
        
        # Kernel 5: Transposed
        g, t = self.benchmark_kernel('matmul_transposed', global_rect, local_rect,
                                      [self.A_buf, self.BT_buf, self.C_buf, np.int32(self.N)])
        results['TRANSPOSED'] = g
        print(f"  TRANSPOSED       : {t:.2f} ms -> {g:.2f} GFLOPS")
        
        # Kernel 6: 2D Register (BEST)
        g, t = self.benchmark_kernel('matmul_2d_register', global_rect, local_rect,
                                      [self.A_buf, self.BT_buf, self.C_buf, np.int32(self.N)])
        results['2D_REGISTER_BEST'] = g
        print(f"  2D_REGISTER_BEST : {t:.2f} ms -> {g:.2f} GFLOPS")
        
        # Kernel 7: Uncoalesced
        g, t = self.benchmark_kernel('matmul_uncoalesced', global_naive, local_naive,
                                      [self.A_buf, self.B_buf, self.C_buf, np.int32(self.N)])
        results['UNCOALESCED'] = g
        print(f"  UNCOALESCED      : {t:.2f} ms -> {g:.2f} GFLOPS")
        
        return results

# ============================================
# EXÉCUTION SUR DEUX GPUs
# ============================================
class MultiDeviceExecutor:
    def __init__(self, N: int):
        self.N = N
        self.gpus = get_all_gpus()
        
    def calculate_optimal_split(self, perf_nvidia: float, perf_other: float) -> Tuple[int, int]:
        """Calcule le split optimal basé sur les performances"""
        total_perf = perf_nvidia + perf_other
        if total_perf == 0:
            return (self.N // 2, self.N // 2)
        
        nvidia_ratio = perf_nvidia / total_perf
        block_size = TSM
        
        nvidia_rows = int((self.N * nvidia_ratio) // block_size) * block_size
        other_rows = self.N - nvidia_rows
        
        if other_rows % block_size != 0:
            nvidia_rows = ((self.N - other_rows) // block_size) * block_size
            other_rows = self.N - nvidia_rows
        
        return nvidia_rows, other_rows
    
    def run_dual_gpu(self):
        """Exécute sur deux GPUs en parallèle"""
        if len(self.gpus) < 2:
            print("Pas assez de GPUs pour l'exécution duale")
            return None
        
        # Identifier NVIDIA et autre GPU
        nvidia = None
        other = None
        
        for gpu in self.gpus:
            if 'NVIDIA' in gpu['name']:
                nvidia = gpu
            else:
                other = gpu
        
        if not nvidia:
            print("GPU NVIDIA non trouvé, utilisation du premier GPU comme baseline")
            nvidia = self.gpus[0]
            other = self.gpus[1] if len(self.gpus) > 1 else self.gpus[0]
        
        print("\n" + "=" * 70)
        print("BENCHMARK INDIVIDUEL DES GPUs")
        print("=" * 70)
        
        print("\n[GPU 1]")
        bench_nvidia = MatrixBenchmark(nvidia, self.N)
        perf_nvidia = bench_nvidia.run_all_benchmarks()
        
        print("\n[GPU 2]")
        bench_other = MatrixBenchmark(other, self.N)
        perf_other = bench_other.run_all_benchmarks()
        
        # Split optimal
        nvidia_rows, other_rows = self.calculate_optimal_split(
            perf_nvidia['UNCOALESCED'], perf_other['2D_REGISTER_BEST'])
        
        print("\n" + "=" * 70)
        print("EXÉCUTION SUR DEUX GPUs EN PARALLÈLE")
        print("=" * 70)
        print(f"  GPU1 ({nvidia['name']}) - UNCOALESCED: {nvidia_rows} lignes")
        print(f"  GPU2 ({other['name']}) - 2D_REGISTER_BEST: {other_rows} lignes")
        
        # Résultats
        results = {'nvidia': None, 'other': None}
        
        def run_nvidia():
            ctx = cl.Context([nvidia['device']])
            queue = cl.CommandQueue(ctx)
            
            A_nvidia = A[:nvidia_rows, :].copy()
            C_nvidia = np.zeros((nvidia_rows, self.N), dtype=np.float32)
            
            mf = cl.mem_flags
            d_A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_nvidia)
            d_B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
            d_C = cl.Buffer(ctx, mf.WRITE_ONLY, C_nvidia.nbytes)
            
            program = cl.Program(ctx, KERNEL_CODE)
            program.build()
            kernel = program.matmul_uncoalesced
            
            start = time.time()
            kernel(queue, (self.N, nvidia_rows), (16, 16), d_A, d_B, d_C, np.int32(self.N))
            queue.finish()
            elapsed = time.time() - start
            
            cl.enqueue_copy(queue, C_nvidia, d_C)
            gflops = 2 * self.N ** 3 / (elapsed * 1e9)
            results['nvidia'] = {'data': C_nvidia, 'time': elapsed, 'gflops': gflops}
            print(f"  ✓ GPU1 terminé: {elapsed:.3f}s ({gflops:.1f} GFLOPS)")
        
        def run_other():
            ctx = cl.Context([other['device']])
            queue = cl.CommandQueue(ctx)
            
            A_other = A[other_rows:, :].copy()
            C_other = np.zeros((other_rows, self.N), dtype=np.float32)
            
            mf = cl.mem_flags
            d_A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_other)
            d_B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_T)
            d_C = cl.Buffer(ctx, mf.WRITE_ONLY, C_other.nbytes)
            
            program = cl.Program(ctx, KERNEL_CODE)
            program.build()
            kernel = program.matmul_2d_register
            
            local_rect = (RTSM, RTSN)
            global_rect = (other_rows * RTSM // TSM, self.N * RTSN // TSN)
            
            start = time.time()
            kernel(queue, global_rect, local_rect, d_A, d_B, d_C, np.int32(self.N))
            queue.finish()
            elapsed = time.time() - start
            
            cl.enqueue_copy(queue, C_other, d_C)
            gflops = 2 * self.N ** 3 / (elapsed * 1e9)
            results['other'] = {'data': C_other, 'time': elapsed, 'gflops': gflops}
            print(f"  ✓ GPU2 terminé: {elapsed:.3f}s ({gflops:.1f} GFLOPS)")
        
        # Exécution parallèle
        t1 = threading.Thread(target=run_nvidia)
        t2 = threading.Thread(target=run_other)
        
        start_total = time.time()
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        total_time = time.time() - start_total
        
        # Assemblage
        C_final = np.zeros((self.N, self.N), dtype=np.float32)
        C_final[:nvidia_rows, :] = results['nvidia']['data']
        C_final[other_rows:, :] = results['other']['data']
        
        # Speedup
        baseline_gflops = perf_nvidia['UNCOALESCED']
        combined_gflops = (2.0 * self.N ** 3) / (total_time * 1e9)
        speedup = combined_gflops / baseline_gflops if baseline_gflops > 0 else 0
        
        print("\n" + "=" * 70)
        print("RÉSULTATS FINAUX")
        print("=" * 70)
        print(f"  Baseline (UNCOALESCED sur {nvidia['name']}): {baseline_gflops:.2f} GFLOPS")
        print(f"  Deux GPUs combinés: {combined_gflops:.2f} GFLOPS")
        print(f"  Temps total: {total_time:.3f}s")
        print(f"  SPEEDUP: {speedup:.2f}x")
        print("=" * 70)
        
        return {
            'baseline_gflops': baseline_gflops,
            'combined_gflops': combined_gflops,
            'speedup': speedup,
            'nvidia_rows': nvidia_rows,
            'other_rows': other_rows
        }

# ============================================
# MAIN
# ============================================
def main():
    print("\n" + "=" * 70)
    print("OPTIMISATION MULTIPLICATION DE MATRICES OPENCL")
    print(f"Taille matrice: {N}x{N}")
    print("=" * 70)
    
    executor = MultiDeviceExecutor(N)
    results = executor.run_dual_gpu()
    
    if results:
        print(f"\n✓ Speedup final: {results['speedup']:.2f}x")
        print(f"✓ Répartition: {results['nvidia_rows']} / {results['other_rows']} lignes")

if __name__ == "__main__":
    main()