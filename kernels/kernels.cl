// =============================================================
//  LAB 1 – OpenCL Matrix Multiplication Kernels
//  All constants injected via Python f-string before compilation
// =============================================================

// ---------------------------------------------------------
// KERNEL 1 : NAIVE (baseline – uncoalesced)
// ---------------------------------------------------------
__kernel void matmul_naive(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;
    for (int k = 0; k < N; k++)
        sum += A[row*N + k] * B[k*N + col];
    C[row*N + col] = sum;
}

// ---------------------------------------------------------
// KERNEL 2 : COALESCED (reference – swap id axes)
// ---------------------------------------------------------
__kernel void matmul_coalesced(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    float sum = 0.0f;
    for (int k = 0; k < N; k++)
        sum += A[row*N + k] * B[k*N + col];
    C[row*N + col] = sum;
}

// ---------------------------------------------------------
// KERNEL 3 : TILING – local (shared) memory
// ---------------------------------------------------------
__kernel void matmul_tiling(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    int row      = get_global_id(1);
    int col      = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    float sum = 0.0f;
    int numTiles = N / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        Asub[localRow][localCol] = A[row*N + t*TILE_SIZE + localCol];
        Bsub[localRow][localCol] = B[(t*TILE_SIZE + localRow)*N + col];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; k++)
            sum += Asub[localRow][k] * Bsub[k][localCol];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[row*N + col] = sum;
}

// ---------------------------------------------------------
// KERNEL 4 : INCREASED WORK PER THREAD (1-D register tiling)
//   Each thread computes WPT elements along the column axis.
// ---------------------------------------------------------
__kernel void matmul_work_per_thread(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    int row      = get_global_id(1);
    int col      = get_global_id(0) * WPT;
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    float results[WPT];
    for (int w = 0; w < WPT; w++) results[w] = 0.0f;

    int numTiles = N / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        Asub[localRow][localCol] = A[row*N + t*TILE_SIZE + localCol];
        Bsub[localRow][localCol] = B[(t*TILE_SIZE + localRow)*N + col];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; k++) {
            float a = Asub[localRow][k];
            for (int w = 0; w < WPT; w++)
                results[w] += a * Bsub[k][localCol + w];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < WPT; w++)
        C[row*N + col + w] = results[w];
}

// ---------------------------------------------------------
// KERNEL 5 : TRANSPOSED A + RECTANGULAR TILES
//   Asub stored transposed → no bank conflicts on column reads.
//   Each thread computes WPTM×WPTN elements (2-D work).
// ---------------------------------------------------------
__kernel void matmul_transposed(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK];

    int tidm    = get_local_id(0);
    int tidn    = get_local_id(1);
    int offsetM = TSM * get_group_id(0);
    int offsetN = TSN * get_group_id(1);

    float acc[WPTM][WPTN];
    for (int wm = 0; wm < WPTM; wm++)
        for (int wn = 0; wn < WPTN; wn++)
            acc[wm][wn] = 0.0f;

    int numTiles = N / TSK;
    for (int t = 0; t < numTiles; t++) {

        for (int wm = 0; wm < WPTM; wm++) {
            int row = offsetM + tidm + wm*RTSM;
            int col = t*TSK + tidn;
            Asub[tidn][tidm + wm*RTSM] = A[row*N + col];
        }
        for (int wn = 0; wn < WPTN; wn++) {
            int row = t*TSK + tidm;
            int col = offsetN + tidn + wn*RTSN;
            Bsub[tidn + wn*RTSN][tidm] = B[row*N + col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TSK; k++)
            for (int wm = 0; wm < WPTM; wm++)
                for (int wn = 0; wn < WPTN; wn++)
                    acc[wm][wn] += Asub[k][tidm + wm*RTSM]
                                 * Bsub[tidn + wn*RTSN][k];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int wm = 0; wm < WPTM; wm++)
        for (int wn = 0; wn < WPTN; wn++)
            C[(offsetM + tidm + wm*RTSM)*N
              + (offsetN + tidn + wn*RTSN)] = acc[wm][wn];
}

// ---------------------------------------------------------
// KERNEL 6 : 2-D REGISTER BLOCKING
//   Pre-loads one column of Asub and one row of Bsub into
//   private registers before the inner MAD loop → removes
//   repeated shared-memory reads.
// ---------------------------------------------------------
__kernel void matmul_2d_register(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK];

    int tidm    = get_local_id(0);
    int tidn    = get_local_id(1);
    int offsetM = TSM * get_group_id(0);
    int offsetN = TSN * get_group_id(1);

    float acc[WPTM][WPTN];
    for (int wm = 0; wm < WPTM; wm++)
        for (int wn = 0; wn < WPTN; wn++)
            acc[wm][wn] = 0.0f;

    float aReg[WPTM];
    float bReg[WPTN];

    int numTiles = N / TSK;
    for (int t = 0; t < numTiles; t++) {

        for (int wm = 0; wm < WPTM; wm++) {
            int row = offsetM + tidm + wm*RTSM;
            Asub[tidn][tidm + wm*RTSM] = A[row*N + t*TSK + tidn];
        }
        for (int wn = 0; wn < WPTN; wn++) {
            int col = offsetN + tidn + wn*RTSN;
            Bsub[tidn + wn*RTSN][tidm] = B[(t*TSK + tidm)*N + col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TSK; k++) {
            for (int wm = 0; wm < WPTM; wm++)
                aReg[wm] = Asub[k][tidm + wm*RTSM];
            for (int wn = 0; wn < WPTN; wn++)
                bReg[wn] = Bsub[tidn + wn*RTSN][k];
            for (int wm = 0; wm < WPTM; wm++)
                for (int wn = 0; wn < WPTN; wn++)
                    acc[wm][wn] += aReg[wm] * bReg[wn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int wm = 0; wm < WPTM; wm++)
        for (int wn = 0; wn < WPTN; wn++)
            C[(offsetM + tidm + wm*RTSM)*N
              + (offsetN + tidn + wn*RTSN)] = acc[wm][wn];
}

// ---------------------------------------------------------
// KERNEL 7 : WIDER DATA-TYPES (float4 vectorised loads)
//   Each thread loads WIDTH=4 floats at once along the k-axis
//   for A, and along the n-axis for B, using float4 casts.
//   tidn steps are halved (RTSN/WIDTH iterations) so each
//   float4 fills 4 consecutive rows/cols of the shared tile.
//   Shared memory layout stays scalar for simple MAD reuse.
// ---------------------------------------------------------
__kernel void matmul_float4(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK];

    int tidm    = get_local_id(0);   // 0..RTSM-1
    int tidn    = get_local_id(1);   // 0..RTSN-1
    int offsetM = TSM * get_group_id(0);
    int offsetN = TSN * get_group_id(1);

    float acc[WPTM][WPTN];
    for (int wm = 0; wm < WPTM; wm++)
        for (int wn = 0; wn < WPTN; wn++)
            acc[wm][wn] = 0.0f;

    // TSK/WIDTH = 16/4 = 4  ->  tidn covers k in groups of 4
    // RTSN=16 threads along n, each loads WIDTH=4 k-values per wm step
    int numTiles = N / TSK;
    for (int t = 0; t < numTiles; t++) {

        // Load A: float4 along k-axis, unpack into Asub[k][m] (transposed)
        // tidn iterates over TSK/WIDTH=4 k-groups  (tidn < RTSN, but only
        // the first TSK/WIDTH=4 values are needed here; the rest repeat)
        if (tidn < TSK / WIDTH) {
            for (int wm = 0; wm < WPTM; wm++) {
                int row    = offsetM + tidm + wm*RTSM;
                int k_base = t*TSK + tidn*WIDTH;
                float4 a4  = *((__global float4*)(A + row*N + k_base));
                Asub[tidn*WIDTH + 0][tidm + wm*RTSM] = a4.x;
                Asub[tidn*WIDTH + 1][tidm + wm*RTSM] = a4.y;
                Asub[tidn*WIDTH + 2][tidm + wm*RTSM] = a4.z;
                Asub[tidn*WIDTH + 3][tidm + wm*RTSM] = a4.w;
            }
        }

        // Load B: float4 along n-axis, unpack into Bsub[n][k]
        // tidm iterates over TSK rows; each thread loads WIDTH=4 n-values
        for (int wn = 0; wn < WPTN / WIDTH; wn++) {
            int k_row  = t*TSK + tidm;
            int n_base = offsetN + tidn*WIDTH + wn*(RTSN*WIDTH);
            float4 b4  = *((__global float4*)(B + k_row*N + n_base));
            Bsub[tidn*WIDTH + wn*(RTSN*WIDTH) + 0][tidm] = b4.x;
            Bsub[tidn*WIDTH + wn*(RTSN*WIDTH) + 1][tidm] = b4.y;
            Bsub[tidn*WIDTH + wn*(RTSN*WIDTH) + 2][tidm] = b4.z;
            Bsub[tidn*WIDTH + wn*(RTSN*WIDTH) + 3][tidm] = b4.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TSK; k++)
            for (int wm = 0; wm < WPTM; wm++)
                for (int wn = 0; wn < WPTN; wn++)
                    acc[wm][wn] += Asub[k][tidm + wm*RTSM]
                                 * Bsub[tidn + wn*RTSN][k];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int wm = 0; wm < WPTM; wm++)
        for (int wn = 0; wn < WPTN; wn++)
            C[(offsetM + tidm + wm*RTSM)*N
              + (offsetN + tidn + wn*RTSN)] = acc[wm][wn];
}

// ---------------------------------------------------------
// KERNEL 8 : WIDER LOADS + 2-D REGISTER BLOCKING
//   Same float4 loads as K7, but adds aReg[]/bReg[] caching
//   before the inner MAD loop (K6 technique).
//   Maximum arithmetic intensity: 4x fewer load instructions
//   AND zero repeated shared-memory reads in inner loop.
// ---------------------------------------------------------
__kernel void matmul_wider_register(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK];

    int tidm    = get_local_id(0);
    int tidn    = get_local_id(1);
    int offsetM = TSM * get_group_id(0);
    int offsetN = TSN * get_group_id(1);

    float acc[WPTM][WPTN];
    for (int wm = 0; wm < WPTM; wm++)
        for (int wn = 0; wn < WPTN; wn++)
            acc[wm][wn] = 0.0f;

    float aReg[WPTM];
    float bReg[WPTN];

    int numTiles = N / TSK;
    for (int t = 0; t < numTiles; t++) {

        // float4 load of A (same as K7)
        if (tidn < TSK / WIDTH) {
            for (int wm = 0; wm < WPTM; wm++) {
                int row    = offsetM + tidm + wm*RTSM;
                int k_base = t*TSK + tidn*WIDTH;
                float4 a4  = *((__global float4*)(A + row*N + k_base));
                Asub[tidn*WIDTH + 0][tidm + wm*RTSM] = a4.x;
                Asub[tidn*WIDTH + 1][tidm + wm*RTSM] = a4.y;
                Asub[tidn*WIDTH + 2][tidm + wm*RTSM] = a4.z;
                Asub[tidn*WIDTH + 3][tidm + wm*RTSM] = a4.w;
            }
        }

        // float4 load of B (same as K7)
        for (int wn = 0; wn < WPTN / WIDTH; wn++) {
            int k_row  = t*TSK + tidm;
            int n_base = offsetN + tidn*WIDTH + wn*(RTSN*WIDTH);
            float4 b4  = *((__global float4*)(B + k_row*N + n_base));
            Bsub[tidn*WIDTH + wn*(RTSN*WIDTH) + 0][tidm] = b4.x;
            Bsub[tidn*WIDTH + wn*(RTSN*WIDTH) + 1][tidm] = b4.y;
            Bsub[tidn*WIDTH + wn*(RTSN*WIDTH) + 2][tidm] = b4.z;
            Bsub[tidn*WIDTH + wn*(RTSN*WIDTH) + 3][tidm] = b4.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 2-D register blocking MAD (K6 technique)
        for (int k = 0; k < TSK; k++) {
            for (int wm = 0; wm < WPTM; wm++)
                aReg[wm] = Asub[k][tidm + wm*RTSM];
            for (int wn = 0; wn < WPTN; wn++)
                bReg[wn] = Bsub[tidn + wn*RTSN][k];
            for (int wm = 0; wm < WPTM; wm++)
                for (int wn = 0; wn < WPTN; wn++)
                    acc[wm][wn] += aReg[wm] * bReg[wn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int wm = 0; wm < WPTM; wm++)
        for (int wn = 0; wn < WPTN; wn++)
            C[(offsetM + tidm + wm*RTSM)*N
              + (offsetN + tidn + wn*RTSN)] = acc[wm][wn];
}

__kernel void matmul_best(
    __global float4* A,
    __global float4* B,
    __global float*  C,
    int N)
{
    int tidm = get_local_id(0);
    int tidn = get_local_id(1);
    int groupm = get_group_id(0);
    int groupn = get_group_id(1);
    
    int offsetM = TSM * groupm;
    int offsetN = TSN * groupn;
    int N4 = N / 4;
    
    // Sortie immédiate si hors limites
    if (offsetM + tidm >= N || offsetN + tidn >= N) {
        return;
    }
    
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK];
    
    float acc = 0.0f;
    
    // Boucle sur les tiles K
    for (int tile = 0; tile < N / TSK; tile++) {
        int k_start = tile * TSK;
        
        // Charger A dans shared memory
        for (int k = 0; k < TSK; k++) {
            int row = offsetM + tidm;
            int col = k_start + k;
            if (row < N && col < N) {
                float4 a_vec = A[row * N4 + col/4];
                // Accès correct aux composants d'un float4
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
            if (row < N && col < N) {
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
    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}