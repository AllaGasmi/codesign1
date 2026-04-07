LAB 1 – OpenCL Matrix Multiplication
=====================================

File structure
--------------
lab1/
├── kernels/
│   └── kernels.cl          All 9 OpenCL kernel functions (shared by A & B)
├── part_A/
│   └── benchmark.py        Runs all kernels on GPU 0, prints GFLOPS & gains
└── part_B/
    └── multidev.py         Splits work across RTX 3050 + Intel Iris Xe

How to run
----------
  # Part A
  cd part_A
  python benchmark.py

  # Part B
  cd part_B
  python multidev.py

Kernel summary
--------------
K1  matmul_naive             Baseline – uncoalesced global reads
K2  matmul_coalesced         Swap id(0)/id(1) → coalesced B reads  [reference]
K3  matmul_tiling            Shared (local) memory tile cache
K4  matmul_work_per_thread   1 thread → WPT=4 outputs (1-D register)
K5  matmul_transposed        Rectangular tiles (128×128) + A transposed (no bank conflicts)
K6  matmul_2d_register       Pre-load aReg[]/bReg[] before MAD loop
K7  matmul_float4            Vectorised float4 loads (4 floats per instruction)
K8  matmul_wider_register    float4 loads + 2-D register blocking combined  [NEW]
K9  matmul_best              Double-buffer prefetch + 2-D register blocking  [BEST]

Best kernel combination (K9)
-----------------------------
K5 rectangular tiles
+ K6 2-D register blocking  (aReg/bReg)
+ K8 double-buffer prefetch (load tile t+1 while computing tile t)
= K9  matmul_best
