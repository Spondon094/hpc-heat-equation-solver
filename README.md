# 🖥️ HPC — 2D Steady-State Heat Equation Solver

![C++](https://img.shields.io/badge/C++-OMP%20Parallel-blue?style=flat-square&logo=c%2B%2B)
![OpenMP](https://img.shields.io/badge/OpenMP-Parallelized-green?style=flat-square)
![LIKWID](https://img.shields.io/badge/LIKWID-Performance%20Analysis-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

A high-performance C++ implementation of a **2D steady-state heat equation solver** developed as part of the *Programming Techniques for Supercomputers* course (SS 2022) at FAU Erlangen-Nürnberg. The project implements and benchmarks **Conjugate Gradient (CG)** and **Preconditioned Conjugate Gradient (PCG)** solvers with **OpenMP parallelization**, validated against Roofline model predictions on the **Fritz HPC cluster** (Intel Ice Lake, up to 72 cores).

> 📄 **Full Report:** [`PTFS_CAM_Project_Report.pdf`](./PTFS_CAM_Project_Report.pdf)

---

## 📐 Problem Overview

The solver numerically solves the 2D steady-state heat (Laplace) equation:

```
-Δu = f   on Ω = [0,1] × [0,1]
```

using a **5-point finite difference stencil** on a structured 2D grid. Two iterative solvers are implemented and compared:

- **CG** — Conjugate Gradient
- **PCG** — Preconditioned Conjugate Gradient (with Gauss-Seidel preconditioner)

---

## 🛠️ Implementation

### Core Components

| File | Description |
|---|---|
| `src/Grid.cpp` / `include/Grid.h` | 2D grid data structure with halo support, boundary conditions, `axpby`, `dotProduct` |
| `src/PDE.cpp` / `include/PDE.h` | PDE operator — `applyStencil` (5-point), `GSPreCon` (GS preconditioner), solver interface |
| `src/Solver.cpp` / `include/Solver.h` | CG and PCG solver implementations |
| `src/perf.cpp` | Performance benchmarking binary |
| `src/test.cpp` | Correctness test suite |
| `src/timer.cpp` / `include/timer.h` | High-resolution timing infrastructure |
| `include/types.h` | Enums for boundary types, directions, solver types |
| `include/test_macros.h` | Colored test output macros |
| `Makefile` | Build system supporting `g++` and Intel `icpc`, with optional LIKWID |

### Parallelization Strategy

OpenMP parallelization was applied to all performance-critical kernels:

- `applyStencil` — `#pragma omp parallel for schedule(static)`
- `axpby` — `#pragma omp parallel for schedule(static)`
- `dotProduct` — `#pragma omp parallel for reduction(+:dot_res) schedule(static)`
- `GSPreCon` — **wavefront parallelism** with `#pragma omp barrier` for forward/backward substitution dependency handling
- Grid initialization and boundary routines — fully parallelized

---

## 📊 Key Results

### Roofline Model vs Real Performance (1 socket, 18 cores, Fritz)

| Solver | Grid Size | Expected (MLUP/s) | Measured (MLUP/s) |
|---|---|---|---|
| CG | 2000 × 20000 | 633 | 665 |
| CG | 1000 × 400000 | 559 | 595 |
| PCG | 2000 × 20000 | 431 | 389 |
| PCG | 1000 × 400000 | 365 | 385 |

Roofline predictions closely matched real hardware measurements. CG slightly exceeded predictions due to hardware prefetching effects.

### Scaling: 18 cores → 72 cores (1 ccNUMA → 4 ccNUMA)

| Solver | Grid Size | 18 cores (MLUP/s) | 72 cores (MLUP/s) |
|---|---|---|---|
| CG | 2000 × 20000 | 658 | 2710 |
| CG | 1000 × 400000 | 595 | 2332 |
| PCG | 2000 × 20000 | 387 | 786 |
| PCG | 1000 × 400000 | 383 | 830 |

CG scales near-linearly. PCG shows limited scaling due to sequential dependencies in the GS preconditioner (wavefront parallelism overhead).

### Layer Condition Analysis

| Grid xSize | Layer Condition | Data Source |
|---|---|---|
| 20000 | ✅ Fits in L3 cache (1728 ≤ 5400 kB) | L3 cache |
| 400000 | ❌ Violates (3456 > 540 kB) | Main memory |

Performance drops for larger grids because data must be loaded from main memory (76 GB/s bandwidth bound).

---

## 📁 Repository Structure

```
hpc-heat-equation-solver/
│
├── README.md
├── Makefile
├── PTFS_CAM_Project_Report.pdf
│
├── include/
│   ├── Grid.h
│   ├── PDE.h
│   ├── Solver.h
│   ├── timer.h
│   ├── types.h
│   └── test_macros.h
│
└── src/
    ├── Grid.cpp
    ├── PDE.cpp
    ├── Solver.cpp
    ├── perf.cpp
    ├── test.cpp
    └── timer.cpp
```

---

## 🚀 Build & Run

### Requirements
- C++ compiler: `g++` or Intel `icpc`
- OpenMP support
- (Optional) LIKWID for hardware performance counters

### Compile

```bash
# Standard build with g++
CXX=g++ make

# With Intel compiler (recommended for HPC)
CXX=icpc make

# With LIKWID performance counters enabled
LIKWID=on CXX=icpc make
```

### Run Tests

```bash
./test
```

### Run Performance Benchmark

```bash
./perf <grid_size_y> <grid_size_x>

# Examples:
./perf 2000 20000
./perf 1000 400000
```

### Run with LIKWID on Fritz HPC Cluster

```bash
# Pin to 1 socket (18 cores), fixed frequency 2GHz
srun --cpu-freq=2000000-2000000 likwid-pin -C S0:0-17 ./perf 2000 20000

# Pin to full node (72 cores, 4 ccNUMA domains)
srun --cpu-freq=2000000-2000000 likwid-pin -C S0:0-71 ./perf 2000 20000

# Measure memory code balance with LIKWID performance counters
srun --cpu-freq=2000000-2000000 likwid-perfctr -C S0:0-17 -g MEM -m ./perf 2000 20000
```

---

## 💡 What I Learned

- Applying the **Roofline model** to predict and validate performance on real HPC hardware
- Analyzing **Layer Conditions** to determine whether data fits in L3 cache or requires main memory access
- Implementing **wavefront parallelism** for data-dependent kernels (Gauss-Seidel preconditioner)
- Using **LIKWID** to measure memory bandwidth, code balance (B/LUP), and hardware counter data
- Understanding **ccNUMA scaling** and why memory-bandwidth-bound kernels don't scale perfectly
- Profiling with `schedule(static)` and `reduction` clauses for optimal OpenMP performance

---

## 📋 Course Information

- **Course:** Programming Techniques for Supercomputers (PTFS-CAM)
- **Institution:** FAU Erlangen-Nürnberg
- **Semester:** Summer Semester 2022
- **HPC Cluster:** Fritz (Intel Xeon Platinum 8360Y, Ice Lake SP, 4 sockets × 18 cores = 72 cores/node)

---

## 👤 Author

**Spondon Sarker**

