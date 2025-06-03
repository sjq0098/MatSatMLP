#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <hip/hip_runtime.h>
#include <immintrin.h>  // AVX intrinsics

// 矩阵尺寸（统一为 CPU/GPU 都用同样的维度）
static const int N = 1024;
static const int M = 2048;
static const int P = 512;

// 用于 CPU 的子块尺寸
static const int BLOCK_SIZE = 64;

// 用于 GPU（HIP/DCU）的子块尺寸
static const int TILE = 16;

// 计算总运算量和总内存访问量（以字节计）
inline void compute_metrics(double elapsed_sec, double& gflops, double& bandwidth_gbps) {
    // 对于矩阵乘法，FLOPs ≈ 2 * N * M * P
    double total_ops = 2.0 * static_cast<double>(N) * M * P;
    gflops = total_ops / elapsed_sec / 1e9;

    // 假设访问 A、B、C 各一次：字节 = sizeof(double) * (N*M + M*P + N*P)
    double memory_bytes = sizeof(double) * (static_cast<double>(N)*M + static_cast<double>(M)*P + static_cast<double>(N)*P);
    bandwidth_gbps = memory_bytes / elapsed_sec / 1e9;
}

// 初始化矩阵（以一维数组形式表示），用于随机填充浮点数
void init_matrix(std::vector<double>& mat, int rows, int cols, double lower = -1.0, double upper = 1.0) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(lower, upper);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = dist(gen);
    }
}

// 验证：比较两个矩阵是否在误差 tol 范围内相等
bool validate(const std::vector<double>& A, const std::vector<double>& B, int rows, int cols, double tol = 1e-6) {
    for (int i = 0; i < rows * cols; ++i) {
        if (std::abs(A[i] - B[i]) > tol) return false;
    }
    return true;
}

// ---------------- CPU 基线：三重循环 ----------------
void matmul_baseline(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::vector<double>& C,
                     int n, int m, int p) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            double sum = 0.0;
            for (int k = 0; k < m; ++k) {
                sum += A[i * m + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

// ---------------- CPU OpenMP 并行 ----------------
void matmul_openmp(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C,
                   int n, int m, int p) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            double sum = 0.0;
            for (int k = 0; k < m; ++k) {
                sum += A[i * m + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

// ---------------- CPU 子块化（Block Tiling）+ OpenMP ----------------
void matmul_block(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C,
                  int n, int m, int p, int block_size = BLOCK_SIZE) {
    std::fill(C.begin(), C.end(), 0.0);
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += block_size) {
        int i_max = std::min(ii + block_size, n);
        for (int kk = 0; kk < m; kk += block_size) {
            int k_max = std::min(kk + block_size, m);
            for (int jj = 0; jj < p; jj += block_size) {
                int j_max = std::min(jj + block_size, p);
                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        double a_val = A[i * m + k];
                        for (int j = jj; j < j_max; ++j) {
                            C[i * p + j] += a_val * B[k * p + j];
                        }
                    }
                }
            }
        }
    }
}

// ---------------- CPU 转置 + OpenMP ----------------
void matmul_transpose(const std::vector<double>& A,
                      const std::vector<double>& B,
                      std::vector<double>& C,
                      int n, int m, int p) {
    // 先转置 B -> Bt (维度 p x m)
    std::vector<double> Bt(p * m);
    for (int k = 0; k < m; ++k) {
        for (int j = 0; j < p; ++j) {
            Bt[j * m + k] = B[k * p + j];
        }
    }
    std::fill(C.begin(), C.end(), 0.0);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            const double* a_row = &A[i * m];
            const double* b_row = &Bt[j * m];
            double sum = 0.0;
            for (int k = 0; k < m; ++k) {
                sum += a_row[k] * b_row[k];
            }
            C[i * p + j] = sum;
        }
    }
}

// ---------------- CPU 纯 SIMD（AVX2） ----------------
void matmul_simd(const std::vector<double>& A,
                 const std::vector<double>& B,
                 std::vector<double>& C,
                 int n, int m, int p) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < n; ++i) {
        const double* a_row = &A[i * m];
        for (int j = 0; j < p; j += 4) {
            __m256d c_vec = _mm256_setzero_pd();
            for (int k = 0; k < m; ++k) {
                __m256d a_vec = _mm256_set1_pd(a_row[k]);
                __m256d b_vec = _mm256_loadu_pd(&B[k * p + j]);
                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
            }
            _mm256_storeu_pd(&C[i * p + j], c_vec);
        }
    }
}

// ---------------- CPU SIMD + OpenMP ----------------
void matmul_simd_openmp(const std::vector<double>& A,
                        const std::vector<double>& B,
                        std::vector<double>& C,
                        int n, int m, int p) {
    std::fill(C.begin(), C.end(), 0.0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const double* a_row = &A[i * m];
        for (int j = 0; j < p; j += 4) {
            __m256d c_vec = _mm256_setzero_pd();
            for (int k = 0; k < m; ++k) {
                __m256d a_vec = _mm256_set1_pd(a_row[k]);
                __m256d b_vec = _mm256_loadu_pd(&B[k * p + j]);
                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
            }
            _mm256_storeu_pd(&C[i * p + j], c_vec);
        }
    }
}

// ---------------- CPU SIMD + Block ----------------
void matmul_simd_block(const std::vector<double>& A,
                       const std::vector<double>& B,
                       std::vector<double>& C,
                       int n, int m, int p, int block_size = BLOCK_SIZE) {
    std::fill(C.begin(), C.end(), 0.0);
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += block_size) {
        int i_max = std::min(ii + block_size, n);
        for (int jj = 0; jj < p; jj += block_size) {
            int j_max = std::min(jj + block_size, p);
            for (int i = ii; i < i_max; ++i) {
                const double* a_row = &A[i * m];
                for (int j = jj; j < j_max; j += 4) {
                    __m256d c_vec = _mm256_setzero_pd();
                    for (int k = 0; k < m; ++k) {
                        __m256d a_vec = _mm256_set1_pd(a_row[k]);
                        __m256d b_vec = _mm256_loadu_pd(&B[k * p + j]);
                        c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                    }
                    _mm256_storeu_pd(&C[i * p + j], c_vec);
                }
            }
        }
    }
}

// ------------------- GPU（HIP/DCU）部分 -------------------

// GPU 核函数：每个线程块处理 TILE×TILE 子块，并使用 shared memory
__global__ void matmul_kernel(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C,
                              int n, int m, int p) {
    int tx = threadIdx.x;  // 0..TILE-1
    int ty = threadIdx.y;  // 0..TILE-1
    int row = blockIdx.y * TILE + ty;  // 全局行索引
    int col = blockIdx.x * TILE + tx;  // 全局列索引

    double sum = 0.0;
    int numTiles = (m + TILE - 1) / TILE;

    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE + tx;
        if (row < n && a_col < m) {
            As[ty][tx] = A[row * m + a_col];
        } else {
            As[ty][tx] = 0.0;
        }
        int b_row = t * TILE + ty;
        if (b_row < m && col < p) {
            Bs[ty][tx] = B[b_row * p + col];
        } else {
            Bs[ty][tx] = 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < n && col < p) {
        C[row * p + col] = sum;
    }
}

// 在 GPU 上执行一次矩阵乘法并测时、拷回、验证，同时计算性能指标
void run_gpu_benchmark(const std::vector<double>& A,
                       const std::vector<double>& B,
                       const std::vector<double>& C_ref) {
    std::cout << "---- GPU (HIP/DCU) Benchmark ----" << std::endl;

    // 1) 分配显存
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    hipError_t err;

    err = hipMalloc(&d_A, sizeof(double) * N * M);
    if (err != hipSuccess) {
        std::cerr << "[Error] hipMalloc d_A: " << hipGetErrorString(err) << std::endl;
        return;
    }
    err = hipMalloc(&d_B, sizeof(double) * M * P);
    if (err != hipSuccess) {
        std::cerr << "[Error] hipMalloc d_B: " << hipGetErrorString(err) << std::endl;
        hipFree(d_A);
        return;
    }
    err = hipMalloc(&d_C, sizeof(double) * N * P);
    if (err != hipSuccess) {
        std::cerr << "[Error] hipMalloc d_C: " << hipGetErrorString(err) << std::endl;
        hipFree(d_A);
        hipFree(d_B);
        return;
    }

    // 2) 拷贝数据到设备
    err = hipMemcpy(d_A, A.data(), sizeof(double) * N * M, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "[Error] hipMemcpy H2D A: " << hipGetErrorString(err) << std::endl;
        hipFree(d_A); hipFree(d_B); hipFree(d_C);
        return;
    }
    err = hipMemcpy(d_B, B.data(), sizeof(double) * M * P, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "[Error] hipMemcpy H2D B: " << hipGetErrorString(err) << std::endl;
        hipFree(d_A); hipFree(d_B); hipFree(d_C);
        return;
    }

    // 3) 配置网格与线程块
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((P + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // 4) 创建事件用于计时
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // 5) 启动核函数并记录时间
    hipEventRecord(start, nullptr);
    hipLaunchKernelGGL(
        matmul_kernel,
        gridDim, blockDim,
        0,        // Shared mem 由 __shared__ 自动分配
        nullptr,  // 默认 stream
        d_A, d_B, d_C, N, M, P
    );
    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[GPU] Kernel execution time: " << milliseconds << " ms" << std::endl;

    // 6) 计算 GPU 性能指标
    double elapsed_sec = static_cast<double>(milliseconds) / 1000.0;
    double gflops_gpu = 0.0, bw_gpu = 0.0;
    compute_metrics(elapsed_sec, gflops_gpu, bw_gpu);
    std::cout << "[GPU] Performance: " << gflops_gpu << " GELOP/S, Memory bandwidth: "
              << bw_gpu << " GB/s" << std::endl;

    // 7) 拷回结果到主机
    std::vector<double> C_gpu(N * P);
    err = hipMemcpy(C_gpu.data(), d_C, sizeof(double) * N * P, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "[Error] hipMemcpy D2H C: " << hipGetErrorString(err) << std::endl;
        hipFree(d_A); hipFree(d_B); hipFree(d_C);
        return;
    }

    // 8) 验证
    bool ok = validate(C_ref, C_gpu, N, P);
    std::cout << "[GPU] Validation: " << (ok ? "PASS" : "FAIL") << std::endl << std::endl;

    // 9) 释放资源
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}

int main() {
    std::cout << "================ Matrix Multiplication Benchmark =================" << std::endl;

    // 1) 分配并初始化矩阵 A、B、C_ref、C
    std::vector<double> A(N * M), B(M * P);
    std::vector<double> C_ref(N * P), C(N * P);

    init_matrix(A, N, M, -1.0, 1.0);
    init_matrix(B, M, P, -1.0, 1.0);

    // 预先计算总运算量和总内存访问量（常量，可重复使用）
    double total_ops = 2.0 * static_cast<double>(N) * M * P;
    double memory_bytes = sizeof(double) * (static_cast<double>(N)*M + static_cast<double>(M)*P + static_cast<double>(N)*P);

    // 2) 先用基线三重循环在 CPU 上算出 C_ref（为后续所有模式做验证）
    double t0 = omp_get_wtime();
    matmul_baseline(A, B, C_ref, N, M, P);
    double t1 = omp_get_wtime();
    double elapsed_base = t1 - t0;
    double elapsed_base_ms = elapsed_base * 1000.0;
    double gflops_base = 0.0, bw_base = 0.0;
    compute_metrics(elapsed_base, gflops_base, bw_base);
    std::cout << "[CPU][Baseline] Time elapsed: " << elapsed_base_ms << " ms"
              << ", Performance: " << gflops_base << " GELOP/S"
              << ", Memory bandwidth: " << bw_base << " GB/s" << std::endl << std::endl;

    // 3) CPU: OpenMP 并行
    {
        double start = omp_get_wtime();
        matmul_openmp(A, B, C, N, M, P);
        double end = omp_get_wtime();
        double elapsed = end - start;
        double elapsed_ms = elapsed * 1000.0;
        bool ok = validate(C, C_ref, N, P);
        double gflops = 0.0, bw = 0.0;
        compute_metrics(elapsed, gflops, bw);
        std::cout << "[CPU][OpenMP] Time elapsed: " << elapsed_ms << " ms"
                  << ", Performance: " << gflops << " GELOP/S"
                  << ", Memory bandwidth: " << bw << " GB/s"
                  << ", Valid: " << std::boolalpha << ok << std::endl << std::endl;
    }

    // 4) CPU: Block Tiling + OpenMP
    {
        double start = omp_get_wtime();
        matmul_block(A, B, C, N, M, P, BLOCK_SIZE);
        double end = omp_get_wtime();
        double elapsed = end - start;
        double elapsed_ms = elapsed * 1000.0;
        bool ok = validate(C, C_ref, N, P);
        double gflops = 0.0, bw = 0.0;
        compute_metrics(elapsed, gflops, bw);
        std::cout << "[CPU][Block Tiling] Time elapsed: " << elapsed_ms << " ms"
                  << ", Performance: " << gflops << " GELOP/S"
                  << ", Memory bandwidth: " << bw << " GB/s"
                  << ", Valid: " << std::boolalpha << ok << std::endl << std::endl;
    }

    // 5) CPU: Transpose + OpenMP
    {
        double start = omp_get_wtime();
        matmul_transpose(A, B, C, N, M, P);
        double end = omp_get_wtime();
        double elapsed = end - start;
        double elapsed_ms = elapsed * 1000.0;
        bool ok = validate(C, C_ref, N, P);
        double gflops = 0.0, bw = 0.0;
        compute_metrics(elapsed, gflops, bw);
        std::cout << "[CPU][Transpose] Time elapsed: " << elapsed_ms << " ms"
                  << ", Performance: " << gflops << " GELOP/S"
                  << ", Memory bandwidth: " << bw << " GB/s"
                  << ", Valid: " << std::boolalpha << ok << std::endl << std::endl;
    }

    // 6) CPU: SIMD （纯 AVX2）
    {
        double start = omp_get_wtime();
        matmul_simd(A, B, C, N, M, P);
        double end = omp_get_wtime();
        double elapsed = end - start;
        double elapsed_ms = elapsed * 1000.0;
        bool ok = validate(C, C_ref, N, P);
        double gflops = 0.0, bw = 0.0;
        compute_metrics(elapsed, gflops, bw);
        std::cout << "[CPU][SIMD] Time elapsed: " << elapsed_ms << " ms"
                  << ", Performance: " << gflops << " GELOP/S"
                  << ", Memory bandwidth: " << bw << " GB/s"
                  << ", Valid: " << std::boolalpha << ok << std::endl << std::endl;
    }

    // 7) CPU: SIMD + OpenMP
    {
        double start = omp_get_wtime();
        matmul_simd_openmp(A, B, C, N, M, P);
        double end = omp_get_wtime();
        double elapsed = end - start;
        double elapsed_ms = elapsed * 1000.0;
        bool ok = validate(C, C_ref, N, P);
        double gflops = 0.0, bw = 0.0;
        compute_metrics(elapsed, gflops, bw);
        std::cout << "[CPU][SIMD+OpenMP] Time elapsed: " << elapsed_ms << " ms"
                  << ", Performance: " << gflops << " GELOP/S"
                  << ", Memory bandwidth: " << bw << " GB/s"
                  << ", Valid: " << std::boolalpha << ok << std::endl << std::endl;
    }

    // 8) CPU: SIMD + Block
    {
        double start = omp_get_wtime();
        matmul_simd_block(A, B, C, N, M, P, BLOCK_SIZE);
        double end = omp_get_wtime();
        double elapsed = end - start;
        double elapsed_ms = elapsed * 1000.0;
        bool ok = validate(C, C_ref, N, P);
        double gflops = 0.0, bw = 0.0;
        compute_metrics(elapsed, gflops, bw);
        std::cout << "[CPU][SIMD+Block] Time elapsed: " << elapsed_ms << " ms"
                  << ", Performance: " << gflops << " GELOP/S"
                  << ", Memory bandwidth: " << bw << " GB/s"
                  << ", Valid: " << std::boolalpha << ok << std::endl << std::endl;
    }

    // 9) GPU 基准测试
    run_gpu_benchmark(A, B, C_ref);

    std::cout << "================ Benchmark Finished =================" << std::endl;
    return 0;
}
