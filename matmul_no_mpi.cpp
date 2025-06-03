#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>  // AVX intrinsics

// 编译执行方式参考（MPI 相关已去除）：
// g++ -fopenmp -O3 -march=native -o matmul_no_mpi matmul_no_mpi.cpp

// 运行 Baseline：
// ./matmul_no_mpi baseline
//
// 运行 OpenMP：
// ./matmul_no_mpi openmp
//
// 运行 子块并行（Block）：
// ./matmul_no_mpi block
//
// 运行 转置+OpenMP（Other）：
// ./matmul_no_mpi other
//
// 运行 SIMD（Transpose + AVX2）：
// ./matmul_no_mpi simd
//
// 运行 SIMD + OpenMP：
// ./matmul_no_mpi simd_openmp
//
// 运行 SIMD + Block（Cache + AVX2）：
// ./matmul_no_mpi simd_block

// 初始化矩阵（以一维数组形式表示），用于随机填充浮点数
void init_matrix(std::vector<double>& mat, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = dist(gen);
}

// 验证计算优化后的矩阵计算与 baseline 实现是否结果一致
bool validate(const std::vector<double>& A, const std::vector<double>& B,
              int rows, int cols, double tol = 1e-6) {
    for (int i = 0; i < rows * cols; ++i)
        if (std::abs(A[i] - B[i]) > tol) return false;
    return true;
}

// 基础的矩阵乘法 baseline 实现（使用一维数组）
// 计算 C = A * B，其中 A 是 N x M，B 是 M x P，C 是 N x P
void matmul_baseline(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::vector<double>& C,
                     int N, int M, int P) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                sum += A[i * M + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

// 方式1: 利用 OpenMP 进行多线程并发的编程
void matmul_openmp(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C,
                   int N, int M, int P) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                sum += A[i * M + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

// 方式2: 利用子块并行思想，进行缓存友好型的并行优化方法
void matmul_block_tiling(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C,
                         int N, int M, int P, int block_size = 64) {
    std::fill(C.begin(), C.end(), 0.0);

    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < N; ii += block_size) {
        int i_max = std::min(ii + block_size, N);
        for (int kk = 0; kk < M; kk += block_size) {
            int k_max = std::min(kk + block_size, M);
            for (int jj = 0; jj < P; jj += block_size) {
                int j_max = std::min(jj + block_size, P);
                // 块内矩阵乘
                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        double a_val = A[i * M + k];
                        for (int j = jj; j < j_max; ++j) {
                            C[i * P + j] += a_val * B[k * P + j];
                        }
                    }
                }
            }
        }
    }
}

// 方式4: 其他优化方式，比如先转置 B 矩阵以提高缓存局部性
void matmul_other(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C,
                  int N, int M, int P) {
    // 将 B 转置为 Bt（大小 P x M）
    std::vector<double> Bt(P * M);
    for (int k = 0; k < M; ++k) {
        for (int j = 0; j < P; ++j) {
            Bt[j * M + k] = B[k * P + j];
        }
    }

    std::fill(C.begin(), C.end(), 0.0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            const double* a_row = &A[i * M];
            const double* b_row = &Bt[j * M];
            for (int k = 0; k < M; ++k) {
                sum += a_row[k] * b_row[k];
            }
            C[i * P + j] = sum;
        }
    }
}

// 方式5’: 纯 SIMD（AVX2）优化，不先转置 B
void matmul_simd(const std::vector<double>& A,
                 const std::vector<double>& B,
                 std::vector<double>& C,
                 int N, int M, int P) {
    // 清零输出
    std::fill(C.begin(), C.end(), 0.0);

    // 单行单次处理 4 列，因为 __m256d 存 4 x double
    for (int i = 0; i < N; ++i) {
        const double* a_row = &A[i * M];
        for (int j = 0; j < P; j += 4) {
            __m256d c_vec = _mm256_setzero_pd();
            // 遍历 k，累计 A[i][k] * B[k][j..j+3]
            for (int k = 0; k < M; ++k) {
                // 把 A[i][k] 广播到 4 个 double
                __m256d a_vec = _mm256_set1_pd(a_row[k]);
                // 从 B 的第 k 行、第 j 列开始加载 4 个连在一起的 double
                __m256d b_vec = _mm256_loadu_pd(&B[k * P + j]);
                // 用 FMA 累加：c_vec += a_vec * b_vec
                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
            }
            // 把这个 4 元组结果存回 C[i][j..j+3]
            _mm256_storeu_pd(&C[i * P + j], c_vec);
        }
    }
}

// 方式6’: SIMD + OpenMP  
void matmul_simd_openmp(const std::vector<double>& A,
                        const std::vector<double>& B,
                        std::vector<double>& C,
                        int N, int M, int P) {
    std::fill(C.begin(), C.end(), 0.0);

    // 并行化最外层 i 循环
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const double* a_row = &A[i * M];
        for (int j = 0; j < P; j += 4) {
            __m256d c_vec = _mm256_setzero_pd();
            for (int k = 0; k < M; ++k) {
                __m256d a_vec = _mm256_set1_pd(a_row[k]);
                __m256d b_vec = _mm256_loadu_pd(&B[k * P + j]);
                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
            }
            _mm256_storeu_pd(&C[i * P + j], c_vec);
        }
    }
}

// 方式7’: SIMD + Cache（块化 + SIMD）
void matmul_simd_block(const std::vector<double>& A,
                       const std::vector<double>& B,
                       std::vector<double>& C,
                       int N, int M, int P, int block_size = 64) {
    std::fill(C.begin(), C.end(), 0.0);

    // 对 i/j 分块，保证每次处理的 i-range 和 j-range 足够“放进缓存”
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < N; ii += block_size) {
        int i_max = std::min(ii + block_size, N);
        for (int jj = 0; jj < P; jj += block_size) {
            int j_max = std::min(jj + block_size, P);
            for (int i = ii; i < i_max; ++i) {
                const double* a_row = &A[i * M];
                // 在 j-block 里，每次处理 4 列
                for (int j = jj; j < j_max; j += 4) {
                    __m256d c_vec = _mm256_setzero_pd();
                    for (int k = 0; k < M; ++k) {
                        __m256d a_vec = _mm256_set1_pd(a_row[k]);
                        __m256d b_vec = _mm256_loadu_pd(&B[k * P + j]);
                        c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                    }
                    _mm256_storeu_pd(&C[i * P + j], c_vec);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    const int N = 1024;
    const int M = 2048;
    const int P = 512;
    std::string mode = (argc >= 2) ? argv[1] : "baseline";

    std::vector<double> A(N * M), B(M * P);
    std::vector<double> C(N * P, 0.0), C_ref(N * P, 0.0);

    // 随机初始化 A 和 B
    init_matrix(A, N, M);
    init_matrix(B, M, P);

    // 先算一遍 baseline 用于后续验证
    double t0 = omp_get_wtime();
    matmul_baseline(A, B, C_ref, N, M, P);
    double t1 = omp_get_wtime();
    if (mode == "baseline") {
        std::cout << "[Baseline] Time elapsed: " << (t1 - t0) << " seconds" << std::endl;
        return 0;
    }

    if (mode == "openmp") {
        double start = omp_get_wtime();
        matmul_openmp(A, B, C, N, M, P);
        double end = omp_get_wtime();
        bool ok = validate(C, C_ref, N, P);
        std::cout << "[OpenMP] Time elapsed: " << (end - start)
                  << " seconds, Valid: " << std::boolalpha << ok << std::endl;
    }
    else if (mode == "block") {
        double start = omp_get_wtime();
        matmul_block_tiling(A, B, C, N, M, P, 64);
        double end = omp_get_wtime();
        bool ok = validate(C, C_ref, N, P);
        std::cout << "[Block Parallel] Time elapsed: " << (end - start)
                  << " seconds, Valid: " << std::boolalpha << ok << std::endl;
    }
    else if (mode == "other") {
        double start = omp_get_wtime();
        matmul_other(A, B, C, N, M, P);
        double end = omp_get_wtime();
        bool ok = validate(C, C_ref, N, P);
        std::cout << "[Other (Transpose)] Time elapsed: " << (end - start)
                  << " seconds, Valid: " << std::boolalpha << ok << std::endl;
    }
    else if (mode == "simd") {
        double start = omp_get_wtime();
        matmul_simd(A, B, C, N, M, P);
        double end = omp_get_wtime();
        bool ok = validate(C, C_ref, N, P);
        std::cout << "[SIMD] Time elapsed: " << (end - start)
                  << " seconds, Valid: " << std::boolalpha << ok << std::endl;
    }
    else if (mode == "simd_openmp") {
        double start = omp_get_wtime();
        matmul_simd_openmp(A, B, C, N, M, P);
        double end = omp_get_wtime();
        bool ok = validate(C, C_ref, N, P);
        std::cout << "[SIMD+OpenMP] Time elapsed: " << (end - start)
                  << " seconds, Valid: " << std::boolalpha << ok << std::endl;
    }
    else if (mode == "simd_block") {
        double start = omp_get_wtime();
        matmul_simd_block(A, B, C, N, M, P, 64);
        double end = omp_get_wtime();
        bool ok = validate(C, C_ref, N, P);
        std::cout << "[SIMD+Block] Time elapsed: " << (end - start)
                  << " seconds, Valid: " << std::boolalpha << ok << std::endl;
    }
    else {
        std::cerr << "Usage: ./matmul_no_mpi [baseline|openmp|block|other|simd|simd_openmp|simd_block]" << std::endl;
        return 1;
    }

    return 0;
}
