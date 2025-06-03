#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// 编译：
// hipcc sourcefile_dcu.cpp -o outputfile_dcu
// 执行：
// ./outputfile_dcu

#define N 1024
#define M 2024
#define P 512

// 每个线程块内使用 TILE×TILE 矩阵块
#define TILE 16

// GPU 上的矩阵乘法核函数（使用 Shared Memory 进行块化优化）
__global__ void matmul_kernel(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C,
                              int n, int m, int p) {
    // 线程在线程块内的坐标
    int tx = threadIdx.x;  // 列内偏移 0..TILE-1
    int ty = threadIdx.y;  // 行内偏移 0..TILE-1

    // 计算当前线程负责的全局元素 (row, col)
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    // 在片内累加 result
    double sum = 0.0;

    // 分块循环：沿 k 方向，每次处理 TILE 宽度
    int numTiles = (m + TILE - 1) / TILE;
    // 声明 Shared Memory 存放 A_block 和 B_block
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    for (int t = 0; t < numTiles; ++t) {
        // 全局 A 坐标：(row, t*TILE + tx)
        int a_col = t * TILE + tx;
        if (row < n && a_col < m) {
            As[ty][tx] = A[row * m + a_col];
        } else {
            As[ty][tx] = 0.0;
        }

        // 全局 B 坐标：(t*TILE + ty, col)
        int b_row = t * TILE + ty;
        if (b_row < m && col < p) {
            Bs[ty][tx] = B[b_row * p + col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        // 等待块内所有线程将数据加载到 Shared Memory
        __syncthreads();

        // 在 Shared Memory 中执行 TILE × TILE 的子矩阵乘加
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 同步，准备加载下一个子块
        __syncthreads();
    }

    // 将累加结果写回 C（检查越界）
    if (row < n && col < p) {
        C[row * p + col] = sum;
    }
}

// 在主机端初始化矩阵，填充随机 [-1.0, 1.0] 的双精度数
void init_matrix(std::vector<double>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& x : mat) {
        x = dist(gen);
    }
}

// CPU 上的基线矩阵乘法，用于结果验证
void matmul_cpu(const std::vector<double>& A,
                const std::vector<double>& B,
                std::vector<double>& C) {
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

// 简单的误差验证函数，检查 CPU 结果与 GPU 结果是否一致
bool validate(const std::vector<double>& ref,
              const std::vector<double>& test) {
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i] - test[i]) > 1e-6) {
            return false;
        }
    }
    return true;
}

int main() {
    // 分配并初始化主机端矩阵 A, B, C_host, C_ref
    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    init_matrix(A);
    init_matrix(B);

    // 1) 在 CPU 上计算基线结果
    std::cout << "[CPU] Computing baseline..." << std::endl;
    matmul_cpu(A, B, C_ref);

    // 2) 在 GPU 上分配显存
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    hipError_t err;

    err = hipMalloc(&d_A, sizeof(double) * N * M);
    if (err != hipSuccess) {
        std::cerr << "hipMalloc d_A failed: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc(&d_B, sizeof(double) * M * P);
    if (err != hipSuccess) {
        std::cerr << "hipMalloc d_B failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_A);
        return 1;
    }
    err = hipMalloc(&d_C, sizeof(double) * N * P);
    if (err != hipSuccess) {
        std::cerr << "hipMalloc d_C failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_A);
        hipFree(d_B);
        return 1;
    }

    // 3) 将 A, B 拷贝到设备
    err = hipMemcpy(d_A, A.data(), sizeof(double) * N * M, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy H2D A failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C);
        return 1;
    }
    err = hipMemcpy(d_B, B.data(), sizeof(double) * M * P, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy H2D B failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C);
        return 1;
    }

    // 4) 设置 Kernel launch 参数
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((P + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // 5) 创建 HIP 事件用于测量时间
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // 6) 记录起始时间并 Launch Kernel
    hipEventRecord(start, nullptr);
    hipLaunchKernelGGL(
        matmul_kernel,
        gridDim, blockDim,         // grid, block
        0,                         // shared memory 大小（已由 __shared__ 定义，不需要动态分配）
        nullptr,                   // stream
        d_A, d_B, d_C, N, M, P     // kernel 参数
    );
    hipEventRecord(stop, nullptr);

    // 等待 Kernel 完成
    hipEventSynchronize(stop);

    // 7) 计算耗时（ms）
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[HIP] Kernel execution time: " << milliseconds << " ms" << std::endl;

    // 8) 将结果拷回主机
    err = hipMemcpy(C.data(), d_C, sizeof(double) * N * P, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy D2H C failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C);
        return 1;
    }

    // 9) 验证 GPU 结果是否与 CPU 基线一致
    bool ok = validate(C_ref, C);
    std::cout << "[HIP] Validation: " << (ok ? "PASS" : "FAIL") << std::endl;

    // 10) 释放显存与事件
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}

