#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define BATCH 8192
#define I     10
#define H     20
#define O     5

// -----------------------------------------------------------------------------
// GPU 矩阵乘法内核：C = A * B
// A: 大小 M×N, row-major
// B: 大小 N×K, row-major
// C: 大小 M×K, row-major
// 每个线程计算 C[row][col] 的一个元素
// -----------------------------------------------------------------------------
__global__ void matmul_kernel(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C,
                              int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        double sum = 0.0;
        for (int n = 0; n < N; ++n) {
            sum += A[row * N + n] * B[n * K + col];
        }
        C[row * K + col] = sum;
    }
}

// -----------------------------------------------------------------------------
// GPU 加偏置内核：对矩阵 A 中的每个元素 A[row][col] 增加 bias[col]
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(double* A, const double* bias, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;
        A[idx] += bias[col];
    }
}

// -----------------------------------------------------------------------------
// GPU ReLU 激活内核：对 A 中的每个元素执行 A[i] = max(0, A[i])
// -----------------------------------------------------------------------------
__global__ void relu_kernel(double* A, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = A[idx] > 0.0 ? A[idx] : 0.0;
    }
}

// -----------------------------------------------------------------------------
// 在主机端用随机数初始化向量或矩阵（范围 [-1, +1]）
// -----------------------------------------------------------------------------
void random_init(std::vector<double>& mat) {
    for (auto& val : mat) {
        val = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
    }
}

// -----------------------------------------------------------------------------
// 简单检查输出（打印前 5 个样本，每个样本输出维度 O）
// -----------------------------------------------------------------------------
void print_sample_output(const std::vector<double>& Y) {
    std::cout << "Sample outputs (前 5 个样本，每个样本输出维度 O=" << O << "):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "  Y[" << i << "] = [ ";
        for (int j = 0; j < O; ++j) {
            std::cout << Y[i * O + j] << (j < O - 1 ? ", " : " ");
        }
        std::cout << "]\n";
    }
}

// -----------------------------------------------------------------------------
// CPU 前向传播：两层全连接网络
// -----------------------------------------------------------------------------
void cpu_forward(const std::vector<double>& X,
                 const std::vector<double>& W1,
                 const std::vector<double>& B1,
                 std::vector<double>& H_vals,
                 const std::vector<double>& W2,
                 const std::vector<double>& B2,
                 std::vector<double>& Y)
{
    // 第一层：H = X * W1，大小 BATCH×H
    for (int b = 0; b < BATCH; ++b) {
        for (int h = 0; h < H; ++h) {
            double sum = 0.0;
            for (int i = 0; i < I; ++i) {
                sum += X[b * I + i] * W1[i * H + h];
            }
            double v = sum + B1[h];
            H_vals[b * H + h] = (v > 0.0 ? v : 0.0);  // ReLU
        }
    }
    // 第二层：Y = H * W2，大小 BATCH×O
    for (int b = 0; b < BATCH; ++b) {
        for (int o = 0; o < O; ++o) {
            double sum = 0.0;
            for (int h = 0; h < H; ++h) {
                sum += H_vals[b * H + h] * W2[h * O + o];
            }
            Y[b * O + o] = sum + B2[o];
        }
    }
}

// -----------------------------------------------------------------------------
// 计算性能指标：GFLOPS 与内存带宽 (GB/s)
// -----------------------------------------------------------------------------
void compute_metrics(double elapsed_sec,
                     double total_ops,
                     double total_bytes,
                     double& gflops,
                     double& bandwidth_gbps)
{
    gflops = total_ops / elapsed_sec / 1e9;
    bandwidth_gbps = total_bytes / elapsed_sec / 1e9;
}

int main()
{
    // 1) 在主机端准备好所有矩阵/向量：X, W1, B1, H, W2, B2, Y
    std::vector<double> h_X(BATCH * I);
    std::vector<double> h_W1(I * H);
    std::vector<double> h_B1(H);
    std::vector<double> h_H(BATCH * H);
    std::vector<double> h_W2(H * O);
    std::vector<double> h_B2(O);
    std::vector<double> h_Y_cpu(BATCH * O);
    std::vector<double> h_Y_gpu(BATCH * O);

    // 随机初始化输入及权重/偏置（确保 CPU 和 GPU 使用同一套数据）
    srand(42);
    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    // 2) 计算总 FLOPs 和内存访问量（字节数），用于性能衡量
    //    第一层：2 * BATCH * I * H
    //    第二层：2 * BATCH * H * O
    double total_ops = 2.0 * (static_cast<double>(BATCH) * I * H
                            + static_cast<double>(BATCH) * H * O);

    //    内存访问量近似按以下读取/写入：X, W1, H(write), B1, H(read), W2, Y(write), B2
    double bytes_X   = static_cast<double>(BATCH) * I * sizeof(double);
    double bytes_W1  = static_cast<double>(I) * H * sizeof(double);
    double bytes_H1  = static_cast<double>(BATCH) * H * sizeof(double); // 写入 H
    double bytes_B1  = static_cast<double>(H) * sizeof(double);
    double bytes_H2  = static_cast<double>(BATCH) * H * sizeof(double); // 读取 H
    double bytes_W2  = static_cast<double>(H) * O * sizeof(double);
    double bytes_Y   = static_cast<double>(BATCH) * O * sizeof(double); // 写入 Y
    double bytes_B2  = static_cast<double>(O) * sizeof(double);
    double total_bytes = bytes_X + bytes_W1 + bytes_H1 + bytes_B1
                       + bytes_H2 + bytes_W2 + bytes_Y + bytes_B2;

    // ------------------- CPU 部分 -------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_forward(h_X, h_W1, h_B1, h_H, h_W2, h_B2, h_Y_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_diff = cpu_end - cpu_start;
    double cpu_elapsed = cpu_diff.count();
    double cpu_elapsed_ms = cpu_elapsed * 1000.0;

    double cpu_gflops = 0.0, cpu_bw = 0.0;
    compute_metrics(cpu_elapsed, total_ops, total_bytes, cpu_gflops, cpu_bw);

    std::cout << "================ CPU Forward Benchmark =================\n";
    std::cout << "CPU Time elapsed: " << cpu_elapsed_ms << " ms\n";
    std::cout << "CPU Performance: " << cpu_gflops << " GFLOP/s\n";
    std::cout << "CPU Approx. Memory Bandwidth: " << cpu_bw << " GB/s\n\n";

    // ------------------- GPU 部分 -------------------
    // 3) 在设备端分配显存
    double *d_X, *d_W1, *d_B1, *d_H, *d_W2, *d_B2, *d_Y;
    hipMalloc(&d_X,  sizeof(double) * BATCH * I);
    hipMalloc(&d_W1, sizeof(double) * I * H);
    hipMalloc(&d_B1, sizeof(double) * H);
    hipMalloc(&d_H,  sizeof(double) * BATCH * H);
    hipMalloc(&d_W2, sizeof(double) * H * O);
    hipMalloc(&d_B2, sizeof(double) * O);
    hipMalloc(&d_Y,  sizeof(double) * BATCH * O);

    // 4) 将数据从主机拷贝到设备
    hipMemcpy(d_X,  h_X.data(),  sizeof(double) * BATCH * I, hipMemcpyHostToDevice);
    hipMemcpy(d_W1, h_W1.data(), sizeof(double) * I * H, hipMemcpyHostToDevice);
    hipMemcpy(d_B1, h_B1.data(), sizeof(double) * H, hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), sizeof(double) * H * O, hipMemcpyHostToDevice);
    hipMemcpy(d_B2, h_B2.data(), sizeof(double) * O, hipMemcpyHostToDevice);

    // 配置 GPU 计时事件
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // 4.1) 第一层：H = X * W1
    dim3 block1(16, 16);
    dim3 grid1((H + block1.x - 1) / block1.x,
               (BATCH + block1.y - 1) / block1.y);

    hipEventRecord(start, nullptr);

    hipLaunchKernelGGL(
        matmul_kernel,
        grid1, block1,
        0, nullptr,
        d_X, d_W1, d_H,
        BATCH, I, H
    );
    hipDeviceSynchronize();

    // 加偏置并 ReLU
    {
        int total_hidden = BATCH * H;
        int threads = 256;
        int blocks  = (total_hidden + threads - 1) / threads;
        add_bias_kernel<<<blocks, threads>>>(d_H, d_B1, BATCH, H);
        hipDeviceSynchronize();
        relu_kernel<<<blocks, threads>>>(d_H, total_hidden);
        hipDeviceSynchronize();
    }

    // 4.2) 第二层：Y = H * W2
    dim3 block2(16, 16);
    dim3 grid2((O + block2.x - 1) / block2.x,
               (BATCH + block2.y - 1) / block2.y);

    hipLaunchKernelGGL(
        matmul_kernel,
        grid2, block2,
        0, nullptr,
        d_H, d_W2, d_Y,
        BATCH, H, O
    );
    hipDeviceSynchronize();

    // 第二层加偏置
    {
        int total_output = BATCH * O;
        int threads = 256;
        int blocks  = (total_output + threads - 1) / threads;
        add_bias_kernel<<<blocks, threads>>>(d_Y, d_B2, BATCH, O);
        hipDeviceSynchronize();
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);

    float gpu_milliseconds = 0.0f;
    hipEventElapsedTime(&gpu_milliseconds, start, stop);

    // 拷回结果到主机
    hipMemcpy(h_Y_gpu.data(), d_Y, sizeof(double) * BATCH * O, hipMemcpyDeviceToHost);

    // 释放显存和事件
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_X);
    hipFree(d_W1);
    hipFree(d_B1);
    hipFree(d_H);
    hipFree(d_W2);
    hipFree(d_B2);
    hipFree(d_Y);

    // 5) 计算 GPU 性能指标（仅核函数执行时间，不含拷贝）
    double gpu_elapsed = static_cast<double>(gpu_milliseconds) / 1000.0;
    double gpu_gflops = 0.0, gpu_bw = 0.0;
    compute_metrics(gpu_elapsed, total_ops, total_bytes, gpu_gflops, gpu_bw);

    std::cout << "================ GPU Forward Benchmark =================\n";
    std::cout << "GPU Kernel Time elapsed: " << gpu_milliseconds << " ms\n";
    std::cout << "GPU Performance: " << gpu_gflops << " GFLOP/s\n";
    std::cout << "GPU Approx. Memory Bandwidth: " << gpu_bw << " GB/s\n\n";

    // 6) 打印部分输出，用于验证
    std::cout << "--- CPU 前向结果（前 5 个样本） ---\n";
    print_sample_output(h_Y_cpu);
    std::cout << "\n--- GPU 前向结果（前 5 个样本） ---\n";
    print_sample_output(h_Y_gpu);

    // 7) 验证 CPU 与 GPU 输出一致性（允许小误差）
    bool valid = true;
    double tol = 1e-6;
    for (int i = 0; i < BATCH * O; ++i) {
        if (std::fabs(h_Y_cpu[i] - h_Y_gpu[i]) > tol) {
            valid = false;
            break;
        }
    }
    std::cout << "\nValidation: " << (valid ? "PASS" : "FAIL") << "\n";

    return 0;
}
