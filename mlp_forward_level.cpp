#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

#define BATCH 1024
#define I     10
#define H     20
#define O     5

// -----------------------------------------------------------------------------
// GPU 矩阵乘法内核（与您原来完全一致，不做改动）
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
// 第一层：Bias + ReLU 融合 Kernel（用常量偏置 c_B1）
// -----------------------------------------------------------------------------
__constant__ double c_B1[H];  // H=20
__global__ void add_bias_relu_kernel(double* A, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;                   // col < H
        double v = A[idx] + c_B1[col];          // 先加 bias
        A[idx] = (v > 0.0 ? v : 0.0);            // 再做 ReLU
    }
}

// -----------------------------------------------------------------------------
// 第二层：Bias 融合 Kernel（用常量偏置 c_B2）
// -----------------------------------------------------------------------------
__constant__ double c_B2[O];  // O=5
__global__ void add_bias_second_kernel(double* A, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;      
        A[idx] += c_B2[col];
    }
}

// -----------------------------------------------------------------------------
// Host 端随机初始化
// -----------------------------------------------------------------------------
void random_init(std::vector<double>& mat) {
    for (auto& val : mat) {
        val = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
    }
}

// -----------------------------------------------------------------------------
// 打印前 5 个输出样本，调试用
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
// 主函数：只修改了 Grid 计算、Bias 上传常量、Bias+ReLU 融合等；matmul_kernel 一律不动
// -----------------------------------------------------------------------------
int main()
{
    // 1) Host 端准备好所有矩阵/向量：X, W1, B1, H, W2, B2, Y
    std::vector<double> h_X(BATCH * I);
    std::vector<double> h_W1(I * H);
    std::vector<double> h_B1(H);
    std::vector<double> h_H(BATCH * H);
    std::vector<double> h_W2(H * O);
    std::vector<double> h_B2(O);
    std::vector<double> h_Y(BATCH * O);

    srand(42);
    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    // 2) 在 Device 上分配显存（与原来一模一样）
    double *d_X, *d_W1, *d_B1_unused, *d_H, *d_W2, *d_B2_unused, *d_Y;
    // 注意：原来我们还给 d_B1, d_B2 也分配了显存。但现在 bias 用常量内存储存，不需要再用 d_B1, d_B2 做 Kernel 输入。
    // 为了兼容后面的 hipMemcpy，本示例保持变量名不动，但不再把它们传给 Kernel。
    hipMalloc(&d_X,  sizeof(double) * BATCH * I);
    hipMalloc(&d_W1, sizeof(double) * I * H);
    hipMalloc(&d_H,  sizeof(double) * BATCH * H);
    hipMalloc(&d_W2, sizeof(double) * H * O);
    hipMalloc(&d_Y,  sizeof(double) * BATCH * O);

    // 3) 把权重和输入拷到 Device
    hipMemcpy(d_X,  h_X.data(), sizeof(double) * BATCH * I, hipMemcpyHostToDevice);
    hipMemcpy(d_W1, h_W1.data(), sizeof(double) * I * H, hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), sizeof(double) * H * O, hipMemcpyHostToDevice);

    // 4) 把 Bias1, Bias2 拷到 常量内存
    hipMemcpyToSymbol(HIP_SYMBOL(c_B1), h_B1.data(), sizeof(double) * H);
    hipMemcpyToSymbol(HIP_SYMBOL(c_B2), h_B2.data(), sizeof(double) * O);

    // --------- 计时 start（用 hipEvent） -----------
    hipEvent_t t_start, t_stop;
    hipEventCreate(&t_start);
    hipEventCreate(&t_stop);
    float time_first_layer = 0.0f, time_second_layer = 0.0f, time_total = 0.0f;

    hipEventRecord(t_start, 0);

    // 5) ----- 第一层： H = X * W1 -----
    dim3 block1(16, 16);
    // 修正：grid1.x 要用 H 而不是 I
    dim3 grid1( (H      + block1.x - 1) / block1.x,
                (BATCH + block1.y - 1) / block1.y );

    // 5.1）只做矩阵乘法
    hipEventRecord(t_start, 0);
    matmul_kernel<<< grid1, block1 >>>(
        d_X,        // A: BATCH×I
        d_W1,       // B: I×H
        d_H,        // C: BATCH×H
        BATCH, I, H
    );
    hipDeviceSynchronize();

    // 5.2）Bias+ReLU 融合
    {
        int total_hidden = BATCH * H;
        int threads = 256;
        int blocks  = (total_hidden + threads - 1) / threads;
        add_bias_relu_kernel<<< blocks, threads >>>(d_H, BATCH, H);
        hipDeviceSynchronize();
    }
    hipEventRecord(t_stop, 0);
    hipEventSynchronize(t_stop);
    hipEventElapsedTime(&time_first_layer, t_start, t_stop);

    // 6) ----- 第二层： Y = H * W2 -----
    dim3 block2(16, 16);
    dim3 grid2( (O      + block2.x - 1) / block2.x,
                (BATCH + block2.y - 1) / block2.y );

    // 6.1）矩阵乘法
    hipEventRecord(t_start, 0);
    matmul_kernel<<< grid2, block2 >>>(
        d_H,         // A: BATCH×H
        d_W2,        // B: H×O
        d_Y,         // C: BATCH×O
        BATCH, H, O
    );
    hipDeviceSynchronize();

    // 6.2）Bias 融合
    {
        int total_output = BATCH * O;
        int threads = 256;
        int blocks  = (total_output + threads - 1) / threads;
        add_bias_second_kernel<<< blocks, threads >>>(d_Y, BATCH, O);
        hipDeviceSynchronize();
    }
    hipEventRecord(t_stop, 0);
    hipEventSynchronize(t_stop);
    hipEventElapsedTime(&time_second_layer, t_start, t_stop);

    // 7) 拷回 Y 到 Host
    hipEventRecord(t_start, 0);
    hipMemcpy(h_Y.data(), d_Y, sizeof(double) * BATCH * O, hipMemcpyDeviceToHost);
    hipEventRecord(t_stop, 0);
    hipEventSynchronize(t_stop);
    hipEventElapsedTime(&time_total, t_start, t_stop);

    // 8) 打印耗时 & 输出
    std::cout << "===== Timing Results =====\n";
    std::cout << "First layer (GEMM + Bias+ReLU) time: " << time_first_layer << " ms\n";
    std::cout << "Second layer (GEMM + Bias)      time: " << time_second_layer << " ms\n";
    std::cout << "Host←Device memcpy (Y)          time: " << time_total << " ms\n";
    std::cout << "===========================\n\n";

    print_sample_output(h_Y);

    // 9) 释放显存与事件
    hipFree(d_X);
    hipFree(d_W1);
    hipFree(d_H);
    hipFree(d_W2);
    hipFree(d_Y);

    hipEventDestroy(t_start);
    hipEventDestroy(t_stop);

    return 0;
}
