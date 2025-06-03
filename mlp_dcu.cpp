#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>

// 编译：
// hipcc mlp_full_dcu_xavier.cpp -o mlp_full_dcu_xavier
// 运行：
// ./mlp_full_dcu_xavier
// 若要剖析性能，可：
// hipprof ./mlp_full_dcu_xavier

// ---------------------- 网络超参数 ----------------------
#define INPUT_DIM     10
#define HIDDEN_DIM    32
#define OUTPUT_DIM    1
#define BATCH_SIZE    256
#define EPOCHS        200
#define LEARNING_RATE 1e-4

// ---------------------- GPU 内核声明 ----------------------

// 1) 通用矩阵乘法：C = A * B
//    A: M×N, row-major; B: N×K, row-major; C: M×K, row-major
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

// 2) 对矩阵 A 的每一行加 bias（按列广播）
//    A: rows×cols, row-major; bias 长度 = cols
__global__ void add_bias_kernel(double* A, const double* bias, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;
        A[idx] += bias[col];
    }
}

// 3) ReLU 激活：A[i] = max(0, A[i])
__global__ void relu_kernel(double* A, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = (A[idx] > 0.0) ? A[idx] : 0.0;
    }
}

// 4) 计算输出层梯度 dL/dY = (pred - target) * (2.0 / batch_size)
//    pred: BATCH×1, target: BATCH×1, grad: BATCH×1
__global__ void compute_output_grad(const double* pred, const double* target, double* grad, int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch) {
        grad[idx] = 2.0 * (pred[idx] - target[idx]) / batch;
    }
}

// 5) 计算隐藏层误差：delta_hidden[i,j] = output_grad[i] * W2[j] * (H_lin[i,j] > 0 ? 1 : 0)
//    delta: BATCH×HIDDEN_DIM; output_grad: BATCH×1; W2: HIDDEN_DIM×1; H_lin: BATCH×HIDDEN_DIM
__global__ void compute_relu_backward(double* delta, 
                                      const double* output_grad, 
                                      const double* W2, 
                                      const double* H_lin,
                                      int batch, int hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_dim;
    if (idx < total) {
        int i = idx / hidden_dim;   // 样本索引
        int j = idx % hidden_dim;   // 隐单元索引
        double grad_out = output_grad[i];     // dL/dY for 样本 i
        double w2j = W2[j];                   // W2[j,0]
        double lin = H_lin[i * hidden_dim + j]; // H层线性输出
        double relu_deriv = (lin > 0.0) ? 1.0 : 0.0;
        delta[idx] = grad_out * w2j * relu_deriv;
    }
}

// 6) 计算 MSE 损失（可选，只在 host 端用得到即可，这里留空）
__global__ void compute_mse_loss(const double* pred, const double* target, double* loss, int size) {
    return;
}

// 7) SGD 更新：weight[i] -= lr * grad[i]
__global__ void sgd_update(double* weights, const double* grad, double lr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grad[idx];
    }
}

// ---------------------- 辅助函数（Host 端） ----------------------

// 从 JSON 文件加载带宽数据（假设 JSON 文件仅包含一个数值数组，如 [0.1,0.2,0.3,...]）
std::vector<double> load_json_bandwidth(const std::string& filename) {
    std::ifstream infile(filename);
    std::vector<double> data;

    if (!infile.is_open()) {
        std::cerr << "[ERROR] 无法打开文件：" << filename << std::endl;
        return data;
    }

    std::string json_string((std::istreambuf_iterator<char>(infile)),
                             std::istreambuf_iterator<char>());

    // 找到 "[" 和 "]" 并提取内容
    auto start = json_string.find('[');
    auto end = json_string.find(']');
    if (start == std::string::npos || end == std::string::npos || end <= start) {
        std::cerr << "[ERROR] JSON 格式错误：找不到 []。" << std::endl;
        return data;
    }

    std::string inner = json_string.substr(start + 1, end - start - 1);
    std::stringstream ss(inner);
    std::string item;

    while (std::getline(ss, item, ',')) {
        try {
            data.push_back(std::stod(item));
        } catch (...) {
            std::cerr << "[WARN] 忽略无效数值: " << item << std::endl;
        }
    }

    std::cout << "[INFO] 成功读取数据量: " << data.size() << std::endl;
    return data;
}

// 根据原始带宽时序 data，用滑动窗口创建 (X, y) 样本
// X: 每个样本前 INPUT_DIM 个值，y: 下一个值
void create_dataset(const std::vector<double>& data,
                    std::vector<double>& X,
                    std::vector<double>& y)
{
    int total_samples = data.size() - INPUT_DIM;
    X.resize(total_samples * INPUT_DIM);
    y.resize(total_samples * OUTPUT_DIM);
    for (int i = 0; i < total_samples; ++i) {
        // X[i,:] = data[i .. i+INPUT_DIM-1]
        for (int j = 0; j < INPUT_DIM; ++j) {
            X[i * INPUT_DIM + j] = data[i + j];
        }
        // y[i] = data[i + INPUT_DIM]
        y[i] = data[i + INPUT_DIM];
    }
}

// 最小-最大归一化到 [0,1]
void normalize_data(std::vector<double>& data, double& min_val, double& max_val) {
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());
    double range = max_val - min_val;
    for (auto& v : data) {
        v = (v - min_val) / range;
    }
}

// 反归一化
void denormalize_data(std::vector<double>& data, double min_val, double max_val) {
    double range = max_val - min_val;
    for (auto& v : data) {
        v = v * range + min_val;
    }
}

// 计算在 Host 端的 MSE 损失（仅用于打印）
double compute_mse_host(const double* pred, const double* target, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        double d = pred[i] - target[i];
        sum += d * d;
    }
    return sum / size;
}

// ---------------------- 主函数 ----------------------
int main() {
    std::cout << "[DEBUG] 程序启动" << std::endl;
    // 1) 读取原始带宽数据（假设文件名为 "starlink_bw.json"）
    std::vector<double> raw_data = load_json_bandwidth("starlink_bw.json");
    if (raw_data.empty()) {
        std::cout << "[DEBUG] 加载数据失败, data.size() = " << raw_data.size() << std::endl;
        return 1;
    }
    std::cout << "[DEBUG] 加载数据完毕, data.size() = " << raw_data.size() << std::endl;

    // 2) 归一化整个原始数据
    double min_val, max_val;
    normalize_data(raw_data, min_val, max_val);

    // 3) 创建数据集 X_total, y_total
    std::vector<double> X_total, y_total;
    create_dataset(raw_data, X_total, y_total);
    int total_samples = static_cast<int>(y_total.size()); // = raw_data.size() - INPUT_DIM

    // 4) 将数据拆分成训练集和测试集 (例如 前 80% 训练，后 20% 测试)
    int train_samples = static_cast<int>(0.8 * total_samples);
    int test_samples  = total_samples - train_samples;

    std::vector<double> X_train(X_total.begin(), X_total.begin() + train_samples * INPUT_DIM);
    std::vector<double> y_train(y_total.begin(), y_total.begin() + train_samples);
    std::vector<double> X_test (X_total.begin() + train_samples * INPUT_DIM, X_total.end());
    std::vector<double> y_test (y_total.begin() + train_samples, y_total.end());

    // 5) 在设备端一次性分配所有必要缓冲区（避免多次 hipMalloc 开销）
    //    - X_batch: BATCH_SIZE×INPUT_DIM
    //    - H_lin, H_act: BATCH_SIZE×HIDDEN_DIM
    //    - Y_pred, y_true, output_grad: BATCH_SIZE×OUTPUT_DIM
    //    - delta_hidden: BATCH_SIZE×HIDDEN_DIM
    //    - 权重与梯度：
    //       W1: INPUT_DIM×HIDDEN_DIM, b1: HIDDEN_DIM
    //       W2: HIDDEN_DIM×OUTPUT_DIM, b2: OUTPUT_DIM
    //       grad_W1: 同 W1 尺寸, grad_b1: HIDDEN_DIM
    //       grad_W2: 同 W2 尺寸, grad_b2: OUTPUT_DIM
    int train_batches = (train_samples + BATCH_SIZE - 1) / BATCH_SIZE;

    double *d_X_batch, *d_H_lin, *d_H_act, *d_Y_pred, *d_y_true, *d_output_grad, *d_delta_hidden;
    double *d_W1, *d_b1, *d_W2, *d_b2;
    double *d_grad_W1, *d_grad_b1, *d_grad_W2, *d_grad_b2;

    hipMalloc(&d_X_batch,     sizeof(double) * BATCH_SIZE * INPUT_DIM);
    hipMalloc(&d_H_lin,       sizeof(double) * BATCH_SIZE * HIDDEN_DIM);
    hipMalloc(&d_H_act,       sizeof(double) * BATCH_SIZE * HIDDEN_DIM);
    hipMalloc(&d_Y_pred,      sizeof(double) * BATCH_SIZE * OUTPUT_DIM);
    hipMalloc(&d_y_true,      sizeof(double) * BATCH_SIZE * OUTPUT_DIM);
    hipMalloc(&d_output_grad, sizeof(double) * BATCH_SIZE * OUTPUT_DIM);
    hipMalloc(&d_delta_hidden,sizeof(double) * BATCH_SIZE * HIDDEN_DIM);

    hipMalloc(&d_W1, sizeof(double) * INPUT_DIM * HIDDEN_DIM);
    hipMalloc(&d_b1, sizeof(double) * HIDDEN_DIM);
    hipMalloc(&d_W2, sizeof(double) * HIDDEN_DIM * OUTPUT_DIM);
    hipMalloc(&d_b2, sizeof(double) * OUTPUT_DIM);

    hipMalloc(&d_grad_W1, sizeof(double) * INPUT_DIM * HIDDEN_DIM);
    hipMalloc(&d_grad_b1, sizeof(double) * HIDDEN_DIM);
    hipMalloc(&d_grad_W2, sizeof(double) * HIDDEN_DIM * OUTPUT_DIM);
    hipMalloc(&d_grad_b2, sizeof(double) * OUTPUT_DIM);

    // 6) 在主机端使用 Xavier 初始化权重与偏置
    std::vector<double> h_W1(INPUT_DIM * HIDDEN_DIM),
                        h_b1(HIDDEN_DIM),
                        h_W2(HIDDEN_DIM * OUTPUT_DIM),
                        h_b2(OUTPUT_DIM);
    srand(123);
    // ----- Xavier 初始化 -----
    // W1: 输入=INPUT_DIM, 输出=HIDDEN_DIM
    double limit1 = std::sqrt(6.0 / (INPUT_DIM + HIDDEN_DIM));
    for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; ++i) {
        double u = rand() / (double)RAND_MAX;       // [0,1)
        h_W1[i] = (u * 2.0 - 1.0) * limit1;         // 映射到 [-limit1, +limit1]
    }
    for (int j = 0; j < HIDDEN_DIM; ++j) {
        h_b1[j] = 0.0;
    }
    // W2: 输入=HIDDEN_DIM, 输出=OUTPUT_DIM
    double limit2 = std::sqrt(6.0 / (HIDDEN_DIM + OUTPUT_DIM));
    for (int i = 0; i < HIDDEN_DIM * OUTPUT_DIM; ++i) {
        double u = rand() / (double)RAND_MAX;
        h_W2[i] = (u * 2.0 - 1.0) * limit2;
    }
    for (int j = 0; j < OUTPUT_DIM; ++j) {
        h_b2[j] = 0.0;
    }
    // ---------------------------

    // 拷到设备
    hipMemcpy(d_W1, h_W1.data(), sizeof(double) * INPUT_DIM * HIDDEN_DIM, hipMemcpyHostToDevice);
    hipMemcpy(d_b1, h_b1.data(), sizeof(double) * HIDDEN_DIM, hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), sizeof(double) * HIDDEN_DIM * OUTPUT_DIM, hipMemcpyHostToDevice);
    hipMemcpy(d_b2, h_b2.data(), sizeof(double) * OUTPUT_DIM, hipMemcpyHostToDevice);

    // 7) 训练主循环
    std::cout << "[INFO] 开始训练，共 " << train_batches << " 个批次，每批大小 " << BATCH_SIZE
              << "。总样本数 = " << train_samples << std::endl;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double epoch_loss = 0.0;
        auto t_start = std::chrono::high_resolution_clock::now();

        for (int b = 0; b < train_batches; ++b) {
            int offset = b * BATCH_SIZE;
            int current_batch_size = std::min(BATCH_SIZE, train_samples - offset);

            // 7.1) 拷贝当前批次的 X 和 y 到设备，如果 batch 不满 BATCH_SIZE，可以用零填充 tail
            std::vector<double> temp_X(BATCH_SIZE * INPUT_DIM, 0.0);
            std::vector<double> temp_y(BATCH_SIZE * OUTPUT_DIM, 0.0);
            for (int i = 0; i < current_batch_size; ++i) {
                std::copy(
                    X_train.begin() + (offset + i) * INPUT_DIM,
                    X_train.begin() + (offset + i) * INPUT_DIM + INPUT_DIM,
                    temp_X.begin() + i * INPUT_DIM
                );
                temp_y[i] = y_train[offset + i];
            }
            hipMemcpy(d_X_batch, temp_X.data(), sizeof(double) * BATCH_SIZE * INPUT_DIM, hipMemcpyHostToDevice);
            hipMemcpy(d_y_true, temp_y.data(), sizeof(double) * BATCH_SIZE * OUTPUT_DIM, hipMemcpyHostToDevice);

            // 7.2) 前向传播——隐藏层线性变换: H_lin = X_batch * W1
            dim3 block1(16, 16);
            dim3 grid1(
                (HIDDEN_DIM + block1.x - 1) / block1.x,
                (BATCH_SIZE + block1.y - 1) / block1.y
            );
            hipLaunchKernelGGL(
                matmul_kernel,
                grid1, block1, 0, 0,
                d_X_batch, d_W1, d_H_lin,
                BATCH_SIZE, INPUT_DIM, HIDDEN_DIM
            );
            hipDeviceSynchronize();

            // 7.3) 隐藏层加偏置 + ReLU：H_act = ReLU(H_lin + b1)
            {
                int total_hidden = BATCH_SIZE * HIDDEN_DIM;
                int threads = 256;
                int blocks  = (total_hidden + threads - 1) / threads;
                hipLaunchKernelGGL(
                    add_bias_kernel,
                    dim3(blocks), dim3(threads), 0, 0,
                    d_H_lin, d_b1, BATCH_SIZE, HIDDEN_DIM
                );
                hipDeviceSynchronize();
                hipLaunchKernelGGL(
                    relu_kernel,
                    dim3(blocks), dim3(threads), 0, 0,
                    d_H_lin, total_hidden
                );
                hipDeviceSynchronize();
                // 将 ReLU 结果留在 d_H_lin，即 H_act
                hipMemcpy(d_H_act, d_H_lin, sizeof(double) * BATCH_SIZE * HIDDEN_DIM, hipMemcpyDeviceToDevice);
            }

            // 7.4) 输出层：Y_pred = H_act * W2
            dim3 block2(16, 16);
            dim3 grid2(
                (OUTPUT_DIM + block2.x - 1) / block2.x,
                (BATCH_SIZE + block2.y - 1) / block2.y
            );
            hipLaunchKernelGGL(
                matmul_kernel,
                grid2, block2, 0, 0,
                d_H_act, d_W2, d_Y_pred,
                BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM
            );
            hipDeviceSynchronize();

            // 7.5) 输出层加偏置：Y_pred += b2
            {
                int total_out = BATCH_SIZE * OUTPUT_DIM;
                int threads = 256;
                int blocks  = (total_out + threads - 1) / threads;
                hipLaunchKernelGGL(
                    add_bias_kernel,
                    dim3(blocks), dim3(threads), 0, 0,
                    d_Y_pred, d_b2, BATCH_SIZE, OUTPUT_DIM
                );
                hipDeviceSynchronize();
            }

            // 7.6) 计算输出梯度：dL/dY = 2*(Y_pred - y_true)/batch
            {
                int threads = 256;
                int blocks  = (BATCH_SIZE * OUTPUT_DIM + threads - 1) / threads;
                hipLaunchKernelGGL(
                    compute_output_grad,
                    dim3(blocks), dim3(threads), 0, 0,
                    d_Y_pred, d_y_true, d_output_grad, BATCH_SIZE
                );
                hipDeviceSynchronize();
            }

            // 7.7) 计算隐藏层误差：delta_hidden = dL/dY * W2^T .* ReLU'(H_lin)
            {
                int total_hidden = BATCH_SIZE * HIDDEN_DIM;
                int threads = 256;
                int blocks  = (total_hidden + threads - 1) / threads;
                hipLaunchKernelGGL(
                    compute_relu_backward,
                    dim3(blocks), dim3(threads), 0, 0,
                    d_delta_hidden, d_output_grad, d_W2, d_H_lin,
                    BATCH_SIZE, HIDDEN_DIM
                );
                hipDeviceSynchronize();
            }

            // 7.8) 计算 grad_W2 & grad_b2（在 GPU 上做 reduction 或者简单 CPU 归约后拷回）
            {
                // 这里仍然用 CPU 做 reduction，拷回后再上载回 GPU
                std::vector<double> temp_Hact(BATCH_SIZE * HIDDEN_DIM);
                std::vector<double> temp_ograd(BATCH_SIZE * OUTPUT_DIM);
                std::vector<double> h_gradW2(HIDDEN_DIM, 0.0);
                std::vector<double> h_gradb2(OUTPUT_DIM, 0.0);

                hipMemcpy(temp_Hact.data(), d_H_act, sizeof(double) * BATCH_SIZE * HIDDEN_DIM, hipMemcpyDeviceToHost);
                hipMemcpy(temp_ograd.data(), d_output_grad, sizeof(double) * BATCH_SIZE * OUTPUT_DIM, hipMemcpyDeviceToHost);

                // grad_W2[j] = sum_{i<batch} H_act[i,j] * output_grad[i]
                for (int j = 0; j < HIDDEN_DIM; ++j) {
                    double sum_w2 = 0.0;
                    for (int i = 0; i < BATCH_SIZE; ++i) {
                        sum_w2 += temp_Hact[i * HIDDEN_DIM + j] * temp_ograd[i];
                    }
                    h_gradW2[j] = sum_w2;
                }
                // grad_b2 = sum(output_grad)
                double sum_b2 = 0.0;
                for (int i = 0; i < BATCH_SIZE; ++i) {
                    sum_b2 += temp_ograd[i];
                }
                h_gradb2[0] = sum_b2;

                hipMemcpy(d_grad_W2, h_gradW2.data(), sizeof(double) * HIDDEN_DIM * OUTPUT_DIM, hipMemcpyHostToDevice);
                hipMemcpy(d_grad_b2, h_gradb2.data(), sizeof(double) * OUTPUT_DIM, hipMemcpyHostToDevice);
            }

            // 7.9) 计算 grad_b1（grad_b1[j] = sum_i delta_hidden[i,j]）
            {
                std::vector<double> temp_dhid(BATCH_SIZE * HIDDEN_DIM);
                std::vector<double> h_gradb1(HIDDEN_DIM, 0.0);
                hipMemcpy(temp_dhid.data(), d_delta_hidden, sizeof(double) * BATCH_SIZE * HIDDEN_DIM, hipMemcpyDeviceToHost);
                for (int j = 0; j < HIDDEN_DIM; ++j) {
                    double s = 0.0;
                    for (int i = 0; i < BATCH_SIZE; ++i) {
                        s += temp_dhid[i * HIDDEN_DIM + j];
                    }
                    h_gradb1[j] = s;
                }
                hipMemcpy(d_grad_b1, h_gradb1.data(), sizeof(double) * HIDDEN_DIM, hipMemcpyHostToDevice);
            }

            // 7.10) 计算 grad_W1： grad_W1[k,j] = sum_{i<batch} X_batch[i,k] * delta_hidden[i,j]
            {
                std::vector<double> temp_Xb(BATCH_SIZE * INPUT_DIM);
                std::vector<double> temp_dhid(BATCH_SIZE * HIDDEN_DIM);
                std::vector<double> h_gradW1(INPUT_DIM * HIDDEN_DIM, 0.0);

                hipMemcpy(temp_Xb.data(), d_X_batch,    sizeof(double) * BATCH_SIZE * INPUT_DIM, hipMemcpyDeviceToHost);
                hipMemcpy(temp_dhid.data(), d_delta_hidden, sizeof(double) * BATCH_SIZE * HIDDEN_DIM, hipMemcpyDeviceToHost);

                for (int k = 0; k < INPUT_DIM; ++k) {
                    for (int j = 0; j < HIDDEN_DIM; ++j) {
                        double sum_w1 = 0.0;
                        for (int i = 0; i < BATCH_SIZE; ++i) {
                            sum_w1 += temp_Xb[i * INPUT_DIM + k] * temp_dhid[i * HIDDEN_DIM + j];
                        }
                        h_gradW1[k * HIDDEN_DIM + j] = sum_w1;
                    }
                }
                hipMemcpy(d_grad_W1, h_gradW1.data(), sizeof(double) * INPUT_DIM * HIDDEN_DIM, hipMemcpyHostToDevice);
            }

            // 7.11) 在 GPU 上用 SGD 更新 W2、b2、W1、b1
            {
                int threads = 256;
                int blocks_W2 = (HIDDEN_DIM * OUTPUT_DIM + threads - 1) / threads;
                int blocks_b2 = (OUTPUT_DIM + threads - 1) / threads;
                int blocks_W1 = (INPUT_DIM * HIDDEN_DIM + threads - 1) / threads;
                int blocks_b1 = (HIDDEN_DIM + threads - 1) / threads;

                hipLaunchKernelGGL(
                    sgd_update,
                    dim3(blocks_W2), dim3(threads), 0, 0,
                    d_W2, d_grad_W2, LEARNING_RATE, HIDDEN_DIM * OUTPUT_DIM
                );
                hipDeviceSynchronize();
                hipLaunchKernelGGL(
                    sgd_update,
                    dim3(blocks_b2), dim3(threads), 0, 0,
                    d_b2, d_grad_b2, LEARNING_RATE, OUTPUT_DIM
                );
                hipDeviceSynchronize();
                hipLaunchKernelGGL(
                    sgd_update,
                    dim3(blocks_W1), dim3(threads), 0, 0,
                    d_W1, d_grad_W1, LEARNING_RATE, INPUT_DIM * HIDDEN_DIM
                );
                hipDeviceSynchronize();
                hipLaunchKernelGGL(
                    sgd_update,
                    dim3(blocks_b1), dim3(threads), 0, 0,
                    d_b1, d_grad_b1, LEARNING_RATE, HIDDEN_DIM
                );
                hipDeviceSynchronize();
            }

            // 7.12) 计算并累加当前批次的 MSE（仅 Host 端，使用 normalized 值）
            {
                std::vector<double> temp_pred(BATCH_SIZE * OUTPUT_DIM);
                std::vector<double> temp_true(BATCH_SIZE * OUTPUT_DIM);
                hipMemcpy(temp_pred.data(), d_Y_pred, sizeof(double) * BATCH_SIZE * OUTPUT_DIM, hipMemcpyDeviceToHost);
                hipMemcpy(temp_true.data(), d_y_true, sizeof(double) * BATCH_SIZE * OUTPUT_DIM, hipMemcpyDeviceToHost);
                epoch_loss += compute_mse_host(temp_pred.data(), temp_true.data(), current_batch_size);
            }
        } // end for each batch

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "[Epoch " << (epoch + 1) << "/" << EPOCHS << "] "
                  << "Loss = " << (epoch_loss / train_batches) 
                  << ", Time = " << elapsed_ms << " ms\n";
    }

    // 8) 训练结束后，在测试集上做推理并计算 normalized MSE，同时收集所有预测值以便后续反归一化
    int test_batches = (test_samples + BATCH_SIZE - 1) / BATCH_SIZE;
    double total_test_loss = 0.0;

    // 用于存放所有测试样本的 normalized 预测值和真实值
    std::vector<double> h_preds_norm(test_samples, 0.0);
    std::vector<double> h_truth_norm(test_samples, 0.0);

    for (int b = 0; b < test_batches; ++b) {
        int curr_bs = std::min(BATCH_SIZE, test_samples - b * BATCH_SIZE);

        // 准备本批次输入
        std::vector<double> temp_X(BATCH_SIZE * INPUT_DIM, 0.0);
        for (int i = 0; i < curr_bs; ++i) {
            std::copy(
                X_test.begin() + (b * BATCH_SIZE + i) * INPUT_DIM,
                X_test.begin() + (b * BATCH_SIZE + i) * INPUT_DIM + INPUT_DIM,
                temp_X.begin() + i * INPUT_DIM
            );
        }
        hipMemcpy(d_X_batch, temp_X.data(), sizeof(double) * BATCH_SIZE * INPUT_DIM, hipMemcpyHostToDevice);

        // 前向：隐藏层
        dim3 block1(16, 16), grid1((HIDDEN_DIM + 15)/16, (BATCH_SIZE + 15)/16);
        hipLaunchKernelGGL(matmul_kernel, grid1, block1, 0, 0, d_X_batch, d_W1, d_H_lin, BATCH_SIZE, INPUT_DIM, HIDDEN_DIM);
        hipDeviceSynchronize();
        {
            int total_hidden = BATCH_SIZE * HIDDEN_DIM;
            int threads = 256;
            int blocks  = (total_hidden + threads - 1) / threads;
            hipLaunchKernelGGL(add_bias_kernel, dim3(blocks), dim3(threads), 0, 0, d_H_lin, d_b1, BATCH_SIZE, HIDDEN_DIM);
            hipDeviceSynchronize();
            hipLaunchKernelGGL(relu_kernel, dim3(blocks), dim3(threads), 0, 0, d_H_lin, total_hidden);
            hipDeviceSynchronize();
            hipMemcpy(d_H_act, d_H_lin, sizeof(double) * BATCH_SIZE * HIDDEN_DIM, hipMemcpyDeviceToDevice);
        }
        // 前向：输出层
        dim3 block2(16, 16), grid2((OUTPUT_DIM + 15)/16, (BATCH_SIZE + 15)/16);
        hipLaunchKernelGGL(matmul_kernel, grid2, block2, 0, 0, d_H_act, d_W2, d_Y_pred, BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM);
        hipDeviceSynchronize();
        {
            int total_out = BATCH_SIZE * OUTPUT_DIM;
            int threads = 256;
            int blocks  = (total_out + threads - 1) / threads;
            hipLaunchKernelGGL(add_bias_kernel, dim3(blocks), dim3(threads), 0, 0, d_Y_pred, d_b2, BATCH_SIZE, OUTPUT_DIM);
            hipDeviceSynchronize();
        }

        // 拷回本批次预测值
        std::vector<double> temp_pred(BATCH_SIZE * OUTPUT_DIM);
        hipMemcpy(temp_pred.data(), d_Y_pred, sizeof(double) * BATCH_SIZE * OUTPUT_DIM, hipMemcpyDeviceToHost);

        // 真实值从 y_test 里取（normalized）
        for (int i = 0; i < curr_bs; ++i) {
            h_preds_norm[b * BATCH_SIZE + i]  = temp_pred[i];
            h_truth_norm[b * BATCH_SIZE + i] = y_test[b * BATCH_SIZE + i];
        }

        // 计算本批次 normalized MSE
        total_test_loss += compute_mse_host(temp_pred.data(), &y_test[b * BATCH_SIZE], curr_bs);
    }
    double test_mse_norm = total_test_loss / test_batches;
    std::cout << "[INFO] 测试集（normalized） MSE = " << test_mse_norm << std::endl;

    // 9) 计算并打印反归一化后的 MSE
    std::vector<double> h_preds_denorm = h_preds_norm;
    std::vector<double> h_truth_denorm = h_truth_norm;
    denormalize_data(h_preds_denorm, min_val, max_val);
    denormalize_data(h_truth_denorm, min_val, max_val);

    double sum_sq_err = 0.0;
    for (int i = 0; i < test_samples; ++i) {
        double diff = h_preds_denorm[i] - h_truth_denorm[i];
        sum_sq_err += diff * diff;
    }
    double test_mse_denorm = sum_sq_err / test_samples;
    std::cout << "[INFO] 测试集（denormalized） MSE = " << test_mse_denorm << std::endl;

    // 打印前 10 个样本（可选）
    std::cout << "[Sample Predictions vs True] (前 10 个测试样本，denormalized)\n";
    for (int i = 0; i < std::min(test_samples, 10); ++i) {
        std::cout << "  Pred[" << i << "] = " << h_preds_denorm[i]
                  << ", True[" << i << "] = " << h_truth_denorm[i] << "\n";
    }

    // 10) 释放所有 GPU 内存
    hipFree(d_X_batch);
    hipFree(d_H_lin);
    hipFree(d_H_act);
    hipFree(d_Y_pred);
    hipFree(d_y_true);
    hipFree(d_output_grad);
    hipFree(d_delta_hidden);
    hipFree(d_W1);
    hipFree(d_b1);
    hipFree(d_W2);
    hipFree(d_b2);
    hipFree(d_grad_W1);
    hipFree(d_grad_b1);
    hipFree(d_grad_W2);
    hipFree(d_grad_b2);

    return 0;
}
