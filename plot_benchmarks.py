import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# Benchmark方法名称，包含CPU方法和GPU
methods = ["Baseline", "OpenMP", "Block Tiling", "Transpose", "SIMD", "SIMD+OpenMP", "SIMD+Block", "GPU"]

# 对应数据（单位分别为：ms、GELOP/S、GB/s）
time_elapsed = [10875.7, 1407.37, 61.8899, 299.239, 2605.42, 360.4, 386.325, 3.27396]
performance = [0.197458, 1.52588, 34.6985, 7.17649, 0.824235, 5.95861, 5.55875, 655.928]
memory_bandwidth = [0.00269962, 0.0208617, 0.474393, 0.098116, 0.0112688, 0.0814654, 0.0759985, 8.96776]

# 设置条形图位置和宽度
x = np.arange(len(methods))
width = 0.6

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 图1: 运行时间对比（使用对数坐标）
axs[0].bar(x, time_elapsed, width, color='skyblue')
axs[0].set_xticks(x)
axs[0].set_xticklabels(methods, rotation=45)
axs[0].set_xlabel("方法")
axs[0].set_ylabel("运行时间 (ms)")
axs[0].set_title("运行时间对比")
axs[0].set_yscale("log")

# 图2: 性能对比（使用对数坐标）
axs[1].bar(x, performance, width, color='salmon')
axs[1].set_xticks(x)
axs[1].set_xticklabels(methods, rotation=45)
axs[1].set_xlabel("方法")
axs[1].set_ylabel("性能 (GELOP/S)")
axs[1].set_title("性能对比")
axs[1].set_yscale("log")

# 图3: 内存带宽对比（使用对数坐标）
axs[2].bar(x, memory_bandwidth, width, color='lightgreen')
axs[2].set_xticks(x)
axs[2].set_xticklabels(methods, rotation=45)
axs[2].set_xlabel("方法")
axs[2].set_ylabel("内存带宽 (GB/s)")
axs[2].set_title("内存带宽对比")
axs[2].set_yscale("log")

plt.tight_layout()
plt.show()