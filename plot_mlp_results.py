import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入数据尺寸
sizes = [1024, 4096, 8192]

# 有效结果数据
# 运行时间（单位：ms）
cpu_time = [0.14133, 0.564101, 1.13315]
gpu_time = [0.649569, 0.694686, 0.698526]

# 性能（单位：GFLOP/s）
cpu_perf = [4.34727, 4.35667, 4.33764]
gpu_perf = [0.945858, 3.53771, 7.03653]

# 内存带宽（单位：GB/s）
cpu_bw = [3.2064, 3.1995, 3.18323]
gpu_bw = [0.697632, 2.59807, 5.16384]

# 设置分组柱状图位置和宽度
x = np.arange(len(sizes))
width = 0.35

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 子图1：运行时间对比
axs[0].bar(x - width/2, cpu_time, width, label='CPU', color='skyblue')
axs[0].bar(x + width/2, gpu_time, width, label='GPU', color='salmon')
axs[0].set_xticks(x)
axs[0].set_xticklabels(sizes)
axs[0].set_xlabel("输入尺寸")
axs[0].set_ylabel("运行时间 (ms)")
axs[0].set_title("运行时间对比")
axs[0].legend()

# 子图2：性能对比
axs[1].bar(x - width/2, cpu_perf, width, label='CPU', color='skyblue')
axs[1].bar(x + width/2, gpu_perf, width, label='GPU', color='salmon')
axs[1].set_xticks(x)
axs[1].set_xticklabels(sizes)
axs[1].set_xlabel("输入尺寸")
axs[1].set_ylabel("性能 (GFLOP/s)")
axs[1].set_title("性能对比")
axs[1].legend()

# 子图3：内存带宽对比
axs[2].bar(x - width/2, cpu_bw, width, label='CPU', color='skyblue')
axs[2].bar(x + width/2, gpu_bw, width, label='GPU', color='salmon')
axs[2].set_xticks(x)
axs[2].set_xticklabels(sizes)
axs[2].set_xlabel("输入尺寸")
axs[2].set_ylabel("内存带宽 (GB/s)")
axs[2].set_title("内存带宽对比")
axs[2].legend()

plt.tight_layout()
plt.show()