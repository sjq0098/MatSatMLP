import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体    
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
# 指标名称
metrics = ["First layer\n(GEMM+Bias+ReLU)", "Second layer\n(GEMM+Bias)", "Memcpy / Total forward"]

# 从 mlp_forward_level.txt 提取的有效数据（Test1）
test1 = [0.146713, 0.078877, 0.036958]

# 从 mlp_forward_level1.txt 提取的有效数据（Test2）
test2 = [0.743805, 0.056958, 0.100316]

# 设置柱状图位置
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制两组柱状图
bars2 = ax.bar(x - width/2, test1, width, label="融核后", color='skyblue')
bars1 = ax.bar(x + width/2, test2, width, label="融核前", color='salmon')

# 添加坐标轴标签、标题等
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("时间 (ms)")
ax.set_title("MLP Forward Timing Comparison")
ax.legend()

# 在柱形上标注具体数值
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()