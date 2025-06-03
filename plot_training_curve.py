import re
import matplotlib.pyplot as plt

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
# 定义正则表达式用于匹配 Epoch 的记录行
pattern = r"\[Epoch (\d+)/\d+\] Loss = ([\d\.]+), Time = ([\d\.]+) ms"

epochs = []
losses = []
times = []

# 读取文件中的数据
with open(r"d:\MatSatMLP\mlp_dcu_result.txt", "r", encoding="utf-8") as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            time_val = float(match.group(3))
            epochs.append(epoch)
            losses.append(loss)
            times.append(time_val)

# 绘制训练曲线
fig, ax1 = plt.subplots(figsize=(10, 5))

# Loss 曲线（左侧坐标轴）
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(epochs, losses, '-o', color=color, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)

# 设置第二个 y 轴显示 Time 曲线
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Time (ms)', color=color)
ax2.plot(epochs, times, '-s', color=color, label='Time')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("MLP DCU - 训练过程曲线")
fig.tight_layout()
plt.show()