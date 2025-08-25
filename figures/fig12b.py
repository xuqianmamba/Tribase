import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import re

# 设置全局字体为Type 1字体
mpl.rcParams['pdf.fonttype'] = 42  # 设置为Type 1字体
mpl.rcParams['ps.fonttype'] = 42   # 设置为Type 1字体

df = pd.read_csv('logs/build_ratio-recall-qps-tribase.csv')
with open("logs/build_ratio-recall-qps-tribase.log") as f:
    text = f.read()
    add_elapseds = re.findall(r'^add elapsed: ([\d.]+)s', text, re.MULTILINE)
    add_elapseds = np.array([float(x) for x in add_elapseds])

query_time = df["query_time"].values

query_time = query_time.reshape(-1, 10).mean(axis=0)
add_elapseds = add_elapseds.reshape(-1, 10).mean(axis=0)


# 设置全局字体大小
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 22})

# 定义数据点
# build_speedup = [1, 1.06, 1.1, 1.16, 1.26, 1.36, 1.43, 1.63, 1.8, 1.96]
# query_time = [100, 99.95, 99.92, 99.8, 99.6, 99.3, 99.1, 98.5, 97.8, 96]
build_speedup = add_elapseds / add_elapseds[-1]
query_speedup = query_time / query_time[-1] * 100
coeffs = np.polyfit(build_speedup, query_speedup, 2)
a, b, c = coeffs
build_smooth = np.linspace(build_speedup.min(), build_speedup.max(), 10)
query_smooth = a * build_smooth**2 + b * build_smooth + c

# 创建图形
plt.figure(figsize=(7, 4.5))
plt.plot(build_smooth, query_smooth, marker='o', linestyle='-', color='blue', markersize=8)
plt.plot(build_speedup, query_speedup, marker='o', linestyle='-.', color='red', markersize=8)

# 设置图形的标题和坐标轴标签
plt.title('Query Speed vs Build Speedup', fontsize=24, fontweight='normal', color='black')
plt.xlabel('Build Speedup', fontsize=24, fontweight='normal', color='black')
plt.ylabel('Query Speed (%)', fontsize=24, fontweight='normal', color='black')

# 设置x轴和y轴的刻度
# plt.xticks([1, 1.2, 1.4, 1.6, 1.8, 2], fontsize=24, fontweight='normal', color='black')
# plt.yticks([96, 97, 98, 99, 100], fontsize=24, fontweight='normal', color='black')

# 设置y轴的显示范围和x轴的显示范围，留有一定空间
# plt.ylim(95.9, 100.1)
# plt.xlim(0.95, 2.1)

# 设置坐标轴颜色和粗细
plt.gca().spines['top'].set_color('black')
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_color('black')
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(1.5)

# 设置坐标轴刻度线的颜色和粗细
plt.gca().tick_params(colors='black', width=1.5)

# 不显示网格
plt.grid(False)

# 调整布局
plt.subplots_adjust(bottom=0.17,left=0.2)  # 整个图表向下移动


plt.tight_layout()
# 保存图形
plt.savefig('figures/fig12b.png', dpi=300)