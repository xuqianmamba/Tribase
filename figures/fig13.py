import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib as mpl
from utils import name_map
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tribase_log",
    type=str,
    default="./logs/edge-recall-qps-tribase.csv",
    help="Path to the Tribase log CSV file",
)
parser.add_argument(
    "--release_tribase_log",
    type=str,
    default="./logs/release-recall-qps-tribase.csv",
    help="Path to the Tribase log CSV file",
)
args = parser.parse_args()

df = pd.read_csv(args.tribase_log)
df_release = pd.read_csv(args.release_tribase_log)

# --- 全局设置 ---
# 设置全局字体为Type 1字体，确保PDF嵌入字体
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# --- 数据定义 (结构化) ---
# 将每个数据集定义为一个字典，方便管理和扩展
datasets = [
    {
        "label": "Faiss-IVF",
        "time": df_release[(df_release["dataset"] == "msong") & (df_release["opt_level"] == 0)]["query_time"].values,
        "disable_time": df[(df["dataset"] == "msong") & (df["opt_level"] == 0)]["query_time"].values,
        "recall": df_release[(df_release["dataset"] == "msong") & (df_release["opt_level"] == 0)]["recall"].values,
        "is_recall_percentage": False, # 召回率是小数形式
        "color": "green",
        "marker": "o",
        "zorder": 2,
    },
    {
        "label": "Tribase-Triangle",
        "time": df_release[(df_release["dataset"] == "msong") & (df_release["opt_level"] == 1)]["query_time"].values,
        "disable_time": df[(df["dataset"] == "msong") & (df["opt_level"] == 1)]["query_time"].values,
        "recall": df_release[(df_release["dataset"] == "msong") & (df_release["opt_level"] == 1)]["recall"].values,
        "is_recall_percentage": False,
        "color": "red",
        "marker": "d",
        "zorder": 1,
    },
    {
        "label": "ADS-IVF++",
        "time": [373.419, 230.475, 188.32, 151.619, 124.293, 99.985, 80.0817, 80.41, 64.6335, 40.8908, 32.7068],
        "disable_time": [373.419, 230.475, 188.32, 151.619, 124.293, 99.985, 80.0817, 80.41, 64.6335, 40.8908, 32.7068],
        "recall": [99.9892, 99.9863, 99.9754, 99.9332, 99.8233, 99.5894, 99.117, 99.117, 98.3303, 95.3133, 92.8234],
        "is_recall_percentage": True, # 召回率是百分比形式
        "color": "blue",
        "marker": "s",
        "zorder": 4,
    },
    {
        "label": "ADS-IVF+",
        "time": [447.84, 268.138, 219.536, 179.02, 144.19, 115.38, 92.6037, 73.2841, 47.0449, 37.5506],
        "disable_time": [447.84, 268.138, 219.536, 179.02, 144.19, 115.38, 92.6037, 73.2841, 47.0449, 37.5506],
        "recall": [99.9892, 99.9863, 99.9754, 99.9332, 99.8233, 99.5894, 99.117, 98.3303, 95.3133, 92.8234],
        "is_recall_percentage": True,
        "color": "purple",
        "marker": "x",
        "zorder": 3,
    }
]

# --- 数据处理 (循环) ---
epsilon = 1e-10
for d in datasets:
    time_np = np.array(d["time"])
    disable_time_np = np.array(d["disable_time"])
    recall_np = np.array(d["recall"])
    
    # 统一转换召回率为 "Loss" (1 - Recall)，并处理小数/百分比两种情况
    if d["is_recall_percentage"]:
        recall_np /= 100.0
    d["loss"] = np.clip(1 - recall_np, epsilon, 1 - epsilon)
    
    # 计算 QPS
    d["qps"] = 10000 / time_np
    d["qps_disable"] = 10000 / disable_time_np

# --- 绘图 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
subplot_titles = ['Performance enable SIMD', 'Performance disable SIMD']

# 循环绘制每个数据集
for d in datasets:
    if "ADS" in d["label"]:
        continue
    # 左子图：原始数据
    ax1.plot(d["loss"], d["qps"], marker=d["marker"], linestyle='--', label=d["label"], 
             color=d["color"], markerfacecolor=d["color"], zorder=d["zorder"])
    
    # 右子图：修改后的数据
    ax2.plot(d["loss"], d["qps_disable"], marker=d["marker"], linestyle='-', label=d["label"], 
             color=d["color"], markerfacecolor=d["color"], zorder=d["zorder"])

# --- 坐标轴和图例设置 (循环) ---
x_values = [1.1e-5, 10**-4, 10**-3, 10**-2, 10**-1]

for i, ax in enumerate((ax1, ax2)):
    ax.set_title(subplot_titles[i], fontsize=17)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Loss', fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray')
    
    # 设置刻度格式
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    
    ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray')
    ax.yaxis.grid(False, which='minor')
    ax.tick_params(which='minor', length=4, color='black')
    
    # 反转横坐标
    ax.set_xlim(x_values[-1], x_values[0])
    
    # 添加图例
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.07), ncol=2, fontsize=17)

# --- 定制化和保存 ---
# 仅为左子图设置Y轴标签
ax1.set_ylabel('Query Per Second (QPS)', fontsize=17, labelpad=15)

# 调整布局以防止标签重叠
plt.tight_layout()

# 保存为多种格式
plt.savefig('figures/fig13.png', dpi=300, bbox_inches='tight')