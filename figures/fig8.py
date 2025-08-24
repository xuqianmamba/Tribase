import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import argparse
import pandas as pd
from utils import name_map

parser = argparse.ArgumentParser()
parser.add_argument('--faiss_log', type=str, default='./logs/release-recall-qps-faiss.csv', help='Path to the Faiss log CSV file')
parser.add_argument('--tribase_log', type=str, default='./logs/edge-recall-qps-tribase.csv', help='Path to the Tribase log CSV file')
parser.add_argument('--dataset', type=str, nargs='+', default=["fasion_mnist_784", "nuswide", "msong", "sift1m", "glove25"], help='List of datasets to plot')
args = parser.parse_args()

# , "HandOutlines", "StarLightCurves"

df = pd.read_csv(args.tribase_log)

# 设置全局字体为Type 1字体
mpl.rcParams['pdf.fonttype'] = 42  # 设置为Type 1字体
mpl.rcParams['ps.fonttype'] = 42   # 设置为Type 1字体

datasets = args.dataset

# 创建图形和轴，更新为2行5列
fig, axs = plt.subplots(1, len(datasets), figsize=(3 + 5 *len(datasets), 4))

# 数据集标签和标记
labels = ['Faiss-IVF', 'Tribase-Triangle', 'Tribase-SubNNL2', 'Tribase-SubNNIP', 'Tribase-All']
markers = ['o--', 's--', 'd--', '*--', 'x--']
colors = ['darkgreen', 'darkorange', 'darkblue', 'red', 'purple']

yticks = [10**3, 10**4]
yticklabels = ['10^3', '10^4']
markers = ['o--', 's--', 'd--', '*--', 'x--']
colors = ['darkgreen', 'darkorange', 'darkblue', 'red', 'purple']
labels = ['Faiss-IVF', 'Tribase-Triangle', 'Tribase-SubNNL2', 'Tribase-SubNNIP', 'Tribase-All']

# 定义各个数据集的x轴标签
xticks = {
    "Fashion": [1.0000, 1.0002, 1.0004],
    "Nuswide": [1.000, 1.002, 1.004],
    "Msong": [1.0000, 1.00025, 1.00050, 1.00075],
    "Sift1m": [1.000, 1.002, 1.004, 1.006],
    "Glove25": [1.0000, 1.0002, 1.0004],
    "Hand Outlines": [1.0000, 1.00025, 1.00050, 1.00075],
    "Star Light Curves": [1.0000, 1.00025, 1.00050, 1.00075]
}

for i, dataset in enumerate(datasets):
    for opt_level, color, label, marker in zip([0, 1, 3, 5, 7], colors, labels, markers):
        distances = df[(df["dataset"] == dataset) & (df["opt_level"] == opt_level)]["r2"].values + 1
        qps = df[(df["dataset"] == dataset) & (df["opt_level"] == opt_level)]["qps"].values
        axs[i].plot(distances, qps, marker, color=color, label=label,markersize=6, alpha=0.6, linewidth=0.4)  # 调整透明度和线宽

    dataset_name = name_map(dataset)
    axs[i].set_title(f'{dataset_name}', fontsize=22)
    axs[i].set_xlabel('Average Distance Ratio', fontsize=20)
    axs[i].set_yscale('log')
    axs[i].grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray')
    axs[i].tick_params(axis='x', rotation=-30, labelsize=22)
    axs[i].tick_params(axis='y', labelsize=22)
    axs[i].set_xlim(1, xticks[dataset_name][-1])

    if i == 0:
        axs[i].set_ylabel('Query per Second (QPS)', fontsize=22, labelpad=-10)

    # if i == 4:
    #     axs[i].set_yticks([10000])  # 只设置一个刻度

    # 设置 x 轴标签为 1.**** 格式
    axs[i].set_xticks(xticks[dataset_name])
    axs[i].set_xticklabels([f'{tick:.4f}' for tick in xticks[dataset_name]])

# 创建单一全局图例
fig.legend(labels, loc='upper center', ncol=len(labels), fontsize=20, bbox_to_anchor=(0.5, 0.999))

# 调整布局
plt.tight_layout()

plt.subplots_adjust(top=0.75) 
# 显示图形
plt.savefig('figures/fig8.png', dpi=300, bbox_inches='tight')

plt.show()