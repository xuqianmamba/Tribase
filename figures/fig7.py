import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib as mpl
from utils import name_map
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--faiss_log', type=str, default='./logs/release-recall-qps-faiss.csv', help='Path to the Faiss log CSV file')
parser.add_argument('--tribase_log', type=str, default='./logs/edge-recall-qps-tribase.csv', help='Path to the Tribase log CSV file')
parser.add_argument('--dataset', type=str, nargs='+', default=["fasion_mnist_784", "nuswide", "msong", "sift1m", "glove25", "HandOutlines", "StarLightCurves"], help='List of datasets to plot')
args = parser.parse_args()

df = pd.read_csv(args.tribase_log)
# faiss_df = pd.read_csv(args.faiss_log)

# 设置全局字体为Type 1字体
mpl.rcParams['pdf.fonttype'] = 42  # 设置为Type 1字体
mpl.rcParams['ps.fonttype'] = 42   # 设置为Type 1字体


plt.rcParams['font.size'] = 22  # 调整字体大小
plt.rcParams['font.family'] = 'Arial'

datasets = args.dataset

# 创建图形和轴，更新为2行5列
fig, axs = plt.subplots(1, len(datasets), figsize=(28, 4))

# 数据集标签和标记
labels = ['Faiss-IVF', 'Tribase-Triangle', 'Tribase-SubNNL2', 'Tribase-SubNNIP', 'Tribase-All']
markers = ['o--', 's--', 'd--', '*--', 'x--']
colors = ['darkgreen', 'darkorange', 'darkblue', 'red', 'purple']

# 自定义 x 轴标签和位置

x_labels = {
    2: {
        "labels": ['$0.99$', '$0.9$'],
        "values": [0.41e-2, 10**-2, 10**-1]
    },
    3: {
        "labels": ['$0.999$', '$0.99$', '$0.9$'],
        "values": [0.51e-3,10**-3, 10**-2, 10**-1]
    },
    4: {
        "labels": ['$0.9999$', '$0.999$', '$0.99$', '$0.9$'],
        "values": [1.1e-5, 10**-4, 10**-3, 10**-2, 10**-1]
    },
}

# 在每个子图上绘制数据
for i, ax in enumerate(axs.flatten()):  # 使用flatten()将2D数组展平
    dataset_label = datasets[i]
    ax.set_title(name_map(dataset_label), fontsize=18, pad=4)
    
    ax.tick_params(axis='y', labelsize=22)  # 设置 y 轴坐标的字体大小
    
    if dataset_label in ["nuswide"]:
        x_label_index = 2
    elif dataset_label in ["msong", "spacev1b"]:
        x_label_index = 3
    else:
        x_label_index = 4
    
    for index, opt_level in enumerate([0, 1, 3, 5, 7]):
        # data = df[(df["dataset"] == dataset_label) & (df["opt_level"] == opt_level)] if opt_level != 0 else faiss_df[faiss_df["dataset"] == dataset_label]
        data = df[(df["dataset"] == dataset_label) & (df["opt_level"] == opt_level)]
        recalls = data["recall"].values
        recalls = np.clip(1 - recalls, 0, 1 - x_labels[x_label_index]["values"][0] * 0.9)  # 避免零值和1值
        qps = data["qps"].values
        ax.plot(recalls, qps, markers[index], label=labels[index], color=colors[index], markerfacecolor=colors[index], zorder=5-index,alpha=0.6,linewidth=0.4,markersize=6)
        print(recalls, qps)
    
    ax.set_yscale('log')
    ax.set_xscale('log')
        
    ax.set_xticks(x_labels[x_label_index]["values"][1:])
    ax.set_xticklabels(x_labels[x_label_index]["labels"], fontsize=18, rotation=-25)
    ax.set_xlim(x_labels[x_label_index]["values"][-1], x_labels[x_label_index]["values"][0])
    ax.set_xlabel('Recall', fontsize=18)

    if i == 0 :
        ax.set_ylabel('Query per second (QPS)', fontsize=15, labelpad=6)   
        
    ax.grid(True, which='major', linestyle='--', linewidth=0.4, color='lightgray')
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    if dataset_label in ["HandOutlines"]:
        ax.yaxis.set_major_locator(ticker.FixedLocator([2e4, 3e4, 4e4, 5e4, 6e4]))
    else:
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
    

    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray')
    ax.yaxis.grid(False, which='minor')
    ax.tick_params(which='minor', length=4, color='black')  # Minor ticks customization

# Create a single legend for all subplots
fig.legend(labels, loc='upper center', ncol=len(labels), fontsize=18, bbox_to_anchor=(0.5, 0.98))

# Adjust layout
plt.tight_layout(w_pad=1)
plt.subplots_adjust(top=0.7, left=0.08) 

plt.savefig('figures/fig7.png', dpi=300)
# plt.savefig('overall.pdf', dpi=300, bbox_inches='tight', format='pdf') 

# Display plot
plt.show()