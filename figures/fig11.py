import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib as mpl
import pandas as pd
import argparse
from utils import name_map

parser = argparse.ArgumentParser()
parser.add_argument(
    "--faiss_log",
    type=str,
    default="./logs/standard-recall-qps-faiss.csv",
    help="Path to the Faiss log CSV file",
)
parser.add_argument(
    "--tribase_log",
    type=str,
    default="./logs/standard-recall-qps-tribase.csv",
    help="Path to the Tribase log CSV file",
)
parser.add_argument(
    "--release_tribase_log",
    type=str,
    default="./logs/edge-recall-qps-tribase.csv",
    help="Path to the release Tribase log CSV file",
)
parser.add_argument(
    "--dataset",
    type=str,
    nargs="+",
    default=["fasion_mnist_784", "glove25", "msong", "nuswide", "sift1m"],
    help="List of datasets to plot",
)
args = parser.parse_args()

df = pd.read_csv(args.tribase_log)
df_release = pd.read_csv(args.release_tribase_log)

# 设置全局字体为Type 1字体
mpl.rcParams["pdf.fonttype"] = 42  # 设置为Type 1字体
mpl.rcParams["ps.fonttype"] = 42  # 设置为Type 1字体


# 设置字体和大小
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Arial"

# 创建图形和子图
fig, axs = plt.subplots(1, 5, figsize=(20, 3.7))  # 创建5个子图

# x轴的标签和位置
x_labels = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
x_positions = np.linspace(0.1, 1.0, 10)
colors = ["green", "orange", "blue", "red"]

# datasets = ['fashion', 'glove25', 'msong', 'nuswide', 'sift1m']

# 标签和标记
labels = ["Tribase-Triangle", "Tribase-SubNNL2", "Tribase-SubNNIP", "Tribase-All"]
markers = ["o--", "s--", "d--", "*--"]

# 绘制每个数据集的图表
for i, dataset_label in enumerate(args.dataset):
    dataset = name_map(dataset_label)
    for j, (opt_level, label) in enumerate(zip([1, 2, 4, 7], labels)):
        # axs[i].plot(x_positions, data[dataset][label] * 100, markers[j], label=label, color=colors[j], markerfacecolor=colors[j])
        data = df[(df["dataset"] == dataset_label) & (df["opt_level"] == opt_level)]
        axs[i].plot(
            data["nprobe"].values / data["nlist"].values,
            100 * (1 - 1 / data["pruning_speedup"].values),
            markers[j],
            label=label,
            color=colors[j],
            markerfacecolor=colors[j],
        )
    axs[i].set_title(dataset, fontsize=25)
    axs[i].set_xticks(x_positions)
    axs[i].set_xticklabels(x_labels)
    axs[i].set_xlabel("Probes/Lists", fontsize=25)
    axs[i].set_ylim([0, 100])
    axs[i].grid(True, which="both", linestyle="--", linewidth=0.5, color="lightgray")
    axs[i].tick_params(axis="x", rotation=-40, labelsize=18)
    axs[i].tick_params(axis="y", labelsize=18)
    # axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))  # 将图例放在子图上方

# 设置共同的y轴标签
fig.text(0.00, 0.55, "Pruning Ratio (%)", va="center", rotation="vertical", fontsize=25)

fig.legend(
    labels,
    loc="upper center",
    ncol=len(labels),
    fontsize=22,
    bbox_to_anchor=(0.5, 1.03),
)
plt.tight_layout()
plt.subplots_adjust(top=0.75, left=0.04)

plt.savefig("figures/fig11.png", dpi=300)
# plt.savefig('purning_nprobe.pdf', dpi=300, bbox_inches='tight', format='pdf')

# plt.show()
