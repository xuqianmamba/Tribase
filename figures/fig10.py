import matplotlib.pyplot as plt
import numpy as np
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
    default=["fasion_mnist_784", "nuswide", "msong", "sift1m", "glove25"],
    help="List of datasets to plot",
)
args = parser.parse_args()

df = pd.read_csv(args.tribase_log)
df_release = pd.read_csv(args.release_tribase_log)

# 设置全局字体为Type 1字体
mpl.rcParams["pdf.fonttype"] = 42  # 设置为Type 1字体
mpl.rcParams["ps.fonttype"] = 42  # 设置为Type 1字体

try:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 17  # 您可以根据需要调整这个值
except Exception:
    plt.rcParams["font.family"] = "Times New Roman"

# 创建一个图表和五个子图
fig, axes = plt.subplots(1, 5, figsize=(22, 4.4))


# 设置子图之间的间距
fig.subplots_adjust(wspace=-0.0)  # 设置子图之间的水平间距为3


# 调整glove25对应的子图位置
axes[4].set_position([0.75, 0.125, 0.133, 0.755])  # 调整左边距，宽度和高度保持不变


# 调整每个子图的位置
for i, ax in enumerate(axes):
    pos = ax.get_position()
    ax.set_position([pos.x0 - 0.02, pos.y0, pos.width, pos.height])  # 向右移动每个子图


# 设置纵坐标标签
for ax in axes:
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])


# 添加Y轴标签，稍微向左上方移动
fig.text(0.001, 0.6, "Pruning Ratio", va="center", rotation="vertical", fontsize=17)


for i, dataset in enumerate(args.dataset):
    ax = axes[i]
    # 绘制剪枝比和开销
    x_positions_pruning = np.array([0.1, 0.4, 0.7, 1.0])  # 第一组位置
    x_positions_cost = np.array([0.18, 0.48, 0.78, 1.08])  # 第二组位置
    pruning_ratios = []
    costs = []
    for opt_level in [1, 2, 4, 7]:
        row_faiss = df[(df["dataset"] == dataset) & (df["opt_level"] == 0)].tail(1)
        row = df[(df["dataset"] == dataset) & (df["opt_level"] == opt_level)].tail(1)
        pruning_speedup = row["pruning_speedup"].values[0]
        row_faiss = df_release[(df_release["dataset"] == dataset) & (df_release["opt_level"] == 0)].tail(1)
        row = df_release[(df_release["dataset"] == dataset) & (df_release["opt_level"] == opt_level)].tail(1)
        time_speedup = row_faiss["query_time"].values[0] / row["query_time"].values[0]
        print(pruning_speedup, time_speedup)
        pruning_ratios.append(
            (1 - 1 / pruning_speedup) * 100 if not row.empty else 0
        )
        costs.append(
            (1 / time_speedup - 1 / pruning_speedup)
            * 100
            if not row.empty
            else 0
        )
    pruning_ratios = np.round(pruning_ratios, 0)
    costs = np.round(costs, 0)
    bars_pruning = ax.bar(x_positions_pruning, pruning_ratios, width=0.05, color="b")
    bars_cost = ax.bar(x_positions_cost, costs, width=0.05, color="r")

    # 在条形图上方标注具体的值
    for j, (pr_height, co_height) in enumerate(zip(pruning_ratios, costs)):
        if pr_height > 45:
            ax.annotate(
                f"{pr_height}%",
                xy=(x_positions_pruning[j], pr_height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=-90,
                bbox=dict(
                    boxstyle="round,pad=0.3", edgecolor="white", facecolor="white"
                ),
            )
            # 绘制折线
            ax.plot(
                [x_positions_pruning[j], x_positions_pruning[j]],
                [pr_height, 55],
                color="black",
                linestyle="--",
                linewidth=0.5,
            )
        else:
            ax.annotate(
                f"{pr_height}%",
                xy=(x_positions_pruning[j], pr_height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=-90,
            )
        ax.annotate(
            f"{co_height}%",
            xy=(x_positions_cost[j], co_height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=-90,
        )

    # if datasets[i] == "sift1m":
    #     ax.set_title(datasets[i], position=(0.6, 1.0))  # 轻微向左调整标题位置
    # if datasets[i] == "glove25":
    #     ax.set_title(datasets[i], position=(0.35, 1.0))  # 轻微向左调整标题位置
    # else:
    ax.set_title(name_map(dataset))
    ax.set_xticks(x_positions_pruning + 0.04)  # 设置x轴标签的位置
    ax.set_xticklabels(
        ["Tribase-Triangle", "Tribase-SubNNL2", "Tribase-SubNNIP", "Tribase-All"],
        rotation=-32,
        fontsize=17,
    )

# # 在整个图表的顶部添加一个全局图例，稍微向左调整
# fig.legend([bars_pruning, bars_cost], ['Pruning Ratio', 'Cost'], bbox_to_anchor=(0.57, 1), ncol=2)
# 在整个图表的顶部添加一个全局图例，稍微向上调整
fig.legend(
    [bars_pruning, bars_cost],
    ["Pruning Ratio", "Cost"],
    bbox_to_anchor=(0.57, 1.05),
    ncol=2,
)  # 向上移动图例位置
# 调整图表的高度以确保子图在整个图片的下方80%的空间
fig.subplots_adjust(top=0.15)


# 调整第一个子图的位置，向右移动一点
# pos = axes[0].get_position()  # 获取第一个子图的位置
# axes[0].set_position([pos.x0 + 0.05, pos.y0, pos.width, pos.height])  # 向右移动第一个子图

plt.tight_layout()
plt.savefig("figures/fig10.png", dpi=300, bbox_inches="tight")
