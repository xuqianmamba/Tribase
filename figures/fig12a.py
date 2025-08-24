import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import name_map
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--faiss_log",
    type=str,
    default="./logs/release-recall-qps-faiss.csv",
    help="Path to the Faiss log CSV file",
)
parser.add_argument(
    "--tribase_log",
    type=str,
    default="./logs/pruning_ratio-recall-qps-tribase.csv",
    help="Path to the Tribase log CSV file",
)
parser.add_argument(
    "--dataset",
    type=str,
    nargs="+",
    default=[
        "fasion_mnist_784",
        "nuswide",
        "msong",
        "sift1m",
        "glove25",
        "HandOutlines",
        "StarLightCurves",
    ],
    help="List of datasets to plot",
)
args = parser.parse_args()

df = pd.read_csv(args.tribase_log)

# 设置全局字体为Type 1字体
mpl.rcParams["pdf.fonttype"] = 42  # 设置为Type 1字体
mpl.rcParams["ps.fonttype"] = 42  # 设置为Type 1字体

# 设置全局字体大小
plt.rcParams.update({"font.size": 21})

# 从提供的文本数据中解析 msong 数据集的比率和召回率
ratios = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

# msong 数据集的召回率
# tribase_all_recall = np.array([1, 0.9997199773788452, 0.9985899925231934, 0.9913600087165833, 0.9646000266075134, 0.8867700099945068, 0.7416099905967712])
# tribase_triangle_recall = np.array([1, 1, 0.9999300241470337, 0.9982900023460388, 0.9888399839401245, 0.9538000226020813, 0.8708800077438354])
# tribase_subnn_L2_recall = np.array([1, 0.9998499751091003, 0.9997000098228455, 0.9992600083351135, 0.9967100024223328, 0.9724500179290771, 0.839900016784668])
# tribase_subnn_IP_recall = np.array([1, 0.9998400211334229, 0.998989999294281, 0.9932500123977661, 0.971750020980835, 0.9066100120544434, 0.8199300169944763])

# 创建图形和轴
fig, ax = plt.subplots(figsize=(7, 4.5))

# 绘制 msong 数据集的召回率
ax.plot(
    ratios,
    df[df["opt_level"] == 7]["recall"][::-1],
    "o--",
    label="Tribase-All",
    color="purple",
)
ax.plot(
    ratios,
    df[df["opt_level"] == 1]["recall"][::-1],
    "s--",
    label="Tribase-Triangle",
    color="orange",
)
ax.plot(
    ratios,
    df[df["opt_level"] == 2]["recall"][::-1],
    "d--",
    label="Tribase-SubNNL2",
    color="blue",
)
ax.plot(
    ratios,
    df[df["opt_level"] == 4]["recall"][::-1],
    "x--",
    label="Tribase-SubNNIP",
    color="red",
)

# 设置图形属性
ax.set_title("Msong Dataset Recall vs. Ratio", loc="center", x=0.44)
ax.set_xlabel("Ratio")
ax.set_ylabel("Recall")
ax.legend()
ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="lightgray")

# 反转横坐标轴
ax.invert_xaxis()
# 调整布局以整体上移图形
plt.subplots_adjust(bottom=0.15, left=0.2)  # 这里的0.2可以根据需要调整

plt.tight_layout()
# 显示图形
plt.savefig("figures/fig12a.png", dpi=300)
