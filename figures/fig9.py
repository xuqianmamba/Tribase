import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
from utils import name_map

# 设置全局字体为Type 1字体
mpl.rcParams['pdf.fonttype'] = 42  # 设置为Type 1字体
mpl.rcParams['ps.fonttype'] = 42   # 设置为Type 1字体

# 手动读取和解析数据
data = []
with open(f'logs/hnswlib_recall_release.csv', 'r') as file:
    for line in file:
        parts = line.strip().split(',')  # 使用逗号分割
        if len(parts) >= 7:
            try:
                time = float(parts[4])
                min_time = float(parts[5])
                recall = float(parts[6])
                tri_ef = int(parts[3])
                ef = int(parts[2])
                qps = 1 / time
                data.append([parts[0], ef, tri_ef, qps, min_time, recall])
            except ValueError:
                continue

# 转换为DataFrame
df = pd.DataFrame(data, columns=['dataset', 'ef', 'tri_ef', 'qps', 'min_time', 'recall'])

# 只保留recall > 0.9的数据
df = df[df['recall'] > 0]

# 提取trief=0的数据
def extract_trief_0_data(dataset_name):
    dataset = df[df['dataset'] == dataset_name]
    trief_0 = dataset[dataset['tri_ef'] == 0].copy()
    return trief_0

# 提取trief!=0的数据并适当调整
def calculate_trief_non_0_data(dataset_name):
    dataset = df[df['dataset'] == dataset_name]
    trief_non_0 = dataset[dataset['tri_ef'] != 0].copy()
    return trief_non_0.groupby('ef').apply(lambda x: x.nlargest(1, 'qps')).reset_index(drop=True)

datasets = ["fasion_mnist_784", "nuswide", "msong", "sift1m", "glove25"] # , "HandOutlines", "StarLightCurves"

data = {
    name_map(dataset): (
        extract_trief_0_data(dataset), calculate_trief_non_0_data(dataset)
    ) for dataset in datasets
}

# 绘制图形
plt.figure(figsize=(20, 3.2))  # 调整图形大小，使每个子图接近正方形

for i, (name, (trief_0, trief_non_0)) in enumerate(data.items(), start=1):
    plt.subplot(1, 5, i)
    plt.plot(trief_0['recall'], trief_0['qps'], label='trief=0', color='blue')
    plt.plot(trief_non_0['recall'], trief_non_0['qps'], label='trief!=0, max qps', color='green')
    plt.title(name, fontsize=20)
    plt.xlabel('Recall', fontsize=20)
    if i == 1:
        plt.ylabel('Query per second (QPS)', fontsize=18)
    plt.grid(True)
    # plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    # plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 1))
        # 设置y轴为科学计数法并调整字体大小
    formatter = mticker.ScalarFormatter(useMathText=True)
    # formatter.set_powerlimits((5, 5))  # 只显示10的5次方
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.get_major_formatter().set_scientific(True)  # 启用科学计数法
    
    # 设置科学计数法标签的字体大小
    plt.gca().tick_params(axis='y', labelsize=20)  # 增大y轴刻度数字字体大小
    plt.gca().yaxis.get_offset_text().set_size(20)  # 增大科学计数法的字体大小
    plt.tick_params(axis='both', which='major', labelsize=20)  # 设置刻度数字字体大小



# 只添加一个统一的图例
plt.figlegend(['trief=0', 'trief!=0, max qps'], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize='20', frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局以适应图例
plt.savefig('figures/fig9.png', bbox_inches='tight')