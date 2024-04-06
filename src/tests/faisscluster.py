import numpy as np
import faiss

def load_fvecs(file_path):
    with open(file_path, 'rb') as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0]  # 读取向量的维度
        f.seek(0, 2)  # 移动到文件末尾
        count = f.tell() // (4 + d * 4)  # 计算向量数量
        f.seek(0)  # 回到文件开头

        # 读取所有向量
        vectors = np.zeros((count, d), dtype='float32')
        for i in range(count):
            f.read(4)  # 跳过每个向量的维度字段
            vectors[i] = np.fromfile(f, dtype=np.float32, count=d)
    return vectors

# 加载数据集
xb = load_fvecs('/home/xuqian/Triangle/Tribase/src/tests/iris.fvecs')

d = xb.shape[1]  # 向量的维度
k = 3  # 聚类中心的数量

# 使用FlatL2索引作为量化器
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, k, faiss.METRIC_L2)

# 准备聚类器
clustering = faiss.Clustering(d, k)
clustering.niter = 0  # 设置迭代次数

# 可以设置其他聚类参数
# clustering.verbose = True
# clustering.min_points_per_centroid = 5   # 每个聚类中心的最小点数
# clustering.max_points_per_centroid = 100  # 每个聚类中心的最大点数

# 进行聚类
clustering.train(xb, index)

# 获取聚类中心
centroids = faiss.vector_float_to_array(clustering.centroids)
centroids = centroids.reshape(k, d)

# 输出聚类中心到命令行
print("聚类中心：")
print(centroids)