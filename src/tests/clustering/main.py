import numpy as np
import sklearn.cluster as cluster
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()

# output2fvecs binary file


# def write_fvecs(file_name, vectors):
#     with open(file_name, "wb") as f:
#         for vec in vectors:
#             # 向量维度，作为一个32位整数写入
#             dim = np.array([len(vec)], dtype=np.int32)
#             dim.tofile(f)
#             # 向量数据，作为32位浮点数写入
#             vec.astype(np.float32).tofile(f)
# write_fvecs("iris.fvecs", iris.data)


# Create a KMeans model
kmeans = cluster.KMeans(n_clusters=3, n_init="auto")

# Fit the model
kmeans.fit(iris.data)

# Get the predicted labels
labels = kmeans.predict(iris.data)

# print cluster centers

print(kmeans.cluster_centers_)

# Plot the data
# plt.scatter(iris.data[:, 0], iris.data[:, 1], c=labels)
# plt.savefig("iris_clusters.png")
