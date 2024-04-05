
#ifndef PRE_COMPUTING_H
#define PRE_COMPUTING_H

#include "Index.h"
#include "IVF.h"
#include <cmath>
#include <memory>

namespace tribase {

void computeAssignments(const Index& index, float* vectors, size_t numVectors, size_t dimension) {
    // 假设 vectors 是所有待分配向量的数组
    // numVectors 是向量的数量
    // dimension 是向量的维度

    // 获取聚类中心
    float* centroids = index.getCentroidCodes();

    // 创建 nlist 个 IVF 实例
    std::unique_ptr<IVF[]> ivfs(new IVF[index.getNlist()]);

    for (size_t i = 0; i < numVectors; ++i) {
        float minDistance = std::numeric_limits<float>::max();
        size_t closestCentroidIndex = 0;

        // 计算当前向量到每个聚类中心的距离
        for (size_t j = 0; j < index.getNlist(); ++j) {
            float distance = 0.0;
            for (size_t k = 0; k < dimension; ++k) {
                float diff = vectors[i * dimension + k] - centroids[j * dimension + k];
                distance += diff * diff;
            }
            if (distance < minDistance) {
                minDistance = distance;
                closestCentroidIndex = j;
            }
        }

        // 将向量分配给最近的聚类中心
        // 并计算到聚类中心的L2距离的平方及其平方根
        ivfs[closestCentroidIndex].addVector(i, vectors + i * dimension, minDistance, std::sqrt(minDistance));
    }

    // 此处省略了IVF类中addVector方法的实现细节
    // 您需要在IVF类中添加相应的方法来处理向量的添加、存储距离等
}

} // namespace tribase

#endif // PRE_COMPUTING_H