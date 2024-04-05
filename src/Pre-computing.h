#include "Index.h"
#include "IVF.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace tribase {

void addVectorsToIVF(Index& index, const float* codes, size_t n) {
    std::vector<size_t> counts(index.nlist, 0); // 每个聚类中心的向量计数
    std::vector<std::vector<size_t>> assignments(index.nlist); // 每个聚类中心的向量索引
    std::vector<std::vector<float>> distances(index.nlist); // 存储每个向量到聚类中心的距离平方

    // 步骤1和2: 计算距离并分配向量到最近的聚类中心
    for (size_t i = 0; i < n; ++i) {
        float minDistance = std::numeric_limits<float>::max();
        size_t closestCentroid = 0;
        for (size_t j = 0; j < index.nlist; ++j) {
            // 使用Eigen计算向量与聚类中心之间的距离平方
            float distance = (codesMatrix.col(i) - centroidsMatrix.col(j)).squaredNorm();
            if (distance < minDistance) {
                minDistance = distance;
                closestCentroid = j;
            }
        }
        assignments[closestCentroid].push_back(i);
        distances[closestCentroid].push_back(minDistance);
        counts[closestCentroid]++;
    }

    // 步骤3和4: 重置每个聚类中心的candidate_codes大小
    for (size_t j = 0; j < index.nlist; ++j) {
        index.lists[j].reset(counts[j], index.d, index.sub_k);
    }

    // 步骤5: 将向量添加到对应的聚类中心，并更新距离信息
    for (size_t j = 0; j < index.nlist; ++j) {
        for (size_t idx = 0; idx < assignments[j].size(); ++idx) {
            size_t vectorIdx = assignments[j][idx];
            float distance = distances[j][idx];
            // 复制向量到candidate_codes
            std::copy(codes + vectorIdx * index.d, codes + (vectorIdx + 1) * index.d, index.lists[j].get_candidate_codes(idx));
            // 更新candidate_id
            *(index.lists[j].get_candidate_id(idx)) = vectorIdx;
            // 更新candidate2centroid和sqrt_candidate2centroid
            *(index.lists[j].get_candidate2centroid() + idx) = distance;
            *(index.lists[j].get_sqrt_candidate2centroid(idx)) = std::sqrt(distance);
        }
    }
}


void performClusteringForAllIVFLists(Index& index) {
    // 假设每个聚类的nlist都设置为100
    const size_t nlistPerCluster = 100;

    // 遍历每个IVFList
    for (size_t i = 0; i < index.nlist; ++i) {
        IVFList& ivfList = index.lists[i];
        size_t n = ivfList.size(); // 获取IVFList中的向量数量
        const float* codes = ivfList.get_codes(); // 获取IVFList中的向量数据

        // 为L2距离创建聚类实例
        ClusteringParameters cpL2;
        cpL2.metric = MetricType::L2;
        Clustering clusteringL2(index.d, nlistPerCluster, cpL2);
        clusteringL2.train(n, codes);

        //

        // 为角度（余弦相似度）创建聚类实例
        ClusteringParameters cpAngular;
        cpAngular.metric = MetricType::Angular;
        Clustering clusteringAngular(index.d, nlistPerCluster, cpAngular);
        clusteringAngular.train(n, codes);

        // PPP
    }
}


} // namespace tribase