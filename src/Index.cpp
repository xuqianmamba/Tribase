#include "Index.h"

namespace tribase {

Index::Index(size_t d, size_t nlist, size_t nprobe, MetricType metric)
    : d(d), nlist(nlist), nprobe(nprobe), metric(metric), sub_k(0) {
    // 初始化lists和centroid_codes
    lists = std::make_unique<IVF[]>(nlist);
    centroid_codes = std::make_unique<float[]>(nlist * d);
}

void Index::train(size_t n, std::unique_ptr<float[]> &codes) {
    // 这里假设Clustering类已经定义好，并且有一个合适的构造函数和train方法
    ClusteringParameters cp;
    cp.metric = this->metric;
    cp.niter = 25;                     // 或其他合适的值
    cp.seed = 6666;                    // 或其他合适的值
    cp.max_points_per_centroid = 256;  // 或其他合适的值

    Clustering clustering(this->d, this->nlist, cp);
    clustering.train(n, codes);

    // 假设get_centroids返回的是未归一化的聚类中心
    this->centroid_codes = clustering.get_centroids();
}

// 其他查询方法的实现

}  // namespace tribase