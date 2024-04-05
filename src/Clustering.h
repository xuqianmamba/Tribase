#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <Eigen/Dense>
#include <random>
#include <vector>
#include <memory>
#include "utils.h"

namespace tribase {

enum class MetricType {
    L2,
    Angular
};

struct ClusteringParameters {
    int niter = 25;
    int seed = 6666;
    int max_points_per_centroid = 256;
    MetricType metric = MetricType::L2; // 默认使用 L2 距离
};

class Clustering {
public:
    Clustering(size_t d, size_t nlist, const ClusteringParameters& cp = ClusteringParameters());
    // 更新train方法的签名，使candidate_codes为const引用
    void train(size_t n, const std::unique_ptr<float[]> &candidate_codes);
    std::unique_ptr<float[]> get_centroids() const;

private:
    size_t d;
    size_t nlist;
    ClusteringParameters cp;

    std::vector<float> centroids;

    // 更新subsample_training_set方法的签名，添加一个用于存储采样数据的参数
    void subsample_training_set(size_t& n, const std::unique_ptr<float[]> &candidate_codes, std::unique_ptr<float[]> &sampled_codes);
    void initialize_centroids(size_t n, std::unique_ptr<float[]> &candidate_codes);
    void update_centroids(size_t n, std::unique_ptr<float[]> &candidate_codes);
    void apply_centroid_perturbations();
};

} // namespace tribase

#endif // CLUSTERING_H