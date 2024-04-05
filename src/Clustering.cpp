#include "Clustering.h"
#include <omp.h>
#include <cmath>
#include <limits>
#include <algorithm>

namespace tribase {

Clustering::Clustering(size_t d, size_t nlist, const ClusteringParameters& cp)
    : d(d), nlist(nlist), cp(cp), centroids(nlist * d, 0.0f) {}

void Clustering::train(size_t n, std::unique_ptr<float[]> &candicate_codes) {
    subsample_training_set(n, candicate_codes);
    initialize_centroids(n, candicate_codes);

    for (int iter = 0; iter < cp.niter; ++iter) {
        update_centroids(n, candicate_codes);
    }

    apply_centroid_perturbations();
}

void Clustering::subsample_training_set(size_t& n, std::unique_ptr<float[]> &candidate_codes) {
    size_t max_samples = nlist * cp.max_points_per_centroid;
    if (n <= max_samples) {
        // 如果数据点数量小于或等于最大样本数，不需要采样
        return;
    }

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0); // 填充索引

    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(cp.seed)); // 随机打乱索引

    std::unique_ptr<float[]> sampled_codes(new float[max_samples * d]);
    for (size_t i = 0; i < max_samples; ++i) {
        std::copy_n(candidate_codes.get() + indices[i] * d, d, sampled_codes.get() + i * d);
    }

    candidate_codes.swap(sampled_codes); // 使用采样后的数据替换原始数据
    n = max_samples; // 更新数据点数量
}



void Clustering::initialize_centroids(size_t n, std::unique_ptr<float[]> &candidate_codes) {
    std::default_random_engine generator(cp.seed);
    std::uniform_int_distribution<size_t> distribution(0, n - 1);

    #pragma omp parallel for
    for (size_t i = 0; i < nlist; ++i) {
        size_t randIndex = distribution(generator);
        // 假设每个数据点的维度为d
        for (size_t j = 0; j < d; ++j) {
            centroids[i * d + j] = candidate_codes[randIndex * d + j];
        }
    }

void Clustering::update_centroids(size_t n, std::unique_ptr<float[]> &candidate_codes) {
    std::vector<size_t> counts(nlist, 0); // 每个聚类中的点数
    std::vector<float> new_centroids(nlist * d, 0.0f); // 新的聚类中心

    // 累加每个聚类中的所有点
    #pragma omp parallel for reduction(+:new_centroids[:nlist*d], counts[:nlist])
    for (size_t i = 0; i < n; ++i) {
        size_t closest_centroid = /* 计算最近的聚类中心索引 */;
        for (size_t j = 0; j < d; ++j) {
            new_centroids[closest_centroid * d + j] += candidate_codes[i * d + j];
        }
        counts[closest_centroid]++;
    }

    // 计算新的聚类中心位置
    for (size_t i = 0; i < nlist; ++i) {
        if (counts[i] > 0) { // 避免除以0
            for (size_t j = 0; j < d; ++j) {
                centroids[i * d + j] = new_centroids[i * d + j] / counts[i];
            }
        }
    }
}

void Clustering::apply_centroid_perturbations() {
    std::default_random_engine generator(cp.seed);
    std::uniform_real_distribution<float> distribution(-0.05, 0.05); // 微小扰动范围

    for (size_t i = 0; i < nlist * d; ++i) {
        centroids[i] += distribution(generator);
    }
}


std::unique_ptr<float[]> Clustering::get_centroids() const {
    std::unique_ptr<float[]> centroid_codes(new float[nlist * d]);
    std::memcpy(centroid_codes.get(), centroids.data(), sizeof(float) * nlist * d);
    return centroid_codes;
}


} // namespace tribase