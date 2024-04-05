#include "Clustering.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace tribase {

Clustering::Clustering(size_t d, size_t nlist, const ClusteringParameters& cp)
    : d(d), nlist(nlist), cp(cp), centroids(nlist * d, 0.0f) {}

void Clustering::train(size_t n, const std::unique_ptr<float[]>& candidate_codes) {
    // 创建一个新的unique_ptr用于存储采样后的数据
    std::unique_ptr<float[]> sampled_codes;
    // 调用修改后的subsample_training_set方法，传入sampled_codes以存储采样数据
    subsample_training_set(n, candidate_codes, sampled_codes);

    // 如果采样后的数据为空（即原始数据量小于等于最大样本数），则使用原始数据进行训练
    if (!sampled_codes) {
        sampled_codes = std::make_unique<float[]>(n * d);
        std::copy_n(candidate_codes.get(), n * d, sampled_codes.get());
    }

    initialize_centroids(n, sampled_codes);

    for (int iter = 0; iter < cp.niter; ++iter) {
        update_centroids(n, sampled_codes);
    }

    apply_centroid_perturbations();
}

// void Clustering::subsample_training_set(size_t& n, std::unique_ptr<float[]> &candidate_codes) {
//     size_t max_samples = nlist * cp.max_points_per_centroid;
//     if (n <= max_samples) {
//         // 如果数据点数量小于或等于最大样本数，不需要采样
//         return;
//     }

//     std::vector<size_t> indices(n);
//     std::iota(indices.begin(), indices.end(), 0); // 填充索引

//     std::shuffle(indices.begin(), indices.end(), std::default_random_engine(cp.seed)); // 随机打乱索引

//     std::unique_ptr<float[]> sampled_codes(new float[max_samples * d]);
//     for (size_t i = 0; i < max_samples; ++i) {
//         std::copy_n(candidate_codes.get() + indices[i] * d, d, sampled_codes.get() + i * d);
//     }

//     candidate_codes.swap(sampled_codes); // 使用采样后的数据替换原始数据
//     n = max_samples; // 更新数据点数量
// }

void Clustering::subsample_training_set(size_t& n, const std::unique_ptr<float[]>& candidate_codes,
                                        std::unique_ptr<float[]>& sampled_codes) {
    size_t max_samples = nlist * cp.max_points_per_centroid;
    if (n <= max_samples) {
        // 如果数据点数量小于或等于最大样本数，不需要采样
        // 直接复制原始数据到sampled_codes
        sampled_codes = std::make_unique<float[]>(n * d);
        std::copy_n(candidate_codes.get(), n * d, sampled_codes.get());
        return;
    }

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);  // 填充索引

    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(cp.seed));  // 随机打乱索引

    sampled_codes = std::make_unique<float[]>(max_samples * d);
    for (size_t i = 0; i < max_samples; ++i) {
        std::copy_n(candidate_codes.get() + indices[i] * d, d, sampled_codes.get() + i * d);
    }

    n = max_samples;  // 更新数据点数量
}

void Clustering::initialize_centroids(size_t n, std::unique_ptr<float[]>& candidate_codes) {
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
}

void Clustering::update_centroids(size_t n, std::unique_ptr<float[]>& candidate_codes) {
    Eigen::Map<Eigen::MatrixXf> codes(candidate_codes.get(), d, n);
    Eigen::Map<Eigen::MatrixXf> centers(centroids.data(), d, nlist);

    // 如果是基于角度的聚类，先归一化
    if (cp.metric == MetricType::METRIC_INNER_PRODUCT) {
        for (int i = 0; i < codes.cols(); ++i) {
            if (codes.col(i).norm() > 0) {
                codes.col(i).normalize();
            }
        }
        for (int i = 0; i < centers.cols(); ++i) {
            if (centers.col(i).norm() > 0) {
                centers.col(i).normalize();
            }
        }
    }

    std::vector<size_t> counts(nlist, 0);
    Eigen::MatrixXf new_centroids = Eigen::MatrixXf::Zero(d, nlist);

    if (cp.metric == MetricType::METRIC_L2) {
        Eigen::MatrixXf dists =
            (centers.transpose().replicate(1, n) - codes.replicate(1, nlist).transpose()).colwise().squaredNorm();
        for (size_t i = 0; i < n; ++i) {
            size_t closest_centroid =
                std::distance(dists.col(i).data(), std::min_element(dists.col(i).data(), dists.col(i).data() + nlist));
            counts[closest_centroid]++;
            new_centroids.col(closest_centroid) += codes.col(i);
        }
    } else if (cp.metric == MetricType::METRIC_INNER_PRODUCT) {
        Eigen::MatrixXf dots = centers.transpose() * codes;
        for (size_t i = 0; i < n; ++i) {
            size_t closest_centroid =
                std::distance(dots.col(i).data(), std::max_element(dots.col(i).data(), dots.col(i).data() + nlist));
            counts[closest_centroid]++;
            new_centroids.col(closest_centroid) += codes.col(i);
        }
    }

    // 更新 centroids 和 counts
    for (size_t i = 0; i < nlist; ++i) {
        if (counts[i] > 0) {
            centers.col(i) = new_centroids.col(i) / counts[i];
        }
    }

    if (cp.metric == MetricType::METRIC_INNER_PRODUCT) {
        for (int i = 0; i < centers.cols(); ++i) {
            if (counts[i] > 0) {                    // 确保不会除以零
                centers.col(i) *= sqrt(counts[i]);  // 使用平方根的计数尝试恢复原始尺度
            }
        }
    }
}

void Clustering::apply_centroid_perturbations() {
    std::default_random_engine generator(cp.seed);
    std::uniform_real_distribution<float> distribution(-0.05, 0.05);  // 微小扰动范围

    for (size_t i = 0; i < nlist * d; ++i) {
        centroids[i] += distribution(generator);
    }
}

std::unique_ptr<float[]> Clustering::get_centroids() const {
    std::unique_ptr<float[]> centroid_codes(new float[nlist * d]);
    std::memcpy(centroid_codes.get(), centroids.data(), sizeof(float) * nlist * d);
    return centroid_codes;
}

}  // namespace tribase