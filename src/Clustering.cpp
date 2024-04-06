#include "Clustering.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace tribase {

Clustering::Clustering(size_t d, size_t nlist, const ClusteringParameters& cp)
    : d(d), nlist(nlist), cp(cp), centroids(nlist * d, 0.0f) {}

void Clustering::train(size_t n, const float* candidate_codes) {
    float* sampling_codes = nullptr;
    subsample_training_set(n, candidate_codes, sampling_codes);

    const float* sampled_codes;

    if (!sampled_codes) {
        sampled_codes = candidate_codes;
    } else {
        sampled_codes = sampling_codes;
    }

    initialize_centroids(n, sampled_codes);

    for (int iter = 0; iter < cp.niter; ++iter) {
        update_centroids(n, sampled_codes);
    }

    apply_centroid_perturbations();

    // 注意：如果sampled_codes是新分配的，需要在这里释放内存
    if (sampled_codes != candidate_codes) {
        delete[] sampled_codes;
    }
}

void Clustering::subsample_training_set(size_t& n, const float* candidate_codes, float*& sampled_codes) {
    size_t max_samples = nlist * cp.max_points_per_centroid;
    if (n <= max_samples) {
        // 如果数据点数量小于或等于最大样本数，不需要采样
        return;
    }

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);  // 填充索引

    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(cp.seed));  // 随机打乱索引

    sampled_codes = new float[max_samples * d];
    for (size_t i = 0; i < max_samples; ++i) {
        std::copy_n(candidate_codes + indices[i] * d, d, sampled_codes + i * d);
    }

    n = max_samples;  // 更新数据点数量
}

void Clustering::initialize_centroids(size_t n, const float* sampled_codes) {
    std::default_random_engine generator(cp.seed);
    std::uniform_int_distribution<size_t> distribution(0, n - 1);

#pragma omp parallel for
    for (size_t i = 0; i < nlist; ++i) {
        size_t randIndex = distribution(generator);
        // 假设每个数据点的维度为d
        for (size_t j = 0; j < d; ++j) {
            centroids[i * d + j] = sampled_codes[randIndex * d + j];
        }
    }
}

void Clustering::update_centroids(size_t n, const float* sampled_codes) {
    Eigen::Map<const Eigen::MatrixXf> codes(sampled_codes, d, n);
    Eigen::Map<Eigen::MatrixXf> centers(centroids.data(), d, nlist);

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
    } else if (cp.metric == MetricType::METRIC_IP) {
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
}

void Clustering::apply_centroid_perturbations() {
    std::default_random_engine generator(cp.seed);
    std::uniform_real_distribution<float> distribution(-0.05, 0.05);  // 微小扰动范围

    for (size_t i = 0; i < nlist * d; ++i) {
        centroids[i] += distribution(generator);
    }
}

float* Clustering::get_centroids() const {
    float* centroid_codes = new float[nlist * d];
    std::memcpy(centroid_codes, centroids.data(), sizeof(float) * nlist * d);
    return centroid_codes;
}

void Clustering::get_centroids(float* centroid_codes) const {
    std::memcpy(centroid_codes, centroids.data(), sizeof(float) * nlist * d);
}

}  // namespace tribase