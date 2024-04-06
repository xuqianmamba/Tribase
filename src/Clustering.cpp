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
    std::cout << "subsampling ..." << std::endl;
    // 初始化sampled_codes指向candidate_codes，确保它总是有效的
    const float* sampled_codes = candidate_codes;
    subsample_training_set(n, candidate_codes, sampling_codes);

    // 如果subsampling实际上分配了内存，则更新sampled_codes指向新的采样数据
    if (sampling_codes != nullptr) {
        sampled_codes = sampling_codes;
    }

    std::cout << "initializing ..." << std::endl;
    initialize_centroids(n, sampled_codes);

    for (int iter = 0; iter < cp.niter; ++iter) {
        std::cout << "updating " << iter << " ..." << std::endl;
        update_centroids(n, sampled_codes);
    }

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
    if (!sampled_codes) {
        std::cerr << "错误：sampled_codes为空指针。" << std::endl;
        return; // 直接返回，避免进一步的操作
    }
    std::default_random_engine generator(cp.seed);
    std::uniform_int_distribution<size_t> distribution(0, n - 1);

    std::cout<<"prepare omp ..."<<std::endl;
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


    std::cout<<"n "<<n<<std::endl;
    if (n > 0) {
        std::cout << "第一个聚类中心: ";
        for (size_t j = 0; j < d; ++j) {
            std::cout << centers(j, 0) << " ";
        }
        std::cout << "\n第一个数据点: ";
        for (size_t j = 0; j < d; ++j) {
            std::cout << codes(j, 0) << " ";
        }
        std::cout << std::endl;

        // 计算并打印这两个向量之间的L2距离
        float distance = (centers.col(0) - codes.col(0)).norm();
        std::cout << "L2距离: " << distance << std::endl;
    }


    std::cout<<"Update Initialized"<<std::endl;
    // if (cp.metric == MetricType::METRIC_L2) {
    //     std::cout << "centers.transpose().replicate(1, n).rows(): " << centers.transpose().replicate(1, n).rows() << ", cols(): " << centers.transpose().replicate(1, n).cols() << std::endl;
    //     std::cout << "codes.replicate(1, nlist).transpose().rows(): " << codes.replicate(1, nlist).transpose().rows() << ", cols(): " << codes.replicate(1, nlist).transpose().cols() << std::endl;

    //     Eigen::MatrixXf dists =
    //         (centers.transpose().replicate(1, n) - codes.replicate(1, nlist).transpose()).colwise().squaredNorm();
    //     for (size_t i = 0; i < n; ++i) {
    //         size_t closest_centroid =
    //             std::distance(dists.col(i).data(), std::min_element(dists.col(i).data(), dists.col(i).data() + nlist));
    //         counts[closest_centroid]++;
    //         new_centroids.col(closest_centroid) += codes.col(i);
    //     }
    // } else if (cp.metric == MetricType::METRIC_IP) {
    //     Eigen::MatrixXf dots = centers.transpose() * codes;
    //     for (size_t i = 0; i < n; ++i) {
    //         size_t closest_centroid =
    //             std::distance(dots.col(i).data(), std::max_element(dots.col(i).data(), dots.col(i).data() + nlist));
    //         counts[closest_centroid]++;
    //         new_centroids.col(closest_centroid) += codes.col(i);
    //     }
    // }
    if (cp.metric == MetricType::METRIC_L2) {
        // 初始化一个n x nlist的矩阵来存储所有距离
        Eigen::MatrixXf dists = Eigen::MatrixXf::Zero(n, nlist);

        // 使用OpenMP并行计算每个数据点到每个聚类中心的距离
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < nlist; ++j) {
                // 计算距离的平方
                dists(i, j) = (centers.col(j) - codes.col(i)).squaredNorm();
            }
        }

        // 找到每个数据点最近的聚类中心
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            size_t closest_centroid = std::distance(dists.row(i).data(), std::min_element(dists.row(i).data(), dists.row(i).data() + nlist));
            #pragma omp atomic
            counts[closest_centroid]++;
            #pragma omp critical
            {
                new_centroids.col(closest_centroid) += codes.col(i);
            }
        }
    } else if (cp.metric == MetricType::METRIC_IP) {
        Eigen::MatrixXf dots = centers.transpose() * codes;
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            size_t closest_centroid = std::distance(dots.col(i).data(), std::max_element(dots.col(i).data(), dots.col(i).data() + nlist));
            #pragma omp atomic
            counts[closest_centroid]++;
            #pragma omp critical
            {
                new_centroids.col(closest_centroid) += codes.col(i);
            }
        }
    }  

    std::cout<<"Updating new ..."<<std::endl;
    // 更新 centroids 和 counts
    for (size_t i = 0; i < nlist; ++i) {
        if (counts[i] > 0) {
            centers.col(i) = new_centroids.col(i) / counts[i];
        }
    }

    
    apply_centroid_perturbations();
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