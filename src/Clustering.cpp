#include "Clustering.h"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace tribase {

Clustering::Clustering(size_t d, size_t nlist, bool verbose, const ClusteringParameters& cp)
    : d(d), nlist(nlist), verbose(verbose), cp(cp), centroids(nlist * d, 0.0f) {}

void Clustering::train(size_t n, const float* candidate_codes) {
    float* sampling_codes = nullptr;
    const float* sampled_codes = candidate_codes;
    subsample_training_set(n, candidate_codes, sampling_codes);

    // 如果subsampling实际上分配了内存，则更新sampled_codes指向新的采样数据
    if (sampling_codes != nullptr) {
        sampled_codes = sampling_codes;
    }

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    initialize_centroids(n, sampled_codes);

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();

    // 计算并输出执行时间
    std::chrono::duration<double, std::milli> elapsed = end - start;

    for (int iter = 0; iter < cp.niter; ++iter) {
        if (verbose) {
            std::cout << "Iteration " << iter + 1 << " of " << cp.niter << std::endl;
        }
        update_centroids(n, sampled_codes);
    }

    // 注意：如果sampled_codes是新分配的，需要在这里释放内存
    if (sampling_codes) {
        delete[] sampling_codes;
    }
}

void Clustering::subsample_training_set(size_t& n, const float* candidate_codes, float*& sampling_codes) {
    size_t max_samples = nlist * cp.max_points_per_centroid;
    if (n <= max_samples) {
        // 如果数据点数量小于或等于最大样本数，不需要采样
        return;
    }

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);  // 填充索引

    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(cp.seed));  // 随机打乱索引

    sampling_codes = new float[max_samples * d];

#pragma omp parallel for
    for (size_t i = 0; i < max_samples; ++i) {
        std::copy_n(candidate_codes + indices[i] * d, d, sampling_codes + i * d);
    }

    n = max_samples;  // 更新数据点数量
}

// void Clustering::initialize_centroids(size_t n, const float* sampled_codes) {
//     if (!sampled_codes) {
//         std::cerr << "错误：sampled_codes为空指针。" << std::endl;
//         return;
//     }

//     std::vector<size_t> chosen_indices;
//     std::default_random_engine generator(cp.seed);
//     std::uniform_int_distribution<size_t> uniform_dist(0, n - 1);

//     size_t first_index = uniform_dist(generator);
//     chosen_indices.push_back(first_index);

//     for (size_t i = 1; i < nlist; ++i) {
//         std::vector<float> distances(n, std::numeric_limits<float>::max());

//         for (size_t j = 0; j < n; ++j) {
//             for (size_t idx : chosen_indices) {
//                 float dist = 0.0;
//                 for (size_t k = 0; k < d; ++k) {
//                     float diff = sampled_codes[j * d + k] - sampled_codes[idx * d + k];
//                     dist += diff * diff;
//                 }
//                 distances[j] = std::min(distances[j], dist);
//             }
//         }

//         std::discrete_distribution<size_t> weighted_dist(distances.begin(), distances.end());
//         size_t next_index = weighted_dist(generator);
//         chosen_indices.push_back(next_index);
//     }

//     for (size_t i = 0; i < nlist; ++i) {
//         size_t index = chosen_indices[i];
//         for (size_t j = 0; j < d; ++j) {
//             centroids[i * d + j] = sampled_codes[index * d + j];
//         }
//     }
// }

// void Clustering::initialize_centroids(size_t n, const float* sampled_codes) {
//     if (!sampled_codes) {
//         std::cerr << "错误：sampled_codes为空指针。" << std::endl;
//         return;
//     }

//     // 生成随机索引
//     std::vector<size_t> indices(n);
//     std::iota(indices.begin(), indices.end(), 0);  // 填充索引
//     std::shuffle(indices.begin(), indices.end(), std::default_random_engine(cp.seed));  // 随机打乱索引

//     // 根据随机索引选择初始聚类中心
//     for (size_t i = 0; i < nlist; ++i) {
//         size_t index = indices[i];
//         std::copy_n(sampled_codes + index * d, d, centroids.data() + i * d);
//     }
// }

// Pre
void Clustering::initialize_centroids(size_t n, const float* sampled_codes) {
    if (!sampled_codes) {
        std::cerr << "Error: sampled_codes is a null pointer." << std::endl;
        return;
    }

    // 生成随机索引
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);                                       // 从0开始填充索引
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(cp.seed));  // 使用随机引擎打乱索引

    // 根据随机索引选择初始聚类中心
    centroids.resize(nlist * d);  // 确保centroids有足够的空间存储所有中心
    for (size_t i = 0; i < nlist; ++i) {
        size_t index = indices[i];                                            // 获取随机索引
        std::copy_n(sampled_codes + index * d, d, centroids.data() + i * d);  // 复制选中的点作为聚类中心
    }
}
// void Clustering::initialize_centroids(size_t n, const float* sampled_codes) {
//     if (!sampled_codes) {
//         std::cerr << "错误：sampled_codes为空指针。" << std::endl;
//         return;
//     }

//     std::vector<size_t> chosen_indices;
//     std::default_random_engine generator(cp.seed);
//     std::uniform_int_distribution<size_t> uniform_dist(0, n - 1);

//     size_t first_index = uniform_dist(generator);
//     chosen_indices.push_back(first_index);
//     centroids.assign(sampled_codes + first_index * d, sampled_codes + (first_index + 1) * d);

//     for (size_t i = 1; i < nlist; ++i) {
//         if (verbose) {
//             std::cout << "init " << i << " of " << nlist << std::endl;
//         }
//         std::vector<double> distances(n);
//         double total_distance = 0.0;

// #pragma omp parallel for reduction(+ : total_distance)
//         for (size_t j = 0; j < n; ++j) {
//             double min_dist = std::numeric_limits<double>::max();
//             for (size_t idx = 0; idx < chosen_indices.size(); ++idx) {
//                 double dist = 0.0;
//                 for (size_t k = 0; k < d; ++k) {
//                     double diff = sampled_codes[j * d + k] - centroids[idx * d + k];
//                     dist += diff * diff;
//                 }
//                 min_dist = std::min(min_dist, dist);
//             }
//             distances[j] = min_dist;
//             total_distance += min_dist;
//         }

//         std::uniform_real_distribution<double> dist(0.0, total_distance);
//         double threshold = dist(generator);
//         double sum = 0.0;
//         for (size_t j = 0; j < n; ++j) {
//             sum += distances[j];
//             if (sum >= threshold) {
//                 chosen_indices.push_back(j);
//                 centroids.insert(centroids.end(), sampled_codes + j * d, sampled_codes + (j + 1) * d);
//                 break;
//             }
//         }
//     }
// }

// void Clustering::initialize_centroids(size_t n, const float* sampled_codes) {
//     if (!sampled_codes) {
//         std::cerr << "错误：sampled_codes为空指针。" << std::endl;
//         return; // 直接返回，避免进一步的操作
//     }
//     std::default_random_engine generator(cp.seed);
//     std::uniform_int_distribution<size_t> distribution(0, n - 1);

//     std::cout<<"prepare omp ..."<<std::endl;
// #pragma omp parallel for
//     for (size_t i = 0; i < nlist; ++i) {
//         size_t randIndex = distribution(generator);
//         // 假设每个数据点的维度为d
//         for (size_t j = 0; j < d; ++j) {
//             centroids[i * d + j] = sampled_codes[randIndex * d + j];
//         }
//     }
// }

// void Clustering::update_centroids(size_t n, const float* sampled_codes) {
//     Eigen::Map<const Eigen::MatrixXf> codes(sampled_codes, d, n);
//     Eigen::Map<Eigen::MatrixXf> centers(centroids.data(), d, nlist);

//     std::vector<size_t> counts(nlist, 0);
//     Eigen::MatrixXf new_centroids = Eigen::MatrixXf::Zero(d, nlist);

//     std::cout<<"n "<<n<<std::endl;
//     // if (n > 0) {
//     //     std::cout << "第一个聚类中心: ";
//     //     for (size_t j = 0; j < d; ++j) {
//     //         std::cout << centers(j, 0) << " ";
//     //     }
//     //     std::cout << "\n第一个数据点: ";
//     //     for (size_t j = 0; j < d; ++j) {
//     //         std::cout << codes(j, 0) << " ";
//     //     }
//     //     std::cout << std::endl;

//     //     // 计算并打印这两个向量之间的L2距离
//     //     float distance = (centers.col(0) - codes.col(0)).norm();
//     //     std::cout << "L2距离: " << distance << std::endl;
//     for (size_t center_idx = 0; center_idx < std::min(nlist, static_cast<size_t>(3)); ++center_idx) {
//         std::cout << "聚类中心 " << center_idx + 1 << ": ";
//         for (size_t j = 0; j < d; ++j) {
//             std::cout << centers(j, center_idx) << " ";
//         }
//         std::cout << std::endl;
//     }

//     std::cout<<"Update Initialized"<<std::endl;
//     // if (cp.metric == MetricType::METRIC_L2) {
//     //     std::cout << "centers.transpose().replicate(1, n).rows(): " << centers.transpose().replicate(1, n).rows() << ", cols(): " << centers.transpose().replicate(1, n).cols() << std::endl;
//     //     std::cout << "codes.replicate(1, nlist).transpose().rows(): " << codes.replicate(1, nlist).transpose().rows() << ", cols(): " << codes.replicate(1, nlist).transpose().cols() << std::endl;

//     //     Eigen::MatrixXf dists =
//     //         (centers.transpose().replicate(1, n) - codes.replicate(1, nlist).transpose()).colwise().squaredNorm();
//     //     for (size_t i = 0; i < n; ++i) {
//     //         size_t closest_centroid =
//     //             std::distance(dists.col(i).data(), std::min_element(dists.col(i).data(), dists.col(i).data() + nlist));
//     //         counts[closest_centroid]++;
//     //         new_centroids.col(closest_centroid) += codes.col(i);
//     //     }
//     // } else if (cp.metric == MetricType::METRIC_IP) {
//     //     Eigen::MatrixXf dots = centers.transpose() * codes;
//     //     for (size_t i = 0; i < n; ++i) {
//     //         size_t closest_centroid =
//     //             std::distance(dots.col(i).data(), std::max_element(dots.col(i).data(), dots.col(i).data() + nlist));
//     //         counts[closest_centroid]++;
//     //         new_centroids.col(closest_centroid) += codes.col(i);
//     //     }
//     // }
//     if (cp.metric == MetricType::METRIC_L2) {
//         // 初始化一个n x nlist的矩阵来存储所有距离
//         Eigen::MatrixXf dists = Eigen::MatrixXf::Zero(n, nlist);

//         // 使用OpenMP并行计算每个数据点到每个聚类中心的距离
//         #pragma omp parallel for collapse(2)
//         for (size_t i = 0; i < n; ++i) {
//             for (size_t j = 0; j < nlist; ++j) {
//                 // 计算距离的平方
//                 dists(i, j) = (centers.col(j) - codes.col(i)).squaredNorm();
//             }
//         }

//         // 找到每个数据点最近的聚类中心
//         #pragma omp parallel for
//         for (size_t i = 0; i < n; ++i) {
//             size_t closest_centroid = std::distance(dists.row(i).data(), std::min_element(dists.row(i).data(), dists.row(i).data() + nlist));
//             #pragma omp atomic
//             counts[closest_centroid]++;
//             #pragma omp critical
//             {
//                 new_centroids.col(closest_centroid) += codes.col(i);
//             }
//         }
//     } else if (cp.metric == MetricType::METRIC_IP) {
//         Eigen::MatrixXf dots = centers.transpose() * codes;
//         #pragma omp parallel for
//         for (size_t i = 0; i < n; ++i) {
//             size_t closest_centroid = std::distance(dots.col(i).data(), std::max_element(dots.col(i).data(), dots.col(i).data() + nlist));
//             #pragma omp atomic
//             counts[closest_centroid]++;
//             #pragma omp critical
//             {
//                 new_centroids.col(closest_centroid) += codes.col(i);
//             }
//         }
//     }

//     std::cout<<"Updating new ..."<<std::endl;
//     // 更新 centroids 和 counts
//     for (size_t i = 0; i < nlist; ++i) {
//         if (counts[i] > 0) {
//             centers.col(i) = new_centroids.col(i) / counts[i];
//         }
//     }

//     apply_centroid_perturbations();
// }
// void Clustering::update_centroids(size_t n, const float* sampled_codes) {
//     Eigen::Map<const Eigen::MatrixXf> codes(sampled_codes, d, n);
//     Eigen::Map<Eigen::MatrixXf> centers(centroids.data(), d, nlist);

//     std::vector<size_t> counts(nlist, 0);
//     Eigen::MatrixXf new_centroids = Eigen::MatrixXf::Zero(d, nlist);
// #pragma omp parallel for
//     for (size_t i = 0; i < n; ++i) {
//         size_t closest_centroid = 0;
//         float min_dist = std::numeric_limits<float>::max();
//         for (size_t j = 0; j < nlist; ++j) {
//             float dist = (centers.col(j) - codes.col(i)).squaredNorm();
//             if (dist < min_dist) {
//                 min_dist = dist;
//                 closest_centroid = j;
//             }
//         }

// #pragma omp atomic
//         counts[closest_centroid]++;

// #pragma omp critical
//         {
//             new_centroids.col(closest_centroid) += codes.col(i);
//         }
//     }

//     for (size_t j = 0; j < nlist; ++j) {
//         if (counts[j] > 0) {
//             centers.col(j) = new_centroids.col(j) / counts[j];
//         }
//     }

//     apply_centroid_perturbations();
// }

void Clustering::update_centroids(size_t n, const float* sampled_codes) {
    Eigen::Map<const Eigen::MatrixXf> codes(sampled_codes, d, n);
    Eigen::Map<Eigen::MatrixXf> centers(centroids.data(), d, nlist);

    std::vector<size_t> counts(nlist, 0);
    Eigen::MatrixXf new_centroids = Eigen::MatrixXf::Zero(d, nlist);

    if (cp.metric == MetricType::METRIC_L2) {
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            size_t closest_centroid = 0;
            float min_dist = std::numeric_limits<float>::max();
            for (size_t j = 0; j < nlist; ++j) {
                float dist = (centers.col(j) - codes.col(i)).squaredNorm();
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }

            #pragma omp atomic
            counts[closest_centroid]++;

            #pragma omp critical
            {
                new_centroids.col(closest_centroid) += codes.col(i);
            }
        }
    } else if (cp.metric == MetricType::METRIC_IP) {
        Eigen::MatrixXf normalized_codes = codes.colwise().normalized();

        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            size_t closest_centroid = 0;
            float max_ip = std::numeric_limits<float>::lowest();
            for (size_t j = 0; j < nlist; ++j) {
                float ip = normalized_codes.col(i).dot(centers.col(j));
                if (ip > max_ip) {
                    max_ip = ip;
                    closest_centroid = j;
                }
            }

            #pragma omp atomic
            counts[closest_centroid]++;

            #pragma omp critical
            {
                new_centroids.col(closest_centroid) += codes.col(i); 
            }
        }
    }

    for (size_t i = 0; i < nlist; ++i) {
        if (counts[i] > 0) {
            centers.col(i) = new_centroids.col(i) / counts[i];
        }
    }

}


void Clustering::apply_centroid_perturbations() {
    for (size_t i = 0; i < nlist; ++i) {
        for (size_t j = 0; j < d; ++j) {
            // 对于单数聚类中心（索引从0开始，因此这里检查的是偶数索引）
            if (i % 2 == 0) {
                centroids[i * d + j] -= 1e-6;
            } else {  // 对于双数聚类中心
                centroids[i * d + j] += 1e-6;
            }
        }
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