#pragma once

#include <eigen3/Eigen/Dense>
#include <memory>
#include <random>
#include <vector>

#include "common.h"
#include "utils.h"

namespace tribase {

struct ClusteringParameters {
    int niter = 25;
    int seed = 1234;
    int max_points_per_centroid = 256;
    MetricType metric = MetricType::METRIC_L2;  // 默认使用 L2 距离
};

class Clustering {
   public:
    Clustering(size_t d, size_t nlist, bool verbose = false, const ClusteringParameters& cp = ClusteringParameters());
    void train(size_t n, const float* candidate_codes);
    float* get_centroids() const;
    void get_centroids(float* centroid_codes) const;

   private:
    size_t d;
    size_t nlist;
    bool verbose;
    ClusteringParameters cp;

    std::vector<float> centroids;

    void subsample_training_set(size_t& n, const float* candidate_codes, float*& sampling_codes);
    void initialize_centroids(size_t n, const float* candidate_codes);
    void update_centroids(size_t n, const float* candidate_codes);
    void apply_centroid_perturbations();
};

}  // namespace tribase