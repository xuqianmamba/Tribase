#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <vector>

namespace tribase{

struct ClusteringParameters {
    int niter = 25; // number of clustering iterations
    int seed = 6666; // seed for the random number generator
    int max_points_per_centroid = 256; // to limit size of dataset, otherwise the training set is subsampled
};


class Clustering {
public:
    Clustering(size_t d, size_t nlist, const ClusteringParameters& cp = ClusteringParameters());

    void train(size_t n, const float* x, Index& index);

private:
    size_t d; // dimension of the vectors
    size_t nlist; // number of centroids
    ClusteringParameters cp;

    std::vector<float> centroids; // centroids (nlist * d)

    void subsample_training_set(size_t& n, const float*& x);
    void initialize_centroids(size_t n, const float* x);
    void update_centroids(size_t n, const float* x, Index& index);
    void apply_centroid_perturbations();
};

}

#endif // CLUSTERING_H