#include "Clustering.h"
#include "utils.h"

using namespace tribase;

int main() {
    auto [data, n, d] = loadFvecs("/home/xuqian/Triangle/Tribase/src/tests/iris.fvecs");
    ClusteringParameters cp;
    cp.niter = 25;
    cp.seed = 6666;
    cp.max_points_per_centroid = 256;
    cp.metric = MetricType::METRIC_L2;

    // for (size_t i = 0; i < n; ++i) {
    //     for (int j = 0; j < d; ++j) {
    //         printf("%f ", data[i * d + j]);
    //     }
    //     printf("\n");
    // }
    Clustering clustering(d, 3, true, cp);
    clustering.train(n, data.get());
    std::unique_ptr<float[]> centroids = std::make_unique<float[]>(3 * d);
    clustering.get_centroids(centroids.get());
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < d; ++j) {
            printf("%f ", centroids[i * d + j]);
        }
        printf("\n");
    }
}