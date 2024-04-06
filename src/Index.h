#ifndef INDEX_H
#define INDEX_H

#include <memory>
#include "Clustering.h"
#include "IVF.h"
#include "IVFScan.hpp"
#include "common.h"

namespace tribase {

class Index {
   public:
    Index(size_t d, size_t nlist, size_t nprobe, MetricType metric = MetricType::METRIC_L2, OptLevel opt_level = OptLevel::OPT_NONE, size_t sub_k = 0, size_t sub_nlist = 1, size_t sub_nprobe = 1);

    void train(size_t n, const float* codes);

    void single_thread_nearest_cluster_search(size_t n, const float* queries, float* distances, idx_t* labels);
    void single_thread_search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, Stats* stats);

    void add(size_t n, const float* codes);

    void search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels);

    std::unique_ptr<IVFScanBase> get_scaner(MetricType metric, OptLevel opt_level, size_t k);

    // 其他查询方法的声明

   private:
    size_t d;
    size_t nlist;
    size_t nprobe;
    MetricType metric;
    OptLevel opt_level;

    size_t sub_k;
    size_t sub_nlist;
    size_t sub_nprobe;
    std::unique_ptr<IVF[]> lists;
    // std::vector<IVF> lists;
    std::unique_ptr<float[]> centroid_codes;
    std::unique_ptr<idx_t[]> centroid_ids;
};

}  // namespace tribase

#endif  // INDEX_H