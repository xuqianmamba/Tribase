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
    Index(size_t d = 0,
          size_t nlist = 0,
          size_t nprobe = 0,
          MetricType metric = MetricType::METRIC_L2,
          OptLevel opt_level = OptLevel::OPT_NONE,
          size_t sub_k = 0,
          size_t sub_nlist = 1,
          size_t sub_nprobe = 1,
          bool verbose = false,
          EdgeDevice edge_device_enabled = EdgeDevice::EDGEDEVIVE_DISABLED);

    Index& operator=(Index&& other) noexcept;

    void train(size_t n, const float* codes, bool faiss = false, bool lite = false);

    void single_thread_nearest_cluster_search(size_t n, const float* queries, float* distances, idx_t* labels);
    void single_thread_search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, float ratio, Stats* stats);

    void add(size_t n, const float* codes);

    Stats search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, float ratio = 1.0);
    void save_index(std::string path) const;
    void load_index(std::string path);
    void load_SPANN(std::string path);

    // 其他查询方法的声明

   private:
    std::unique_ptr<IVFScanBase> get_scanner(MetricType metric, OptLevel opt_level, size_t k, EdgeDevice edge_device_enabled = EdgeDevice::EDGEDEVIVE_DISABLED);

   public:
    size_t d;
    size_t nlist;
    size_t nprobe;
    MetricType metric;
    OptLevel opt_level;
    OptLevel added_opt_level;

    size_t sub_k;
    size_t sub_nlist;
    size_t sub_nprobe;

    bool verbose;
    EdgeDevice edge_device_enabled;

    std::unique_ptr<IVF[]> lists;
    std::unique_ptr<float[]> centroid_codes;
    std::unique_ptr<idx_t[]> centroid_ids;
};

}  // namespace tribase

#endif  // INDEX_H