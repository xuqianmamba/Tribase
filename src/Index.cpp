#include "Index.h"
#include <omp.h>
#include <memory>
#include "IVF.h"
#include "IVFScan.hpp"
#include "heap.hpp"

namespace tribase {

Index::Index(size_t d, size_t nlist, size_t nprobe, MetricType metric, size_t sub_k, OptLevel opt_level)
    : d(d), nlist(nlist), nprobe(nprobe), metric(metric), sub_k(sub_k), opt_level(opt_level) {
    lists = std::make_unique<IVF[]>(nlist);
    centroid_codes = std::make_unique<float[]>(nlist * d);
    centroid_ids = std::make_unique<idx_t[]>(nprobe);
    std::iota(centroid_ids.get(), centroid_ids.get() + nlist, 0);
}

// void Index::train(size_t n, std::unique_ptr<float[]>& codes) {
//     // 这里假设Clustering类已经定义好，并且有一个合适的构造函数和train方法
//     ClusteringParameters cp;
//     cp.metric = this->metric;
//     cp.niter = 25;                     // 或其他合适的值
//     cp.seed = 6666;                    // 或其他合适的值
//     cp.max_points_per_centroid = 256;  // 或其他合适的值

//     Clustering clustering(this->d, this->nlist, cp);
//     clustering.train(n, codes);

//     // 假设get_centroids返回的是未归一化的聚类中心
//     this->centroid_codes = clustering.get_centroids();
// }

std::unique_ptr<IVFScanBase> Index::get_scaner(MetricType metric, OptLevel opt_level, size_t k) {
    if (metric == MetricType::METRIC_L2) {
        switch (opt_level) {
            case OptLevel::OPT_NONE:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_NONE>(d, k));
            case OptLevel::OPT_TRIANGLE:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRIANGLE>(d, k));
            case OptLevel::OPT_SUBNN_L2:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_SUBNN_L2>(d, k));
            case OptLevel::OPT_SUBNN_IP:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_SUBNN_IP>(d, k));
            case OptLevel::OPT_TRI_SUBNN_L2:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRI_SUBNN_L2>(d, k));
            case OptLevel::OPT_TRI_SUBNN_IP:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_TRI_SUBNN_IP>(d, k));
            case OptLevel::OPT_ALL:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_L2, OptLevel::OPT_ALL>(d, k));
        }
    } else {
        switch (opt_level) {
            case OptLevel::OPT_NONE:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_INNER_PRODUCT, OptLevel::OPT_NONE>(d, k));
            case OptLevel::OPT_TRIANGLE:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_INNER_PRODUCT, OptLevel::OPT_TRIANGLE>(d, k));
            case OptLevel::OPT_SUBNN_L2:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_INNER_PRODUCT, OptLevel::OPT_SUBNN_L2>(d, k));
            case OptLevel::OPT_SUBNN_IP:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_INNER_PRODUCT, OptLevel::OPT_SUBNN_IP>(d, k));
            case OptLevel::OPT_TRI_SUBNN_L2:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_INNER_PRODUCT, OptLevel::OPT_TRI_SUBNN_L2>(d, k));
            case OptLevel::OPT_TRI_SUBNN_IP:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_INNER_PRODUCT, OptLevel::OPT_TRI_SUBNN_IP>(d, k));
            case OptLevel::OPT_ALL:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_INNER_PRODUCT, OptLevel::OPT_ALL>(d, k));
        }
    }
};

void Index::single_thread_search(size_t n, const float* queries, float* distances, idx_t* labels, Stats* stats) {
    std::unique_ptr<IVFScanBase> scaner_quantizer = get_scaner(metric, OPT_NONE, sub_k);
    std::unique_ptr<IVFScanBase> scaner = get_scaner(metric, opt_level, k);

    std::unique_ptr<float[]> centroid2queries = std::make_unique<float[]>(n * nprobe);
    std::unique_ptr<idx_t[]> listidqueries = std::make_unique<idx_t[]>(n * nprobe);

    float* simi = distances;
    idx_t* idxi = labels;
    float* centroid2query = centroid2queries.get();
    idx_t* listids = listidqueries.get();

    for (size_t i = 0; i < n; i++) {
        scaner_quantizer->set_query(queries + i * d);
        scaner->set_query(queries + i * d);
        scaner_quantizer->lite_scan_codes(nlist, centroid_codes.get(), centroid_ids.get(), centroid2query, listids);
        for (size_t j = 0; j < nprobe; j++) {
            IVF& list = lists[listids[j]];
            float centroid2query = centroid2queries[i * nprobe + j];
            size_t list_size = list.get_list_size();

            const float* codes = list.get_codes();
            const idx_t* ids = reinterpret_cast<const idx_t*>(list.get_ids());
            std::unique_ptr<bool[]> if_skip = std::make_unique<bool[]>(list_size);

            size_t skip_count = 0;
            size_t skip_count_large = 0;
            size_t scan_begin = 0;
            size_t scan_end = list_size;

            const float* sqrt_candidate2centroid = list.get_sqrt_candidate2centroid();
            const float* candidate2centroid = list.get_candidate2centroid();

            float sqrt_simi = sqrt(simi[0]);
            float sqrt_centroid2query = sqrt(centroid2query);
            if (opt_level & OptLevel::OPT_TRIANGLE) {
                for (size_t ii = 0; ii < list_size; ii++) {
                    float tmp = sqrt_simi + sqrt_candidate2centroid[ii];
                    if (tmp < sqrt_centroid2query) {
                        skip_count++;
                    } else {
                        break;
                    }
                }

                for (size_t ii = list_size - 1; ii >= 0; ii--) {
                    float tmp_large = sqrt_simi + sqrt_centroid2query;
                    tmp_large *= tmp_large;
                    if (tmp_large < candidate2centroid[ii]) {
                        skip_count_large++;
                    } else {
                        break;
                    }
                }
                scan_begin = skip_count;
                scan_end -= skip_count_large;
            }

#ifdef DEBUG
            stats->skip_triangle_count += skip_count;
            stats->skip_triangle_large_count += skip_count_large;
            stats->total_count += list_size;
#endif

            scaner->scan_codes(scan_begin, scan_end, list_size, codes, ids, centroid2query, candidate2centroid,
                               sqrt_candidate2centroid, sub_k, list.get_sub_nearest_IP_id(),
                               list.get_sub_nearest_IP_dis(), list.get_sub_farest_IP_id(), list.get_sub_farest_IP_dis(),
                               list.get_sub_nearest_L2_id(), list.get_sub_nearest_L2_dis(), if_skip.get(), simi, idxi,
                               stats);
        }

        simi += k;
        idxi += k;
        centroid2query += nprobe;
        listids += nprobe;
    }
}

void Index::add(size_t n, const float* codes) {
    // lists[0].reset(list_size);
}

void Index::search(size_t n, const float* queries, float* distances, idx_t* labels) {
    // std::unique_ptr<float[]> distances = std::make_unique<float[]>(n * k);
    // std::unique_ptr<idx_t[]> labels = std::make_unique<idx_t[]>(n * k);
    auto init_result = [&](size_t size, float* dis, idx_t* ids) {
        if (metric == MetricType::METRIC_L2) {
            heap_init<MetricType::METRIC_L2>(size, dis, ids);
        } else {
            heap_init<MetricType::METRIC_INNER_PRODUCT>(size, dis, ids);
        }
    };
    init_result(n * k, distances, labels);
    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), n);
    size_t batch_size = (n + nt - 1) / nt;
    std::vector<Stats> stats(nt);

#pragma omp parallel for num_threads(nt)
    for (int i = 0; i < nt; i++) {
        size_t start = i * batch_size;
        size_t end = std::min(start + batch_size, n);
        single_thread_search(end - start, queries + start * d, distances + start * k, labels + start * k, &stats[i]);
    }

    Stats total_stats = mergeStats(stats);
}

}  // namespace tribase