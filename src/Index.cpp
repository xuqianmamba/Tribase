#include "Index.h"
#include <omp.h>
#include <cmath>
#include <memory>
#include "IVF.h"
#include "IVFScan.hpp"
#include "heap.hpp"

#define SUB_LIST_SIZE 8ul
#define COS_SUB_RATIO 1.5

namespace tribase {

Index::Index(size_t d, size_t nlist, size_t nprobe, MetricType metric, OptLevel opt_level, size_t sub_k, size_t sub_nlist, size_t sub_nprobe)
    : d(d), nlist(nlist), nprobe(nprobe), metric(metric), opt_level(opt_level), sub_k(sub_k), sub_nlist(sub_nlist), sub_nprobe(sub_nprobe) {
    lists = std::make_unique<IVF[]>(nlist);
    centroid_codes = std::make_unique<float[]>(nlist * d);
    centroid_ids = std::make_unique<idx_t[]>(nprobe);
    std::iota(centroid_ids.get(), centroid_ids.get() + nlist, 0);
}

void Index::train(size_t n, const float* codes) {
    // 这里假设Clustering类已经定义好，并且有一个合适的构造函数和train方法
    ClusteringParameters cp;
    cp.metric = this->metric;
    cp.niter = 25;                     // 或其他合适的值
    cp.seed = 6666;                    // 或其他合适的值
    cp.max_points_per_centroid = 256;  // 或其他合适的值

    Clustering clustering(this->d, this->nlist, cp);
    clustering.train(n, codes);

    this->centroid_codes.reset(clustering.get_centroids());
}

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
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_NONE>(d, k));
            case OptLevel::OPT_TRIANGLE:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_TRIANGLE>(d, k));
            case OptLevel::OPT_SUBNN_L2:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_L2>(d, k));
            case OptLevel::OPT_SUBNN_IP:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_IP>(d, k));
            case OptLevel::OPT_TRI_SUBNN_L2:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_TRI_SUBNN_L2>(d, k));
            case OptLevel::OPT_TRI_SUBNN_IP:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_TRI_SUBNN_IP>(d, k));
            case OptLevel::OPT_ALL:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_ALL>(d, k));
        }
    }
};

void Index::single_thread_nearest_cluster_search(size_t n, const float* queries, float* distances, idx_t* labels) {
    std::unique_ptr<IVFScanBase> scaner_quantizer = get_scaner(metric, OPT_NONE, sub_k);
    for (size_t i = 0; i < n; i++) {
        scaner_quantizer->set_query(queries + i * d);
        scaner_quantizer->lite_scan_codes(nlist, centroid_codes.get(), centroid_ids.get(), distances + i, labels + i);
    }
}

void Index::add(size_t n, const float* codes) {
    std::unique_ptr<float[]> candicate2centroid = std::make_unique<float[]>(n);
    std::unique_ptr<idx_t[]> listidcandicates = std::make_unique<idx_t[]>(n);
    init_result(metric, n, candicate2centroid.get(), listidcandicates.get());
    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), n);
    size_t batch_size = (n + nt - 1) / nt;
#pragma omp parallel for num_threads(nt)
    for (size_t i = 0; i < nt; i++) {
        size_t start = i * batch_size;
        size_t end = std::min(start + batch_size, n);
        single_thread_nearest_cluster_search(end - start, codes + start * d, candicate2centroid.get() + start, listidcandicates.get() + start);
    }

    size_t list_sizes[nlist];
    std::fill_n(list_sizes, nlist, 0);

#pragma omp parallel for reduction(+ : list_sizes[ : nlist])
    for (size_t i = 0; i < n; i++) {
        list_sizes[listidcandicates[i]]++;
    }

#pragma omp parallel for
    for (size_t i = 0; i < nlist; i++) {
        lists[i].reset(list_sizes[i], d, sub_k);
    }

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();

        for (size_t i = 0; i < n; i++) {
            size_t list_id = listidcandicates[i];
            if (list_id % nt == tid) {
                size_t list_size = list_sizes[list_id];
                lists[list_id].candidate_id[list_size] = i;
                lists[list_id].candidate2centroid[list_size] = candicate2centroid[i];
                lists[list_id].sqrt_candidate2centroid[list_size] = std::sqrt(candicate2centroid[i]);
                std::copy_n(codes + i * d, d, lists[list_id].candidate_codes.get() + list_size * d);
                list_sizes[list_id]++;
            }
        }
    }

    {
        size_t total_processd = 0;
        size_t total_add = 0;
        size_t target_add = 0;

        size_t true_train_count = 0;
        size_t all_train_count = 0;

        size_t total_sub_count_cos = 0;
        size_t total_sub_recall_cos = 0;
        size_t total_sub_count_l2 = 0;
        size_t total_sub_recall_l2 = 0;
        // size_t total_sub_count_cos_point_10 = 0;
        size_t total_sub_recall_cos_point_10 = 0;
        // size_t total_sub_count_l2_point_10 = 0;
        size_t total_sub_recall_l2_point_10 = 0;

        Stopwatch logwatch;
        if (opt_level & OptLevel::OPT_SUBNN_IP) {
            target_add += n;
        }
        if (opt_level & OptLevel::OPT_SUBNN_L2) {
            target_add += n;
        }
        const size_t ADD_BATCH_SIZE = 10000;

        double train_elapsed = 0;
        double add_elapsed = 0;
        double search_elapsed = 0;
        double log_interval = 2;

        auto running_log = [&]() -> void {
            auto now = std::chrono::system_clock::now();
            if (logwatch.elapsedSeconds() > log_interval || total_add == target_add) {
                logwatch.reset();
                printf("add: %.3f%%    build: %.2f%%\n", 100.0 * total_add / target_add, 100.0 * total_processd / nlist);
                double total_elapsed = train_elapsed + add_elapsed + search_elapsed;
                double train_percent = 100.0 * train_elapsed / total_elapsed;
                double add_percent = 100.0 * add_elapsed / total_elapsed;
                double search_percent = 100.0 * search_elapsed / total_elapsed;
                printf("train: %.2f%% (%.2f%%)    add: %.2f%%    search: %.2f%%    total: %.2f\n",
                       train_percent,
                       all_train_count ? 100.0 * true_train_count / all_train_count : 1,
                       add_percent,
                       search_percent,
                       total_elapsed);
                float sub_recall_cos = total_sub_count_cos ? 100.0 * total_sub_recall_cos / total_sub_count_cos : 0;
                float sub_recall_l2 = total_sub_count_l2 ? 100.0 * total_sub_recall_l2 / total_sub_count_l2 : 0;
                float sub_recall_cos_point_10 = total_sub_count_cos ? 1000.0 * total_sub_recall_cos_point_10 / total_sub_count_cos : 0;
                float sub_recall_l2_point_10 = total_sub_count_l2 ? 1000.0 * total_sub_recall_l2_point_10 / total_sub_count_l2 : 0;
                printf("Recall    cos 1/10: %.2f%%    cos: %.2f%%    l2 1/10: %.2f%%    l2: %.2f%%    %d/%d\n",
                       sub_recall_cos_point_10,
                       sub_recall_cos,
                       sub_recall_l2_point_10,
                       sub_recall_l2,
                       sub_nprobe,
                       sub_nlist);
            }
        };

        auto end_log = [&]() -> void {
            double total_elapsed = train_elapsed + add_elapsed + search_elapsed;
            printf("add: 100.0%%    build: 100.0%%\n");
            printf("train: %.2f (%.2f%%)    add: %.2f    search: %.2f    total: %.2f\n",
                   train_elapsed,
                   100.0 * true_train_count / all_train_count,
                   add_elapsed,
                   search_elapsed,
                   total_elapsed);
        };

#pragma omp parallel for
        for (size_t listid = 0; listid < nlist; listid++) {
            IVF& list = lists[listid];
            const float* xb = list.get_candidate_codes();
            size_t nb = list.get_list_size();

            size_t this_sub_nlist_L2 = std::min(sub_nlist, (nb + SUB_LIST_SIZE - 1) / SUB_LIST_SIZE);
            size_t this_sub_nlist_cos = std::min(std::max(1ul, static_cast<size_t>(sub_nlist)), static_cast<size_t>((nb + SUB_LIST_SIZE - 1) / SUB_LIST_SIZE));
            size_t this_sub_nprobe_L2 = std::min(std::max(1ul, static_cast<size_t>(1.0 * this_sub_nlist_L2 * sub_nprobe / sub_nlist)), this_sub_nlist_L2);
            size_t this_sub_nprobe_cos = std::min(std::max(1ul, static_cast<size_t>(1.0 * this_sub_nlist_cos * sub_nprobe / sub_nlist * COS_SUB_RATIO)), this_sub_nlist_cos);

            std::unique_ptr<float[]> norm_xb = std::make_unique<float[]>(nb * d);
            const float* centroid_code = centroid_codes.get() + listid * d;

            for (size_t j = 0; j < nb; j++) {
                float norm_xb_value = 0;
                const float* x = xb + j * d;
                for (size_t k = 0; k < d; k++) {
                    norm_xb[j * d + k] = (x[k] - centroid_code[k]);
                    norm_xb_value += norm_xb[j * d + k] * norm_xb[j * d + k];
                }

                norm_xb_value = sqrt(norm_xb_value);
                if (norm_xb_value > 0) {
                    for (size_t k = 0; k < d; k++) {
                        norm_xb[j * d + k] /= norm_xb_value;
                    }
                }
            }

            if (opt_level & OptLevel::OPT_SUBNN_L2) {
                Index sub_index(d, this_sub_nlist_L2, this_sub_nprobe_L2, MetricType::METRIC_L2, OptLevel::OPT_NONE, sub_k);
                sub_index.train(nb, norm_xb.get());
                sub_index.add(nb, norm_xb.get());
                sub_index.search(nb, norm_xb.get(), sub_k, list.sub_nearest_L2_dis.get(), list.sub_nearest_L2_id.get());
            }

            if (opt_level & OptLevel::OPT_SUBNN_IP) {
                Index sub_index(d, this_sub_nlist_cos, this_sub_nprobe_cos, MetricType::METRIC_IP, OptLevel::OPT_NONE, sub_k);
                sub_index.train(nb, norm_xb.get());
                sub_index.add(nb, norm_xb.get());
                sub_index.search(nb, norm_xb.get(), sub_k, list.sub_nearest_IP_dis.get(), list.sub_nearest_IP_id.get());

                // -x
                for (size_t j = 0; j < nb * d; j++) {
                    norm_xb[j] = -norm_xb[j];
                }

                sub_index.search(nb, norm_xb.get(), sub_k, list.sub_farest_IP_dis.get(), list.sub_farest_IP_id.get());

                // -dis
                for (size_t j = 0; j < nb; j++) {
                    list.sub_farest_IP_dis[j] = -list.sub_farest_IP_dis[j];
                }
            }
        }
    }

}  // namespace tribase

void Index::single_thread_search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, Stats* stats) {
    std::unique_ptr<IVFScanBase> scaner_quantizer = get_scaner(metric, OPT_NONE, sub_k);
    std::unique_ptr<IVFScanBase> scaner = get_scaner(metric, opt_level, k);

    std::unique_ptr<float[]> centroid2queries = std::make_unique<float[]>(n * nprobe);
    std::unique_ptr<idx_t[]> listidqueries = std::make_unique<idx_t[]>(n * nprobe);
    init_result(metric, n * nprobe, centroid2queries.get(), listidqueries.get());

    float* simi = distances;
    idx_t* idxi = labels;
    float* centroids2query = centroid2queries.get();
    idx_t* listids = listidqueries.get();

    for (size_t i = 0; i < n; i++) {
        scaner_quantizer->set_query(queries + i * d);
        scaner->set_query(queries + i * d);
        scaner_quantizer->lite_scan_codes(nlist, centroid_codes.get(), centroid_ids.get(), centroids2query, listids);
        for (size_t j = 0; j < nprobe; j++) {
            IVF& list = lists[listids[j]];
            float centroid2query = centroids2query[j];
            size_t list_size = list.get_list_size();

            const float* codes = list.get_codes();
            const idx_t* ids = reinterpret_cast<const idx_t*>(list.get_ids());
            std::unique_ptr<bool[]> if_skip = std::make_unique<bool[]>(list_size);

            size_t skip_count = 0;
            size_t skip_count_large = 0;
            size_t scan_begin = 0;
            size_t scan_end = list_size;

            if (opt_level & OptLevel::OPT_TRIANGLE) {
                const float* sqrt_candidate2centroid = list.get_sqrt_candidate2centroid();
                const float* candidate2centroid = list.get_candidate2centroid();
                float sqrt_simi = sqrt(simi[0]);
                float sqrt_centroid2query = sqrt(centroid2query);
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

            IF_STATS {
                stats->skip_triangle_count += skip_count;
                stats->skip_triangle_large_count += skip_count_large;
                stats->total_count += list_size;
            }

            scaner->scan_codes(scan_begin, scan_end, list_size, codes, ids, centroid2query, list.get_candidate2centroid(),
                               list.get_sqrt_candidate2centroid(), sub_k, list.get_sub_nearest_IP_id(),
                               list.get_sub_nearest_IP_dis(), list.get_sub_farest_IP_id(), list.get_sub_farest_IP_dis(),
                               list.get_sub_nearest_L2_id(), list.get_sub_nearest_L2_dis(), if_skip.get(), simi, idxi,
                               stats);
        }

        simi += k;
        idxi += k;
        centroids2query += nprobe;
        listids += nprobe;
    }
}

void Index::search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels) {
    init_result(metric, n * k, distances, labels);
    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), n);
    size_t batch_size = (n + nt - 1) / nt;
    std::vector<Stats> stats(nt);

#pragma omp parallel for num_threads(nt)
    for (int i = 0; i < nt; i++) {
        size_t start = i * batch_size;
        size_t end = std::min(start + batch_size, n);
        single_thread_search(end - start, queries + start * d, k, distances + start * k, labels + start * k, &stats[i]);
    }

    Stats total_stats = mergeStats(stats);
}

}  // namespace tribase