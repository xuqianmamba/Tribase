#include "Index.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <omp.h>
#include <atomic>
#include <cmath>
#include <memory>
#include "IVF.h"
#include "IVFScan.hpp"
#include "heap.hpp"

#define SUB_LIST_SIZE 8ul
#define IP_SUB_RATIO 1.5
#define RECALL_TEST_RATIO 0.1

namespace tribase {

Index::Index(size_t d, size_t nlist, size_t nprobe, MetricType metric, OptLevel opt_level, size_t sub_k, size_t sub_nlist, size_t sub_nprobe, bool verbose)
    : d(d), nlist(nlist), nprobe(nprobe), metric(metric), opt_level(opt_level), sub_k(sub_k), sub_nlist(sub_nlist), sub_nprobe(sub_nprobe), verbose(verbose) {
    lists = std::make_unique<IVF[]>(nlist);
    centroid_codes = std::make_unique<float[]>(nlist * d);
    centroid_ids = std::make_unique<idx_t[]>(nlist);
    std::iota(centroid_ids.get(), centroid_ids.get() + nlist, 0);
}

Index& Index::operator=(Index&& other) noexcept {
    d = other.d;
    nlist = other.nlist;
    nprobe = other.nprobe;
    metric = other.metric;
    opt_level = other.opt_level;
    added_opt_level = other.added_opt_level;
    sub_k = other.sub_k;
    sub_nlist = other.sub_nlist;
    sub_nprobe = other.sub_nprobe;
    verbose = other.verbose;
    lists = std::move(other.lists);
    centroid_codes = std::move(other.centroid_codes);
    centroid_ids = std::move(other.centroid_ids);
    return *this;
}

void Index::train(size_t n, const float* codes, bool faiss) {
    // 这里假设Clustering类已经定义好，并且有一个合适的构造函数和train方法
    if (!faiss) {
        ClusteringParameters cp;
        cp.metric = this->metric;
        cp.niter = 25;                     // 或其他合适的值
        cp.seed = 6666;                    // 或其他合适的值
        cp.max_points_per_centroid = 256;  // 或其他合适的值

        Clustering clustering(this->d, this->nlist, verbose, cp);
        clustering.train(n, codes);

        this->centroid_codes.reset(clustering.get_centroids());
    } else {
        ::faiss::IndexFlatL2 quantizer(d);  // the other index
        ::faiss::IndexIVFFlat index(&quantizer, d, nlist);
        index.train(n, codes);
        this->centroid_codes = std::make_unique<float[]>(nlist * d);
        std::copy_n(quantizer.get_xb(), nlist * d, this->centroid_codes.get());
    }
}

std::unique_ptr<IVFScanBase> Index::get_scanner(MetricType metric, OptLevel opt_level, size_t k) {
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
            default:
                throw std::runtime_error("Unsupported opt_level");
        }
    } else {
        switch (opt_level) {
            case OptLevel::OPT_NONE:
                return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_NONE>(d, k));
            // case OptLevel::OPT_TRIANGLE:
            //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_TRIANGLE>(d, k));
            // case OptLevel::OPT_SUBNN_L2:
            //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_L2>(d, k));
            // case OptLevel::OPT_SUBNN_IP:
            //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_SUBNN_IP>(d, k));
            // case OptLevel::OPT_TRI_SUBNN_L2:
            //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_TRI_SUBNN_L2>(d, k));
            // case OptLevel::OPT_TRI_SUBNN_IP:
            //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_TRI_SUBNN_IP>(d, k));
            // case OptLevel::OPT_ALL:
            //     return std::unique_ptr<IVFScanBase>(new IVFScan<MetricType::METRIC_IP, OptLevel::OPT_ALL>(d, k));
            default:
                throw std::runtime_error("Unsupported opt_level");
        }
    }
};

void Index::single_thread_nearest_cluster_search(size_t n, const float* queries, float* distances, idx_t* labels) {
    if (n == 0) {
        return;
    }
    std::unique_ptr<IVFScanBase> scaner_quantizer = get_scanner(metric, OPT_NONE, 1);
    for (size_t i = 0; i < n; i++) {
        scaner_quantizer->set_query(queries + i * d);
        scaner_quantizer->lite_scan_codes(nlist,
                                          centroid_codes.get(),
                                          reinterpret_cast<const size_t*>(centroid_ids.get()),
                                          distances + i,
                                          labels + i);
        // no need to sort result, because only one result
    }
}

void Index::add(size_t n, const float* codes) {
    if (n == 0) {
        return;
    }

    added_opt_level = opt_level;
    std::unique_ptr<float[]> candicate2centroid = std::make_unique<float[]>(n);
    std::unique_ptr<idx_t[]> listidcandicates = std::make_unique<idx_t[]>(n);
    init_result(metric, n, candicate2centroid.get(), listidcandicates.get());
    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), n);
    size_t batch_size = n / nt;
    size_t extra = n % nt;
#pragma omp parallel for num_threads(nt)
    for (size_t i = 0; i < nt; i++) {
        size_t start, end;
        if (i < extra) {
            start = i * (batch_size + 1);
            end = start + batch_size + 1;
        } else {
            start = i * batch_size + extra;
            end = start + batch_size;
        }
        if (start < end) {
            single_thread_nearest_cluster_search(end - start, codes + start * d, candicate2centroid.get() + start, listidcandicates.get() + start);
        }
    }

    size_t list_sizes[nlist];
    std::fill_n(list_sizes, nlist, 0);

#pragma omp parallel for reduction(+ : list_sizes[ : nlist])
    for (size_t i = 0; i < n; i++) {
        list_sizes[listidcandicates[i]]++;
    }

    size_t total_add = 0;
    for (size_t i = 0; i < nlist; i++) {
        total_add += list_sizes[i];
    }

#pragma omp parallel for
    for (size_t i = 0; i < nlist; i++) {
        lists[i].reset(list_sizes[i], d, sub_k, added_opt_level);
    }

    std::fill_n(list_sizes, nlist, 0);

    std::unique_ptr<size_t[]> add_order = std::make_unique<size_t[]>(n);
    std::iota(add_order.get(), add_order.get() + n, 0);
    if (metric == MetricType::METRIC_L2) {
        std::sort(add_order.get(), add_order.get() + n, [&](size_t i, size_t j) { return candicate2centroid[i] < candicate2centroid[j]; });
    } else {
        std::sort(add_order.get(), add_order.get() + n, [&](size_t i, size_t j) { return candicate2centroid[i] > candicate2centroid[j]; });
    }

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();

        for (size_t oi = 0; oi < n; oi++) {
            size_t i = add_order[oi];
            int list_id = listidcandicates[i];
            if (list_id % nt == tid) {
                size_t list_size = list_sizes[list_id];
                lists[list_id].candidate_id[list_size] = i;
                if ((opt_level & OptLevel::OPT_TRIANGLE) || (opt_level & OptLevel::OPT_SUBNN_IP)) {
                    lists[list_id].candidate2centroid[list_size] = candicate2centroid[i];
                    lists[list_id].sqrt_candidate2centroid[list_size] = std::sqrt(candicate2centroid[i]);
#ifdef CORRECTNESS_CHECK
                    if (list_size > 0) {
                        if (metric == MetricType::METRIC_L2) {
                            assert(candicate2centroid[i] >= candicate2centroid[lists[list_id].candidate_id[list_size - 1]]);
                        } else {
                            assert(candicate2centroid[i] <= candicate2centroid[lists[list_id].candidate_id[list_size - 1]]);
                        }
                    }
#endif
                }
                std::copy_n(codes + i * d, d, lists[list_id].candidate_codes.get() + list_size * d);
                lists[list_id].candidate_norms[list_size] = calculatedInnerProduct(codes + i * d, codes + i * d, d);
                list_sizes[list_id]++;

                // if (metric == MetricType::METRIC_L2) {
                //     const float* x = codes + i * d;
                //     const float* centroid_code = centroid_codes.get() + list_id * d;
                //     float dis = calculatedEuclideanDistance(x, centroid_code, d);
                //     assert(dis == candicate2centroid[i]);
                // }
            }
        }
    }

    {
        size_t total_processd = 0;

        size_t total_sub_count_ip = 0;
        size_t total_sub_recall_ip = 0;
        size_t total_sub_count_l2 = 0;
        size_t total_sub_recall_l2 = 0;
        size_t total_sub_count_ip_5 = 0;
        size_t total_sub_recall_ip_5 = 0;
        size_t total_sub_count_l2_5 = 0;
        size_t total_sub_recall_l2_5 = 0;

        Stopwatch logwatch;

        double train_elapsed = 0;
        double add_elapsed = 0;
        double search_elapsed = 0;
        double log_interval = 2;

        [[maybe_unused]] auto running_log = [&]() -> void {
            if (verbose) {
                if (logwatch.elapsedSeconds() > log_interval || total_processd == nlist) {
                    logwatch.reset();
                    std::cout << std::format("build: {:.2f}%", 100.0 * total_processd / nlist) << std::endl;
                    double total_elapsed = train_elapsed + add_elapsed + search_elapsed;
                    double train_percent = 100.0 * train_elapsed / total_elapsed;
                    double add_percent = 100.0 * add_elapsed / total_elapsed;
                    double search_percent = 100.0 * search_elapsed / total_elapsed;
                    std::cout << std::format("train: {:.2f}%    add: {:.2f}%    search: {:.2f}%    total: {:.2f}\n", train_percent, add_percent, search_percent, total_elapsed);
                    float sub_recall_ip = total_sub_count_ip ? 100.0 * total_sub_recall_ip / total_sub_count_ip : 0;
                    float sub_recall_l2 = total_sub_count_l2 ? 100.0 * total_sub_recall_l2 / total_sub_count_l2 : 0;
                    float sub_recall_ip_5 = total_sub_count_ip_5 ? 100.0 * total_sub_recall_ip_5 / total_sub_count_ip_5 : 0;
                    float sub_recall_l2_5 = total_sub_count_l2_5 ? 100.0 * total_sub_recall_l2_5 / total_sub_count_l2_5 : 0;
                    std::cout << std::format("Recall    SUBNN_IP top5: {:.2f}%    topk: {:.2f}%    SUBNN_L2 top5: {:.2f}%    topk: {:.2f}%    {}/{}\n",
                                             sub_recall_ip_5, sub_recall_ip, sub_recall_l2_5, sub_recall_l2, sub_nprobe, sub_nlist);
                }
            }
        };

        [[maybe_unused]] auto end_log = [&]() -> void {
            if (verbose) {
                double total_elapsed = train_elapsed + add_elapsed + search_elapsed;
                std::cout << std::format("build: 100.0%\n");
                std::cout << std::format("train: {:.2f}   add: {:.2f}    search: {:.2f}    total: {:.2f}\n",
                                         train_elapsed,
                                         add_elapsed,
                                         search_elapsed,
                                         total_elapsed);
            }
        };

#pragma omp parallel for
        for (size_t listid = 0; listid < nlist; listid++) {
            IVF& list = lists[listid];
            const float* xb = list.get_candidate_codes();
            size_t nb = list.get_list_size();

            size_t this_sub_nlist_L2 = std::min(sub_nlist, (nb + SUB_LIST_SIZE - 1) / SUB_LIST_SIZE);
            size_t this_sub_nprobe_L2 = std::min(std::max(1ul, static_cast<size_t>(1.0 * this_sub_nlist_L2 * sub_nprobe / sub_nlist)), this_sub_nlist_L2);

            size_t this_sub_nlist_IP = std::min(std::max(1ul, static_cast<size_t>(sub_nlist)), static_cast<size_t>((nb + SUB_LIST_SIZE - 1) / SUB_LIST_SIZE));
            size_t this_sub_nprobe_IP = std::min(std::max(1ul, static_cast<size_t>(1.0 * this_sub_nlist_IP * sub_nprobe / sub_nlist * IP_SUB_RATIO)), this_sub_nlist_IP);

            const float* centroid_code = centroid_codes.get() + listid * d;

            if (opt_level & OptLevel::OPT_SUBNN_L2) {
                Index sub_index(d, this_sub_nlist_L2, this_sub_nprobe_L2, MetricType::METRIC_L2, OptLevel::OPT_NONE, 0, 0, 0, false);
                Stopwatch watch;
                sub_index.train(nb, xb);
#pragma omp atomic
                train_elapsed += watch.elapsedSeconds(true);
                sub_index.add(nb, xb);
#pragma omp atomic
                add_elapsed += watch.elapsedSeconds(true);
                sub_index.search(nb, xb, sub_k, list.sub_nearest_L2_dis.get(), list.sub_nearest_L2_id.get());
#pragma omp atomic
                search_elapsed += watch.elapsedSeconds(true);

                if (true) {
                    size_t recall_nb = static_cast<size_t>(1.0 * nb * RECALL_TEST_RATIO / sub_nlist * sub_nprobe);
                    std::unique_ptr<float[]> recall_dis = std::make_unique<float[]>(recall_nb * sub_k);
                    std::unique_ptr<idx_t[]> recall_id = std::make_unique<idx_t[]>(recall_nb * sub_k);
                    sub_index.nprobe = sub_index.nlist;
                    sub_index.search(recall_nb, xb, sub_k, recall_dis.get(), recall_id.get());

                    for (size_t j = 0; j < recall_nb; j++) {
                        float top_recall_dis = recall_dis[j * sub_k + sub_k - 1];
                        float top_recall_dis_5 = recall_dis[j * sub_k + std::min(4ul, sub_k - 1)];
                        for (size_t k = 0; k < sub_k; k++) {
                            float dis = list.get_sub_nearest_L2_dis(j, k);
                            if (dis <= top_recall_dis) {
                                total_sub_recall_l2++;
                                if (dis <= top_recall_dis_5 && k < 5) {
                                    total_sub_recall_l2_5++;
                                }
                            }
                        }
                    }
                    total_sub_count_l2 += recall_nb * sub_k;
                    total_sub_count_l2_5 += recall_nb * std::min(5ul, sub_k);
                }

                for (size_t j = 0; j < nb * sub_k; j++) {
                    list.sub_nearest_L2_dis[j] = sqrt(list.sub_nearest_L2_dis[j]);
                }
            }

            if (opt_level & OptLevel::OPT_SUBNN_IP) {
                std::unique_ptr<float[]> norm_xb_u = std::make_unique<float[]>(nb * d);
                float* norm_xb = norm_xb_u.get();
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
                Index sub_index(d, this_sub_nlist_IP, this_sub_nprobe_IP, MetricType::METRIC_IP, OptLevel::OPT_NONE, 0, 0, 0, false);
                Stopwatch watch;
                sub_index.train(nb, norm_xb);
#pragma omp atomic
                train_elapsed += watch.elapsedSeconds(true);
                sub_index.add(nb, norm_xb);
#pragma omp atomic
                add_elapsed += watch.elapsedSeconds(true);
                sub_index.search(nb, norm_xb, sub_k, list.sub_nearest_IP_dis.get(), list.sub_nearest_IP_id.get());
#pragma omp atomic
                search_elapsed += watch.elapsedSeconds(true);

                for (size_t j = 0; j < nb * d; j++) {
                    norm_xb[j] = -norm_xb[j];
                }

                watch.reset();
                sub_index.search(nb, norm_xb, sub_k, list.sub_farest_IP_dis.get(), list.sub_farest_IP_id.get());
                search_elapsed += watch.elapsedSeconds(true);

                for (size_t j = 0; j < nb * sub_k; j++) {
                    list.sub_farest_IP_dis[j] = -list.sub_farest_IP_dis[j];
                }

                if (true) {
                    size_t recall_nb = static_cast<size_t>(1.0 * nb * RECALL_TEST_RATIO / sub_nlist * sub_nprobe);
                    std::unique_ptr<float[]> recall_dis = std::make_unique<float[]>(recall_nb * sub_k);
                    std::unique_ptr<idx_t[]> recall_id = std::make_unique<idx_t[]>(recall_nb * sub_k);
                    sub_index.nprobe = sub_index.nlist;
                    sub_index.search(recall_nb, norm_xb, sub_k, recall_dis.get(), recall_id.get());

                    for (size_t j = 0; j < recall_nb; j++) {
                        float top_recall_dis = recall_dis[j * sub_k + sub_k - 1];
                        float top_recall_dis_5 = recall_dis[j * sub_k + std::min(4ul, sub_k - 1)];
                        for (size_t k = 0; k < sub_k; k++) {
                            float dis = list.get_sub_nearest_IP_dis(j, k);
                            if (dis >= top_recall_dis) {
                                total_sub_recall_ip++;
                                if (dis >= top_recall_dis_5 && k < 5) {
                                    total_sub_recall_ip_5++;
                                }
                            }
                        }
                    }
                    total_sub_count_ip += recall_nb * sub_k;
                    total_sub_count_ip_5 += recall_nb * std::min(5ul, sub_k);
                }
            }

#pragma omp critical
            {
                total_processd++;
                running_log();
            }
        }
        end_log();
    }
}

void Index::single_thread_search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels, Stats* stats) {
    std::unique_ptr<IVFScanBase> scaner_quantizer = get_scanner(metric, OPT_NONE, nprobe);
    std::unique_ptr<IVFScanBase> scaner = get_scanner(metric, opt_level, k);

    std::unique_ptr<float[]> centroid2queries = std::make_unique<float[]>(n * nprobe);
    std::unique_ptr<idx_t[]> listidqueries = std::make_unique<idx_t[]>(n * nprobe);
    init_result(metric, n * nprobe, centroid2queries.get(), listidqueries.get());

    float* simi = distances;
    idx_t* idxi = labels;
    float* centroids2query = centroid2queries.get();
    idx_t* listids = listidqueries.get();

    assert(scaner->k == k);

    for (size_t i = 0; i < n; i++) {
        scaner_quantizer->set_query(queries + i * d);
        scaner->set_query(queries + i * d);
        scaner_quantizer->lite_scan_codes(nlist,
                                          centroid_codes.get(),
                                          reinterpret_cast<const size_t*>(centroid_ids.get()),
                                          centroids2query,
                                          listids);
        sort_result(metric, nprobe, centroids2query, listids);
        for (size_t j = 0; j < nprobe; j++) {
            IVF& list = lists[listids[j]];
            float centroid2query = centroids2query[j];
            size_t list_size = list.get_list_size();

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

                for (int64_t ii = list_size - 1; ii >= 0; ii--) {
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

            scaner->scan_codes(scan_begin, scan_end, list_size, list.get_candidate_codes(), list.get_candidate_id(), list.get_candidate_norms(), centroid2query, list.get_candidate2centroid(),
                               list.get_sqrt_candidate2centroid(), sub_k, list.get_sub_nearest_IP_id(),
                               list.get_sub_nearest_IP_dis(), list.get_sub_farest_IP_id(), list.get_sub_farest_IP_dis(),
                               list.get_sub_nearest_L2_id(), list.get_sub_nearest_L2_dis(), if_skip.get(), simi, idxi,
                               stats, centroid_codes.get() + listids[j] * d);
        }
        sort_result(metric, k, simi, idxi);

        simi += k;
        idxi += k;
        centroids2query += nprobe;
        listids += nprobe;
    }
}

Stats Index::search(size_t n, const float* queries, size_t k, float* distances, idx_t* labels) {
    if (n == 0) {
        return Stats();
    }
    if ((opt_level & added_opt_level) != opt_level) {
        throw std::runtime_error("opt_level is not subset of added_opt_level");
    }
    if (nprobe > nlist) {
        nprobe = nlist;
    }
    init_result(metric, n * k, distances, labels);
    size_t nt = std::min(static_cast<size_t>(omp_get_max_threads()), n);
    size_t batch_size = n / nt;
    size_t extra = n % nt;
    std::vector<Stats> stats(nt);

#pragma omp parallel for num_threads(nt)
    for (size_t i = 0; i < nt; i++) {
        size_t start, end;
        if (i < extra) {
            start = i * (batch_size + 1);
            end = start + batch_size + 1;
        } else {
            start = i * batch_size + extra;
            end = start + batch_size;
        }
        if (start < end) {
            single_thread_search(end - start, queries + start * d, k, distances + start * k, labels + start * k, &stats[i]);
        }
    }

    [[maybe_unused]] Stats total_stats = mergeStats(stats);
    return total_stats;
}

void Index::save_index(std::string path) const {
    prepareDirectory(path);
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file " + path);
    }
    out.write(reinterpret_cast<const char*>(&d), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&nlist), sizeof(size_t));
    // out.write(reinterpret_cast<const char*>(&nprobe), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&metric), sizeof(MetricType));
    out.write(reinterpret_cast<const char*>(&added_opt_level), sizeof(OptLevel));
    out.write(reinterpret_cast<const char*>(&sub_k), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&sub_nlist), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&sub_nprobe), sizeof(size_t));

    out.write(reinterpret_cast<const char*>(centroid_codes.get()), nlist * d * sizeof(float));
    // out.write(reinterpret_cast<const char*>(centroid_ids.get()), nlist * sizeof(idx_t)); // 0 ~ nlist-1

    for (size_t i = 0; i < nlist; i++) {
        if (lists[i].get_list_size() > 0) {
            out.write(reinterpret_cast<const char*>(&i), sizeof(size_t));
            lists[i].save_IVF(out);
        }
    }
}

void Index::load_index(std::string path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file " + path);
    }
    in.read(reinterpret_cast<char*>(&d), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&nlist), sizeof(size_t));
    // in.read(reinterpret_cast<char*>(&nprobe), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&metric), sizeof(MetricType));
    in.read(reinterpret_cast<char*>(&added_opt_level), sizeof(OptLevel));
    opt_level = OptLevel::OPT_NONE;
    in.read(reinterpret_cast<char*>(&sub_k), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&sub_nlist), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&sub_nprobe), sizeof(size_t));

    centroid_codes = std::make_unique<float[]>(nlist * d);
    in.read(reinterpret_cast<char*>(centroid_codes.get()), nlist * d * sizeof(float));
    centroid_ids = std::make_unique<idx_t[]>(nlist);
    std::iota(centroid_ids.get(), centroid_ids.get() + nlist, 0);

    lists = std::make_unique<IVF[]>(nlist);
    Stopwatch watch;
    while (true) {
        size_t listid;
        in.read(reinterpret_cast<char*>(&listid), sizeof(size_t));
        if (in.eof()) {
            break;
        }
        lists[listid].load_IVF(in);
        if (watch.elapsedSeconds() > 2) {
            watch.reset();
            std::cout << std::format("loaded list {}/{}, size {}", listid, nlist, lists[listid].get_list_size()) << std::endl;
        }
    }
}

}  // namespace tribase