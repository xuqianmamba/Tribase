#include "IndexIVFwithDistance.h"
#include <faiss/IndexIVF.h>

#include <omp.h>
#include <cstdint>
#include <memory>
#include <mutex>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <memory>

#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <fstream>

#include <faiss/mycommon.h>
#include <faiss/utils/distances.h>
#include <random>

#define SUB_LIST_SIZE 8ul
#define RECALL_RATIO 0.05
#define COS_SUB_RATIO 2

namespace faiss {

std::mutex mtx;
IndexIVFwithDistance::IndexIVFwithDistance(
    IndexFlatL2* quantizer,
    size_t d,
    size_t nlist,
    MetricType metric,
    size_t sub_k,
    size_t sub_nlist,
    size_t sub_nprobe,
    float sub_sample)
    : IndexIVFFlat(quantizer, d, nlist, metric), sub_k(sub_k), sub_nlist(sub_nlist), sub_nprobe(sub_nprobe), sub_sample(sub_sample) {
    code_size = sizeof(float) * d;
    by_residual = false;

    // 初始化InvertedListswithDistance对象
    invlistswithdist = new InvertedListswithDistance(nlist, code_size);

    // // 为 quantizer 中的 distances 字段分配空间
    // quantizer->distances = new float[nlist];
    enable_triangle = !std::getenv("DISABLE_TRIANGLE");
    enable_intersection = !std::getenv("DISABLE_INTERSECTION");
    enable_sub_knn_cos = !std::getenv("DISABLE_SUB_KNN_COS");
    enable_sub_knn_l2 = !std::getenv("DISABLE_SUB_KNN_L2");
    enable_lite_sub_knn = std::getenv("ENABLE_LITE_SUB_KNN");

    const char* log_interval_str = std::getenv("LOG_INTERVAL");
    if (log_interval_str) {
        log_interval = std::stof(log_interval_str);
    } else {
        log_interval = 0.5;
    }
}

IndexIVFwithDistance::~IndexIVFwithDistance() {
    // 删除InvertedListswithDistance对象
    delete invlistswithdist;
}

void IndexIVFwithDistance::assign(idx_t n, const float* x, idx_t* labels, idx_t k, float* dis2nearest_center) const {
    std::vector<float> distances(n * k);

    IndexIVFFlat::search(n, x, k, distances.data(), labels);

#pragma omp parallel for
    for (idx_t i = 0; i < n; ++i) {
        dis2nearest_center[i] = distances[i * k];
    }
}

void IndexIVFwithDistance::add_core(
    idx_t n,
    const float* x,
    const idx_t* xids,
    const idx_t* coarse_idx,
    const float* dis2nearest_center) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(!by_residual);
    assert(invlistswithdist);
    direct_map.check_can_add(xids);

    int64_t n_add = 0;

    DirectMapAdd dm_adder(direct_map, n, xids);

#pragma omp parallel reduction(+ : n_add)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];

            // 打印list_no和nt的值
            // std::cout << "list_no: " << list_no << ", nt: " << nt << ", rank: " << rank << std::endl;

            if (list_no >= 0 && list_no % nt == rank) {
                // std::cout<<"add_entry"<<std::endl;
                idx_t id = xids ? xids[i] : ntotal + i;
                const float* xi = x + i * d;
                size_t offset = invlistswithdist->add_entry(list_no, id, (const uint8_t*)xi, dis2nearest_center[i]);
                dm_adder.add(i, list_no, offset);
                n_add++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }

    if (verbose) {
        printf("IndexIVFwithDistance::add_core: added %" PRId64 " / %" PRId64
               " vectors\n",
               n_add,
               n);
    }
    ntotal += n;

    invlistswithdist->sort_all_lists();
}

void IndexIVFwithDistance::add_with_ids(
    idx_t n,
    const float* x,
    const idx_t* xids) {
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
    std::unique_ptr<float[]> dis2nearest_center(new float[n]);
    // 确保quantizer确实是IndexFlatL2类型
    auto flat_quantizer = dynamic_cast<faiss::IndexFlatL2*>(quantizer);
    auto last_log_clock = std::chrono::system_clock::now();
    using elapsed_type = std::chrono::duration<double>;
    elapsed_type init_elapsed = elapsed_type(0);
    elapsed_type norm_elapsed = elapsed_type(0);
    elapsed_type train_elapsed = elapsed_type(0);
    elapsed_type add_elapsed = elapsed_type(0);
    elapsed_type search_elapsed = elapsed_type(0);
    size_t true_train_count = 0;
    size_t all_train_count = 0;

    size_t total_sub_recall_cos = 0;
    size_t total_sub_recall_cos_point_10 = 0;
    size_t total_sub_count_cos = 0;
    size_t total_sub_recall_l2 = 0;
    size_t total_sub_recall_l2_point_10 = 0;
    size_t total_sub_count_l2 = 0;

    auto sample_train = [&](faiss::IndexIVFFlat& index, size_t nb, const float* xb, float sample_fraction = 0.1) -> void {
        all_train_count += nb;
        size_t this_sub_nlist = index.nlist;
        size_t sample_nb = std::min(std::max(static_cast<size_t>(nb * sample_fraction), 10 * this_sub_nlist), nb);
#if 1
        if (1.0 * sample_nb / nb >= 0.5) {
            true_train_count += nb;
            index.train(nb, xb);
            return;
        }
        true_train_count += sample_nb;
        std::unique_ptr<float[]> sample_xb = std::make_unique<float[]>(sample_nb * d);
#pragma omp parallel
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, nb - 1);
#pragma omp for
            for (size_t i = 0; i < sample_nb; i++) {
                size_t x_id = dis(gen);
                memcpy(sample_xb.get() + i * d, xb + x_id * d, d * sizeof(float));
            }
        }
        index.train(sample_nb, sample_xb.get());
#else
        true_train_count += sample_nb;
        index.train(sample_nb, xb);
#endif
    };

    if (flat_quantizer != nullptr) {
        // 调用IndexFlatL2的assign方法
        auto t1 = std::chrono::system_clock::now();
        flat_quantizer->assign(n, x, coarse_idx.get(), 1, dis2nearest_center.get());
        add_core(n, x, xids, coarse_idx.get(), dis2nearest_center.get());

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nlist; i++) {
            invlistswithdist->sqrt_distances[i].resize(invlistswithdist->list_size(i));
            for (int j = 0; j < invlistswithdist->list_size(i); j++) {
                invlistswithdist->sqrt_distances[i][j] = sqrt(invlistswithdist->distances[i][j]);
            }
        }
        auto t2 = std::chrono::system_clock::now();
        printf("faiss add: %f seconds\n", std::chrono::duration_cast<elapsed_type>(t2 - t1).count());

        if (enable_intersection || enable_sub_knn_cos || enable_sub_knn_l2) {
            size_t total_processd = 0;
            size_t total_add = 0;
            size_t target_add = n * (enable_sub_knn_l2 + enable_sub_knn_cos);
            const size_t ADD_BATCH_SIZE = 10000;

            auto running_log = [&]() -> void {
                auto now = std::chrono::system_clock::now();
                if (std::chrono::duration_cast<elapsed_type>(now - last_log_clock).count() > log_interval || total_add == target_add) {
                    last_log_clock = now;
                    printf("add: %.3f%%    build: %.2f%%\n", 100.0 * total_add / target_add, 100.0 * total_processd / invlistswithdist->nlist);
                    elapsed_type total_elapsed = train_elapsed + add_elapsed + search_elapsed + init_elapsed + norm_elapsed;
                    double init_percent = 100.0 * init_elapsed.count() / total_elapsed.count();
                    double norm_percent = 100.0 * norm_elapsed.count() / total_elapsed.count();
                    double train_percent = 100.0 * train_elapsed.count() / total_elapsed.count();
                    double add_percent = 100.0 * add_elapsed.count() / total_elapsed.count();
                    double search_percent = 100.0 * search_elapsed.count() / total_elapsed.count();
                    printf("init: %.2f%%    norm: %.2f%%    train: %.2f%% (%.2f%%)    add: %.2f%%    search: %.2f%%    total: %.2f\n",
                           init_percent,
                           norm_percent,
                           train_percent,
                           100.0 * true_train_count / all_train_count,
                           add_percent,
                           search_percent,
                           total_elapsed.count());
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
                auto total_elapsed = train_elapsed + add_elapsed + search_elapsed + init_elapsed + norm_elapsed;
                printf("add: 100.0%%    build: 100.0%%\n");
                printf("init: %.2f    norm: %.2f    train: %.2f (%.2f%%)    add: %.2f    search: %.2f    total: %.2f\n",
                       init_elapsed.count(),
                       norm_elapsed.count(),
                       train_elapsed.count(),
                       100.0 * true_train_count / all_train_count,
                       add_elapsed.count(),
                       search_elapsed.count(),
                       total_elapsed.count());
            };

            std::unique_ptr<size_t[]> list_order = std::make_unique<size_t[]>(invlistswithdist->nlist);
            std::iota(list_order.get(), list_order.get() + invlistswithdist->nlist, 0);
            std::sort(list_order.get(), list_order.get() + invlistswithdist->nlist, [&](size_t a, size_t b) {
                return invlistswithdist->list_size(a) > invlistswithdist->list_size(b);
            });

            // #pragma omp parallel for schedule(dynamic)
            for (size_t oi = 0; oi < invlistswithdist->nlist; oi++) {  //
                // size_t i = list_order[oi];
                size_t i = oi;
                const float* xb = reinterpret_cast<const float*>(invlistswithdist->get_codes(i));
                faiss::idx_t nb = invlistswithdist->list_size(i);
                size_t this_sub_nlist_L2 = sub_nlist;
                size_t this_sub_nlist_cos = sub_nlist;
                size_t this_sub_nprobe_L2 = sub_nprobe;
                size_t this_sub_nprobe_cos = sub_nprobe;
                this_sub_nlist_L2 = std::min(sub_nlist, (nb + SUB_LIST_SIZE - 1) / SUB_LIST_SIZE);
                this_sub_nlist_cos = std::min(std::max(1ul, static_cast<size_t>(sub_nlist)), static_cast<size_t>((nb + SUB_LIST_SIZE - 1) / SUB_LIST_SIZE));
                this_sub_nprobe_L2 = std::min(std::max(1ul, static_cast<size_t>(1.0 * this_sub_nlist_L2 * sub_nprobe / sub_nlist)), this_sub_nlist_L2);
                this_sub_nprobe_cos = std::min(std::max(1ul, static_cast<size_t>(1.0 * this_sub_nlist_cos * sub_nprobe / sub_nlist * COS_SUB_RATIO)), this_sub_nlist_cos);
                // printf("%ld/%ld\t%ld/%ld\t", this_sub_nprobe_cos, this_sub_nlist_cos, this_sub_nprobe_L2, this_sub_nlist_L2);
                auto t1 = std::chrono::system_clock::now();
                std::unique_ptr<float[]> norm_xb = std::make_unique<float[]>(nb * d);
                if (enable_intersection) {
                    invlistswithdist->mcos[i].resize(nb);
                }
                auto t2 = std::chrono::system_clock::now();
                init_elapsed += std::chrono::duration_cast<elapsed_type>(t2 - t1);

                if (enable_intersection || enable_sub_knn_cos) {
                    auto t1 = std::chrono::system_clock::now();
                    const float* coarse_code = get_centroid_codes(i);
#pragma omp parallel for
                    for (size_t j = 0; j < nb; j++) {
                        float norm_xb_value = 0;
                        size_t x_id = invlistswithdist->get_ids(i)[j];
                        for (size_t k = 0; k < d; k++) {
                            norm_xb[j * d + k] = (x[x_id * d + k] - coarse_code[k]);
                            norm_xb_value += norm_xb[j * d + k] * norm_xb[j * d + k];
                        }

                        norm_xb_value = sqrt(norm_xb_value);
                        if (norm_xb_value > 1e-7) {
                            for (size_t k = 0; k < d; k++) {
                                norm_xb[j * d + k] /= norm_xb_value;
                            }
                        }

                        if (enable_intersection) {
                            invlistswithdist->mcos[i][j].resize(invlistswithdist->ref_vector_total);
                            for (size_t k = 0; k < invlistswithdist->ref_vector_total; k++) {
                                invlistswithdist->mcos[i][j][k] = norm_xb[j * d + k] / norm_xb_value;
                            }
                        }
                    }
                    auto t2 = std::chrono::system_clock::now();
                    norm_elapsed += std::chrono::duration_cast<elapsed_type>(t2 - t1);
                }

                if (enable_sub_knn_cos) {
                    auto t1 = std::chrono::system_clock::now();
                    invlistswithdist->nearest_dis[i].resize(nb * sub_k * 2);
                    invlistswithdist->nearest_idx[i].resize(nb * sub_k * 2);
                    auto t2 = std::chrono::system_clock::now();
                    init_elapsed += std::chrono::duration_cast<elapsed_type>(t2 - t1);
                    if (!enable_lite_sub_knn) {
                        faiss::IndexFlatL2 quantizer(d);
                        faiss::IndexIVFFlat index(&quantizer, d, this_sub_nlist_cos, faiss::METRIC_INNER_PRODUCT);
                        auto t1 = std::chrono::system_clock::now();
                        // index.train(nb, norm_xb.get());
                        sample_train(index, nb, norm_xb.get(), sub_sample);
                        auto t2 = std::chrono::system_clock::now();
                        train_elapsed += std::chrono::duration_cast<elapsed_type>(t2 - t1);
                        for (int i = 0; i * ADD_BATCH_SIZE < nb; i++) {
                            index.add(std::min(ADD_BATCH_SIZE, nb - i * ADD_BATCH_SIZE), norm_xb.get() + i * ADD_BATCH_SIZE);
                            total_add += std::min(ADD_BATCH_SIZE, nb - i * ADD_BATCH_SIZE);
                            running_log();
                        }
                        auto t3 = std::chrono::system_clock::now();
                        add_elapsed += std::chrono::duration_cast<elapsed_type>(t3 - t2);
                        index.nprobe = this_sub_nprobe_cos;
                        index.search(nb, norm_xb.get(), sub_k, invlistswithdist->nearest_dis[i].data(), invlistswithdist->nearest_idx[i].data());
                        if (true) {
                            size_t recall_nb = RECALL_RATIO * nb / (sub_nlist / sub_nprobe);
                            std::unique_ptr<float[]> g_dis = std::make_unique<float[]>(recall_nb * sub_k);
                            std::unique_ptr<idx_t[]> g_idx = std::make_unique<idx_t[]>(recall_nb * sub_k);
                            index.nprobe = this_sub_nlist_cos;
                            index.search(recall_nb, norm_xb.get(), sub_k, g_dis.get(), g_idx.get());
                            index.nprobe = this_sub_nprobe_cos;
                            for (size_t j = 0; j < recall_nb; j++) {
                                std::unordered_set<idx_t> g_set;
                                std::unordered_set<idx_t> g_set_point_10;
                                for (size_t k = 0; k < sub_k; k++) {
                                    g_set.insert(g_idx[j * sub_k + k]);
                                    if (10 * k < sub_k) {
                                        g_set_point_10.insert(g_idx[j * sub_k + k]);
                                    }
                                }
                                for (size_t k = 0; k < sub_k; k++) {
                                    if (g_set.find(invlistswithdist->nearest_idx[i][j * sub_k + k]) != g_set.end()) {
                                        total_sub_recall_cos++;
                                    }
                                    if (g_set_point_10.find(invlistswithdist->nearest_idx[i][j * sub_k + k]) != g_set_point_10.end()) {
                                        total_sub_recall_cos_point_10++;
                                    }
                                }
                            }
                            total_sub_count_cos += recall_nb * sub_k;
                        }
#pragma omp parallel for
                        for (int j = 0; j < nb * d; j++) {
                            norm_xb[j] = -norm_xb[j];
                        }
                        index.search(nb, norm_xb.get(), sub_k, invlistswithdist->nearest_dis[i].data() + nb * sub_k, invlistswithdist->nearest_idx[i].data() + nb * sub_k);
#pragma omp parallel for
                        for (size_t j = nb * sub_k; j < nb * sub_k * 2; j++) {
                            invlistswithdist->nearest_dis[i][j] = -invlistswithdist->nearest_dis[i][j];
                        }
                        auto t4 = std::chrono::system_clock::now();
                        search_elapsed += std::chrono::duration_cast<elapsed_type>(t4 - t3);
                    } else {
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::uniform_int_distribution<> dis(0, nb - 1);
                        for (size_t j = 0; j < nb; j++) {
                            for (size_t k = 0; k < 2 * sub_k; k++) {
                                idx_t x_id = dis(gen);
                                invlistswithdist->nearest_idx[i][j * sub_k + k] = x_id;
                                float dis = fvec_inner_product(norm_xb.get() + j * d, norm_xb.get() + x_id * d, d);
                                invlistswithdist->nearest_dis[i][j * sub_k + k] = dis;
                            }
                        }
                    }
                } else {
                    // std::cout<<"sub_knn_cos is disabled"<<std::endl;
                }

                // norm_xb is -norm_xb

                if (enable_sub_knn_l2) {
                    auto t1 = std::chrono::system_clock::now();
                    invlistswithdist->nearest_dis_l2[i].resize(nb * sub_k);
                    invlistswithdist->nearest_idx_l2[i].resize(nb * sub_k);
                    auto t2 = std::chrono::system_clock::now();
                    init_elapsed += std::chrono::duration_cast<elapsed_type>(t2 - t1);
                    if (!enable_lite_sub_knn) {
                        faiss::IndexFlatL2 quantizer_l2(d);
                        faiss::IndexIVFFlat index_l2(&quantizer_l2, d, this_sub_nlist_L2, faiss::METRIC_L2);
                        auto t1 = std::chrono::system_clock::now();
                        // index_l2.train(nb, xb);
                        sample_train(index_l2, nb, xb, sub_sample);
                        auto t2 = std::chrono::system_clock::now();
                        train_elapsed += std::chrono::duration_cast<elapsed_type>(t2 - t1);
                        for (int i = 0; i * ADD_BATCH_SIZE < nb; i++) {
                            index_l2.add(std::min(ADD_BATCH_SIZE, nb - i * ADD_BATCH_SIZE), xb + i * ADD_BATCH_SIZE);
                            total_add += std::min(ADD_BATCH_SIZE, nb - i * ADD_BATCH_SIZE);
                            running_log();
                        }
                        auto t3 = std::chrono::system_clock::now();
                        add_elapsed += std::chrono::duration_cast<elapsed_type>(t3 - t2);
                        index_l2.nprobe = this_sub_nprobe_L2;
                        index_l2.search(nb, xb, sub_k, invlistswithdist->nearest_dis_l2[i].data(), invlistswithdist->nearest_idx_l2[i].data());
                        if (true) {
                            size_t recall_nb = RECALL_RATIO * nb / (sub_nlist / sub_nprobe);
                            std::unique_ptr<float[]> g_dis = std::make_unique<float[]>(recall_nb * sub_k);
                            std::unique_ptr<idx_t[]> g_idx = std::make_unique<idx_t[]>(recall_nb * sub_k);
                            index_l2.nprobe = this_sub_nlist_L2;
                            index_l2.search(recall_nb, xb, sub_k, g_dis.get(), g_idx.get());
                            index_l2.nprobe = this_sub_nprobe_L2;
                            for (size_t j = 0; j < recall_nb; j++) {
                                std::unordered_set<idx_t> g_set;
                                std::unordered_set<idx_t> g_set_point_10;
                                for (size_t k = 0; k < sub_k; k++) {
                                    g_set.insert(g_idx[j * sub_k + k]);
                                    if (10 * k < sub_k) {
                                        g_set_point_10.insert(g_idx[j * sub_k + k]);
                                    }
                                }
                                for (size_t k = 0; k < sub_k; k++) {
                                    if (g_set.find(invlistswithdist->nearest_idx_l2[i][j * sub_k + k]) != g_set.end()) {
                                        total_sub_recall_l2++;
                                    }
                                    if (g_set_point_10.find(invlistswithdist->nearest_idx_l2[i][j * sub_k + k]) != g_set_point_10.end()) {
                                        total_sub_recall_l2_point_10++;
                                    }
                                }
                            }
                            total_sub_count_l2 += recall_nb * sub_k;
                        }
                        // sqrt all
#pragma omp parallel for
                        for (size_t j = 0; j < nb * sub_k; j++) {
                            invlistswithdist->nearest_dis_l2[i][j] = sqrt(invlistswithdist->nearest_dis_l2[i][j]);
                        }
                        auto t4 = std::chrono::system_clock::now();
                        search_elapsed += std::chrono::duration_cast<elapsed_type>(t4 - t3);
                    } else {
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::uniform_int_distribution<> dis(0, nb - 1);

#pragma omp parallel for
                        for (size_t j = 0; j < nb; j++) {
                            for (size_t k = 0; k < sub_k; k++) {
                                idx_t x_id = dis(gen);
                                invlistswithdist->nearest_idx_l2[i][j * sub_k + k] = x_id;
                                float dis = fvec_L2sqr(xb + j * d, xb + x_id * d, d);
                                invlistswithdist->nearest_dis_l2[i][j * sub_k + k] = sqrt(dis);
                            }
                        }
                    }
                } else {
                    // std::cout<<"sub_knn_l2 is disabled"<<std::endl;
                }

                if (enable_intersection || enable_sub_knn_cos || enable_sub_knn_l2) {
                    total_processd += 1;
                    running_log();
                }
            }
            end_log();
        }
    } else {
        std::cerr << "Quantizer is not of type IndexFlatL2" << std::endl;
        // 处理错误
    }
}

void IndexIVFwithDistance::add(idx_t n, const float* x) {
    // std::cout<<"IndexIVFwithDistance::add"<<std::endl;
    add_with_ids(n, x, nullptr);
}

/** It is a sad fact of software that a conceptually simple function like this
 * becomes very complex when you factor in several ways of parallelizing +
 * interrupt/error handling + collecting stats + min/max collection. The
 * codepath that is used 95% of time is the one for parallel_mode = 0 */
void IndexIVFwithDistance::search(
    idx_t n,
    const float* x,
    idx_t k,
    float* distances,
    idx_t* labels,
    const SearchParameters* params_in) {
    enable_triangle = !std::getenv("DISABLE_TRIANGLE");
    enable_intersection = !std::getenv("DISABLE_INTERSECTION");
    enable_sub_knn_cos = !std::getenv("DISABLE_SUB_KNN_COS");
    enable_sub_knn_l2 = !std::getenv("DISABLE_SUB_KNN_L2");
    opt_level = std::getenv("OPT_LEVEL") ? std::stoi(std::getenv("OPT_LEVEL")) : OPT_NONE;
    // std::cout << "enable_triangle: " << enable_triangle << std::endl
    //           << "enable_intersection: " << enable_intersection << std::endl
    //           << "enable_sub_knn_cos: " << enable_sub_knn_cos << std::endl
    //           << "enable_sub_knn_l2: " << enable_sub_knn_l2 << std::endl;

    // 创建一个数组来保存每个查询向量的估计值
    estimates.resize(n);
    total_skip_count = 0;
    total_skip_cos_count = 0;
    total_count = 0;
    total_compute_cos_skip_count = 0;
    total_compute_cos_skip_count_true = 0;
    total_compute_L2_skip_count_true = 0;
    total_skip_count_large = 0;
    total_used_sub_k_cos = 0;
    total_used_sub_k_l2 = 0;
    total_used_sub_k_cos_count = 0;
    total_used_sub_k_l2_count = 0;

    // for(int i=0;i<invlistswithdist->nlist;i++){
    //     std::cout<<"nvlists->is_empty() "<<i<<" : "<<invlistswithdist->is_empty(i)<<std::endl;
    //     std::cout<<"nvlists->list_size() "<<i<<" : "<<invlistswithdist->list_size(i)<<std::endl;
    // }

    FAISS_THROW_IF_NOT(k > 0);
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
        std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // search function for a subset of queries
    auto sub_search_func = [this, k, nprobe, params](
                               idx_t n,
                               const float* x,
                               float* distances,
                               idx_t* labels,
                               IndexIVFStats* ivf_stats,
                               idx_t global_begin) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        double t0 = getmillisecs();
        quantizer->search(
            n,
            x,
            nprobe,
            coarse_dis.get(),
            idx.get(),
            params ? params->quantizer_params : nullptr);

// 给每个向量估计一个上界，coarse_dis[i*nprobe]是最近聚类中心的距离
// 这里要做的是get_single_distance(idx[i*nprobe],k);的k可能越界，这个还要放在一个循环里面
#pragma omp for
        for (size_t i = 0; i < n; i++) {
            estimates[i] = coarse_dis[i * nprobe] + invlistswithdist->get_single_distance(idx[i * nprobe], k);
        }

        double t1 = getmillisecs();
        invlistswithdist->prefetch_lists(idx.get(), n * nprobe);

        search_preassigned(
            n,
            x,
            k,
            idx.get(),
            coarse_dis.get(),
            distances,
            labels,
            false,
            params,
            ivf_stats,
            global_begin);
        double t2 = getmillisecs();
        ivf_stats->quantization_time += t1 - t0;
        ivf_stats->search_time += t2 - t0;
    };

    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);
        std::mutex exception_mutex;
        std::string exception_string;

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            IndexIVFStats local_stats;
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt;
            if (i1 > i0) {
                try {
                    sub_search_func(
                        i1 - i0,
                        x + i0 * d,
                        distances + i0 * k,
                        labels + i0 * k,
                        &stats[slice],
                        i0);
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    exception_string = e.what();
                }
            }
        }

        if (!exception_string.empty()) {
            FAISS_THROW_MSG(exception_string.c_str());
        }

        // collect stats
        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        // handle parallelization at level below (or don't run in parallel at
        // all)
        sub_search_func(n, x, distances, labels, &indexIVF_stats, 0);
    }
    const char* skip_logging_path = getenv("SKIP_LOGGING_PATH");
    if (!skip_logging_path) {
        std::cerr << "SKIP_LOGGING_PATH not set" << std::endl;
    } else {
        std::fstream file(skip_logging_path, std::ios::out);
        total_used_sub_k_cos_count = std::max(total_used_sub_k_cos_count, 1ul);
        total_used_sub_k_l2_count = std::max(total_used_sub_k_l2_count, 1ul);
        file
            << "{" << std::endl
            << "\"mean_used_sub_k_cos\":" << 1.0 * total_used_sub_k_cos / total_used_sub_k_cos_count << "," << std::endl
            << "\"mean_used_sub_k_l2\":" << 1.0 * total_used_sub_k_l2 / total_used_sub_k_l2_count << "," << std::endl
            << "\"total_skip_count\":" << total_skip_count << "," << std::endl
            << "\"total_skip_count_large\":" << total_skip_count_large << "," << std::endl
            // << "\"total_skip_cos_count\":" << total_skip_cos_count << "," << std::endl
            << "\"total_compute_cos_skip_count_true\":" << total_compute_cos_skip_count_true << "," << std::endl
            << "\"total_compute_L2_skip_count_true\":" << total_compute_L2_skip_count_true << "," << std::endl
            << "\"total_count\":" << total_count << std::endl
            << "}";
    }
}

void IndexIVFwithDistance::search_preassigned(
    idx_t n,
    const float* x,
    idx_t k,
    const idx_t* keys,
    const float* coarse_dis,
    float* distances,
    idx_t* labels,
    bool store_pairs,
    const IVFSearchParameters* params,
    IndexIVFStats* ivf_stats,
    idx_t global_begin) {
    FAISS_THROW_IF_NOT(k > 0);

    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    const idx_t unlimited_list_size = std::numeric_limits<idx_t>::max();
    idx_t max_codes = params ? params->max_codes : this->max_codes;
    IDSelector* sel = params ? params->sel : nullptr;
    const IDSelectorRange* selr = dynamic_cast<const IDSelectorRange*>(sel);
    if (selr) {
        if (selr->assume_sorted) {
            sel = nullptr;  // use special IDSelectorRange processing
        } else {
            selr = nullptr;  // use generic processing
        }
    }

    FAISS_THROW_IF_NOT_MSG(
        !(sel && store_pairs),
        "selector and store_pairs cannot be combined");

    FAISS_THROW_IF_NOT_MSG(
        !invlistswithdist->use_iterator || (max_codes == 0 && store_pairs == false),
        "iterable inverted lists don't support max_codes and store_pairs");

    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    bool do_heap_init = !(this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT);

    FAISS_THROW_IF_NOT_MSG(
        max_codes == 0 || pmode == 0 || pmode == 3,
        "max_codes supported only for parallel_mode = 0 or 3");

    if (max_codes == 0) {
        max_codes = unlimited_list_size;
    }

    bool do_parallel = omp_get_max_threads() >= 2 &&
                       (pmode == 0   ? false
                        : pmode == 3 ? n > 1
                        : pmode == 1 ? nprobe > 1
                                     : nprobe * n > 1);

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap)
    {
        InvertedListScanner* scanner =
            get_opt_InvertedListScanner(store_pairs, sel, opt_level);
        ScopeDeleter1<InvertedListScanner> del(scanner);

        /*****************************************************
         * Depending on parallel_mode, there are two possible ways
         * to organize the search. Here we define local functions
         * that are in common between the two
         ******************************************************/

        // initialize + reorder a result heap

        auto init_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }
        };

        // std::cout<<"success in 0"<<std::endl;
        auto add_local_results = [&](const float* local_dis,
                                     const idx_t* local_idx,
                                     float* simi,
                                     idx_t* idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
            } else {
                heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
            }
        };

        auto reorder_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP>(k, simi, idxi);
            } else {
                heap_reorder<HeapForL2>(k, simi, idxi);
            }
        };
        // single list scan using the current scanner (with query
        // set porperly) and storing results in simi and idxi
        auto scan_one_list = [&](idx_t key,
                                 float coarse_dis_i,
                                 float* simi,
                                 idx_t* idxi,
                                 idx_t list_size_max,
                                 float estimate_diff,
                                 idx_t sub_query_i = 0) {
            if (key < 0) {
                return (size_t)0;
            }
            FAISS_THROW_IF_NOT_FMT(
                key < (idx_t)nlist,
                "Invalid key=%" PRId64 " nlist=%zd\n",
                key,
                nlist);

            // don't waste time on empty lists
            if (invlistswithdist->is_empty(key)) {
                // std::cout<<"invlistswithdist->is_empty"<<std::endl;
                return (size_t)0;
            }
            scanner->set_list(key, coarse_dis_i);

            nlistv++;

            try {
                if (invlistswithdist->use_iterator) {
                    // std::cout<<"unsuccess"<<std::endl;
                    size_t list_size = 0;

                    std::unique_ptr<InvertedListsIterator> it(
                        invlistswithdist->get_iterator(key));

                    nheap += scanner->iterate_codes(
                        it.get(), simi, idxi, k, list_size);

                    return list_size;
                } else {
                    // std::cout<<"success in it"<<std::endl;
                    size_t list_size = invlistswithdist->list_size(key);
                    if (list_size > list_size_max) {
                        list_size = list_size_max;
                    }

                    InvertedLists::ScopedCodes scodes(invlistswithdist, key);
                    const uint8_t* codes = scodes.get();

                    std::unique_ptr<InvertedLists::ScopedIds> sids;
                    const idx_t* ids = nullptr;

                    if (!store_pairs) {
                        sids = std::make_unique<InvertedLists::ScopedIds>(
                            invlistswithdist, key);
                        ids = sids->get();
                    }

                    if (selr) {  // IDSelectorRange
                        // restrict search to a section of the inverted list
                        size_t jmin, jmax;
                        selr->find_sorted_ids_bounds(
                            list_size, ids, &jmin, &jmax);
                        list_size = jmax - jmin;
                        if (list_size == 0) {
                            return (size_t)0;
                        }
                        codes += jmin * code_size;
                        ids += jmin;
                    }

                    const float* dists = invlistswithdist->get_distances(key);
                    const float* sqrt_dists = invlistswithdist->get_sqrt_distances(key);

                    size_t skip_count = 0;
                    size_t skip_count_large = 0;
                    size_t scan_begin = 0;
                    size_t scan_end = list_size;

                    float sqrt_simi = sqrt(simi[0]);
                    float sqrt_coarse_dis_i = sqrt(coarse_dis_i);
                    if (enable_triangle) {
                        for (size_t j = 0; j < list_size; j++) {
                            float tmp = sqrt_simi + sqrt_dists[j];
                            if (tmp < sqrt_coarse_dis_i) {
                                skip_count++;
                            } else {
                                break;
                            }
                        }

                        for (size_t j = list_size - 1; j >= 0; j--) {
                            float tmp_large = sqrt_simi + sqrt_coarse_dis_i;
                            tmp_large *= tmp_large;
                            if (tmp_large < dists[j]) {
                                skip_count_large++;
                            } else {
                                break;
                            }
                        }
                        scan_begin = skip_count;
                        scan_end -= skip_count_large;
                    }

                    size_t skip_cos_count = 0;

                    std::unique_ptr<bool[]> if_skip = std::make_unique<bool[]>(list_size);
                    // for (size_t i = 0; i < list_size; i++) {
                    //     if_skip[i] = 0;
                    // }

                    // if ((skip_count || coarse_dis_i - dists[0] > simi[0]) && list_size != 0 && enable_intersection) {
                    //     float max_radius = dists[list_size - 1];
                    //     float possible_cos1[invlistswithdist->ref_vector_total];
                    //     float possible_cos2[invlistswithdist->ref_vector_total];

                    //     float diff_cos = 0;
                    //     float diff_sin;
                    //     std::vector<float> delta_vec(d);

                    //     float delta_vec_norm = 0;
                    //     for (size_t dim = 0; dim < d; ++dim) {
                    //         float c = get_centroid_codes(key)[dim];
                    //         delta_vec[dim] = x[sub_query_i * d + dim] - c;
                    //         delta_vec_norm += delta_vec[dim] * delta_vec[dim];
                    //     }
                    //     delta_vec_norm = sqrt(delta_vec_norm);

                    //     float cos_to_ref_vector[invlistswithdist->ref_vector_total];
                    //     float sin_to_ref_vector[invlistswithdist->ref_vector_total];
                    //     for (size_t i = 0; i < invlistswithdist->ref_vector_total; i++) {
                    //         cos_to_ref_vector[i] = delta_vec[i] / delta_vec_norm;
                    //         sin_to_ref_vector[i] = sqrt(1 - cos_to_ref_vector[i] * cos_to_ref_vector[i]);
                    //     }

                    //     // diff cos 表示可控范围的cos大小，这个值应该是在0-90°之间，首先要解决的是max_radius过大的问题
                    //     // 之前觉得这个是该排除，现在我想了一下不应该是排除
                    //     // 应该是if(a²+b²>c²就可以认为可以相切),这里可以先画图看看

                    //     if (max_radius + simi[0] >= coarse_dis_i) {
                    //         diff_cos = sqrt(1 - simi[0] / coarse_dis_i);
                    //         diff_sin = sqrt(simi[0] / coarse_dis_i);
                    //     } else {
                    //         diff_cos = (max_radius + coarse_dis_i - simi[0]) / (2 * sqrt(coarse_dis_i) * sqrt(max_radius));
                    //         diff_sin = sqrt(1 - diff_cos * diff_cos);
                    //     }

                    //     for (size_t i = 0; i < invlistswithdist->ref_vector_total; i++) {
                    //         // 第二个点是这里，要处理有没有越过cos=1或者cos=-1的情况，解决这种情况很简单（画个图就明白了）
                    //         // 可活动范围的角度（diff_sin）大于到cos=1或者cos=-1所需的角度（正好也是sin[i]）
                    //         if (diff_sin > sin_to_ref_vector[i]) {
                    //             if (diff_cos > 0) {
                    //                 // std::cout<<"case 1!"<<std::endl;
                    //                 possible_cos2[i] = 1;
                    //                 possible_cos1[i] = diff_cos * cos_to_ref_vector[i] - diff_sin * sin_to_ref_vector[i];
                    //             } else {
                    //                 // std::cout<<"case 2!"<<std::endl;
                    //                 possible_cos1[i] = -1;
                    //                 possible_cos2[i] = diff_cos * cos_to_ref_vector[i] + diff_sin * sin_to_ref_vector[i];
                    //             }
                    //         } else {
                    //             // std::cout<<diff_sin<<" "<<sin1[i]<<std::endl;
                    //             possible_cos1[i] = diff_cos * cos_to_ref_vector[i] - diff_sin * sin_to_ref_vector[i];
                    //             possible_cos2[i] = diff_cos * cos_to_ref_vector[i] + diff_sin * sin_to_ref_vector[i];
                    //             if (possible_cos1[i] > possible_cos2[i]) {
                    //                 std::swap(possible_cos1[i], possible_cos2[i]);
                    //             }
                    //         }
                    //     }
                    //     for (size_t ii = 0; ii < list_size; ii++) {
                    //         for (size_t jj = 0; jj < invlistswithdist->ref_vector_total; jj++) {
                    //             float m_cos = invlistswithdist->get_single_mcos(key, ii + skip_count)[jj];
                    //             // std::cout<<"from "<<possible_cos1[jj]<<" to "<<possible_cos2[jj]<<"  and cur is "<<m_cos<<std::endl;
                    //             if (m_cos < possible_cos1[jj] ||
                    //                 m_cos > possible_cos2[jj]) {
                    //                 if_skip[ii] = 0;
                    //                 skip_cos_count++;
                    //                 break;
                    //             }
                    //         }
                    //     }
                    // }

                    // const idx_t* nearest_cos_id = enable_sub_knn_cos ? invlistswithdist->get_nearest_idx(key) : nullptr;
                    // const float* nearest_cos_dis = enable_sub_knn_cos ? invlistswithdist->get_nearest_dis(key) : nullptr;

                    // const idx_t* nearest_L2_id = enable_sub_knn_l2 ? invlistswithdist->get_nearest_idx_l2(key) : nullptr;
                    // const float* nearest_L2_dis = enable_sub_knn_l2 ? invlistswithdist->get_nearest_dis_l2(key) : nullptr;

                    const idx_t* nearest_cos_id = invlistswithdist->get_nearest_idx(key);
                    const float* nearest_cos_dis = invlistswithdist->get_nearest_dis(key);

                    const idx_t* nearest_L2_id = invlistswithdist->get_nearest_idx_l2(key);
                    const float* nearest_L2_dis = invlistswithdist->get_nearest_dis_l2(key);

                    size_t compute_skip_cos_count_true = 0;
                    size_t compute_skip_L2_count_true = 0;
                    size_t used_sub_k_cos = 0;
                    size_t used_sub_k_L2 = 0;
                    size_t used_sub_k_cos_count = 0;
                    size_t used_sub_k_L2_count = 0;
                    nheap += scanner->my_scan_codes(
                        scan_begin,
                        scan_end,
                        list_size,
                        codes,
                        ids,
                        simi,
                        idxi,
                        k,
                        if_skip.get(),
                        coarse_dis_i,
                        dists,
                        sqrt_dists,
                        nearest_cos_id,
                        nearest_cos_dis,
                        nearest_L2_id,
                        nearest_L2_dis,
                        sub_k,
                        compute_skip_cos_count_true,
                        compute_skip_L2_count_true,
                        used_sub_k_cos,
                        used_sub_k_cos_count,
                        used_sub_k_L2,
                        used_sub_k_L2_count);

#ifdef DEBUG
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        total_skip_count += skip_count;  // 更新total_skip_count
                        total_skip_count_large += skip_count_large;
                        total_skip_cos_count += skip_cos_count;
                        total_count += list_size;  // 更新total_skip_count
                        total_compute_cos_skip_count_true += compute_skip_cos_count_true;
                        total_compute_L2_skip_count_true += compute_skip_L2_count_true;
                        total_used_sub_k_cos += used_sub_k_cos;
                        total_used_sub_k_cos_count += used_sub_k_cos_count;
                        total_used_sub_k_l2 += used_sub_k_L2;
                        total_used_sub_k_l2_count += used_sub_k_L2_count;
                    }
#endif
                    return list_size;
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                    demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
                return size_t(0);
            }
        };

        /****************************************************
         * Actual loops, depending on parallel_mode
         ****************************************************/

        if (pmode == 0 || pmode == 3) {
            float* simi;
#pragma omp for private(simi)
            for (idx_t i = 0; i < n; i++) {
                if (interrupt) {
                    continue;
                }

                // loop over queries
                scanner->set_query(x + i * d);
                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;

                init_result(simi, idxi);

                idx_t nscan = 0;
                idx_t times_scan_one_list = 0;
                for (size_t ik = 0; ik < nprobe; ik++) {
                    float estimate_diff = coarse_dis[i * nprobe + ik] - simi[0];
                    nscan += scan_one_list(
                        keys[i * nprobe + ik],
                        coarse_dis[i * nprobe + ik],
                        simi,
                        idxi,
                        max_codes - nscan,
                        estimates[i],
                        i);
                    times_scan_one_list++;
                    if (nscan >= max_codes) {
                        break;
                    }
                }

                ndis += nscan;
                reorder_result(simi, idxi);

                if (InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }

            }  // parallel for
        } else if (pmode == 1) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

            for (size_t i = 0; i < n; i++) {
                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
                for (idx_t ik = 0; ik < nprobe; ik++) {
                    float estimate_diff;
                    ndis += scan_one_list(
                        keys[i * nprobe + ik],
                        coarse_dis[i * nprobe + ik],
                        local_dis.data(),
                        local_idx.data(),
                        unlimited_list_size,
                        estimate_diff);

                    // can't do the test on max_codes
                }
                // merge thread-local results

                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;
#pragma omp single
                init_result(simi, idxi);

#pragma omp barrier
#pragma omp critical
                {
                    add_local_results(
                        local_dis.data(), local_idx.data(), simi, idxi);
                }
#pragma omp barrier
#pragma omp single
                reorder_result(simi, idxi);
            }
        } else if (pmode == 2) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                init_result(distances + i * k, labels + i * k);
            }

#pragma omp for schedule(dynamic)
            for (int64_t ij = 0; ij < n * nprobe; ij++) {
                size_t i = ij / nprobe;
                size_t j = ij % nprobe;

                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());
                float estimate_diff;
                ndis += scan_one_list(
                    keys[ij],
                    coarse_dis[ij],
                    local_dis.data(),
                    local_idx.data(),
                    unlimited_list_size,
                    estimate_diff);

#pragma omp critical
                {
                    add_local_results(
                        local_dis.data(),
                        local_idx.data(),
                        distances + i * k,
                        labels + i * k);
                }
            }
#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                reorder_result(distances + i * k, labels + i * k);
            }
        } else {
            FAISS_THROW_FMT("parallel_mode %d not supported\n", pmode);
        }
    }  // parallel section

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                "search interrupted with: %s", exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (ivf_stats == nullptr) {
        ivf_stats = &indexIVF_stats;
    }
    ivf_stats->nq += n;
    ivf_stats->nlist += nlistv;
    ivf_stats->ndis += ndis;
    ivf_stats->nheap_updates += nheap;
}

}  // namespace faiss