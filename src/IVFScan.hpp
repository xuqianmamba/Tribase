#pragma once
#include "common.h"
#include "heap.hpp"
#include "stats.h"
#include "utils.h"

#define MANUAL_SIMD

#ifndef MANUAL_SIMD
#include "hnswlib/hnswlib.h"
#endif

namespace tribase {

class IVFScanBase {
   public:
    size_t d;
    size_t k;
    const float* query;
    float query_norm;

    IVFScanBase(size_t d, size_t k)
        : d(d), k(k) {}

    void set_query(const float* query) {
        this->query = query;
        this->query_norm = calculatedInnerProduct(query, query, d);
    }

    virtual void lite_scan_codes(size_t list_size,
                                 const float* codes,
                                 const size_t* ids,
                                 float* simi,
                                 idx_t* idxi) = 0;

    virtual void scan_codes(size_t scan_begin,
                            size_t scan_end,
                            size_t list_size,
                            const float* codes,
                            const size_t* ids,
                            float* simi,
                            idx_t* idxi) = 0;

    virtual void scan_codes(size_t scan_begin,
                            size_t scan_end,
                            size_t list_size,
                            const float* codes,
                            const size_t* ids,
                            const float* codes_norms,
                            const float centroid2query,
                            const float* candicate2centroid,
                            const float* sqrt_candicate2centroid,
                            const size_t sub_k,
                            const idx_t* nearest_IP_id,
                            const float* nearest_IP_dis,
                            const idx_t* farest_IP_id,
                            const float* farest_IP_dis,
                            const idx_t* nearest_L2_id,
                            const float* nearest_L2_dis,
                            bool* if_skip,
                            float* simi,
                            idx_t* idxi,
                            Stats* stats,
                            const float* centroid_code,
                            float sqrt_ratio,
                            float i_ratio) = 0;
};

template <MetricType metric, OptLevel opt_level, EdgeDevice edge_device_enabled>
class IVFScan : public IVFScanBase {
   public:

#if defined(MANUAL_SIMD)
    using dis_calculator_t = std::function<float(const float*, const float*, size_t)>;
    dis_calculator_t dis_calculator;
    IVFScan(size_t d, size_t k)
        : IVFScanBase(d, k) {
        if constexpr (metric == MetricType::METRIC_IP) {
            if constexpr (edge_device_enabled) {
                dis_calculator = calculatedInnerProduct0;
            } else {
                dis_calculator = calculatedInnerProduct;
            }
        } else if constexpr (metric == MetricType::METRIC_L2) {
            if constexpr (edge_device_enabled) {
                dis_calculator = calculatedEuclideanDistance0;
            } else {
                dis_calculator = calculatedEuclideanDistance;
            }
        } else {
            static_assert(false, "Unsupported metric type");
        }
    }
#else
    hnswlib::DISTFUNC<float> dis_calculator;
    IVFScan(size_t d, size_t k)
        : IVFScanBase(d, k) {
        if constexpr (metric == MetricType::METRIC_IP) {
            auto s = hnswlib::InnerProductSpace(d);
            dis_calculator = s.get_dist_func();
        } else if constexpr (metric == MetricType::METRIC_L2) {
            auto s = hnswlib::L2Space(d);
            dis_calculator = s.get_dist_func();
        } else {
            static_assert(false, "Unsupported metric type");
        }
    }
#endif

    void lite_scan_codes(size_t list_size,
                         const float* codes,
                         const size_t* ids,
                         float* simi,
                         idx_t* idxi) override {
        for (size_t i = 0; i < list_size; i++) {
            const float* candicate = codes + i * d;
            float dis = 0;
            if constexpr (metric == MetricType::METRIC_IP) {
                if constexpr (!edge_device_enabled) {
                    dis = calculatedInnerProduct(query, candicate, d);
                } else {
                    dis = calculatedInnerProduct0(query, candicate, d);
                }
                if (dis > simi[0]) {
                    heap_replace_top<metric>(k, simi, idxi, dis, ids[i]);
                }
            } else if constexpr (metric == MetricType::METRIC_L2) {
                if constexpr (!edge_device_enabled) {
                    dis = calculatedEuclideanDistance(query, candicate, d);
                } else {
                    dis = calculatedEuclideanDistance0(query, candicate, d);
                }
                if (dis < simi[0]) {
                    heap_replace_top<metric>(k, simi, idxi, dis, ids[i]);
                }
            } else {
                static_assert(false, "Unsupported metric type");
            }
        }
    }

    void scan_codes(size_t scan_begin,
                    size_t scan_end,
                    [[maybe_unused]] size_t list_size,
                    const float* codes,
                    const size_t* ids,
                    float* simi,
                    idx_t* idxi) override {
        for (size_t i = scan_begin; i < scan_end; i++) {
            const float* candicate = codes + i * d;
            float dis = 0;
            if constexpr (metric == MetricType::METRIC_IP) {
                if constexpr (!edge_device_enabled) {
                    dis = calculatedInnerProduct(query, candicate, d);
                } else {
                    dis = calculatedInnerProduct0(query, candicate, d);
                }
                if (dis > simi[0]) {
                    heap_replace_top<metric>(k, simi, idxi, dis, ids[i]);
                }
            } else if constexpr (metric == MetricType::METRIC_L2) {
                if constexpr (!edge_device_enabled) {
                    dis = calculatedEuclideanDistance(query, candicate, d);
                } else {
                    dis = calculatedEuclideanDistance0(query, candicate, d);
                }
                if (dis < simi[0]) {
                    heap_replace_top<metric>(k, simi, idxi, dis, ids[i]);
                }
            } else {
                static_assert(false, "Unsupported metric type");
            }
        }
    }

    void scan_codes(size_t scan_begin,
                    size_t scan_end,
                    [[maybe_unused]] size_t list_size,
                    const float* codes,
                    const size_t* ids,
                    [[maybe_unused]] const float* codes_norms,
                    const float centroid2query,
                    const float* candicate2centroid,
                    const float* sqrt_candicate2centroid,
                    const size_t sub_k,
                    const idx_t* nearest_IP_id,
                    const float* nearest_IP_dis,
                    const idx_t* farest_IP_id,
                    const float* farest_IP_dis,
                    const idx_t* nearest_L2_id,
                    const float* nearest_L2_dis,
                    bool* if_skip,
                    float* simi,
                    idx_t* idxi,
                    Stats* stats,
                    [[maybe_unused]] const float* centroid_code = nullptr,
                    float sqrt_ratio = 1,
                    float i_ratio = 1) override {
        float max_radius;
        float diff_cos, diff_sin;
        float max_radius_plus_centroid2query;
        float inv_two_times_sqrt_max_radius_times_centroid2query;
        float inv_sqrt_centroid2query;
        float point5_times_inv_sqrt_centroid2query;
        float sqrt_simi;

        // auto dis_calculator = [](const float* vec1, const float* vec2, size_t size) {
        //     if constexpr (metric == MetricType::METRIC_IP) {
        //         return calculatedInnerProduct(vec1, vec2, size);
        //     } else if constexpr (metric == MetricType::METRIC_L2) {
        //         return calculatedEuclideanDistance(vec1, vec2, size);
        //     } else {
        //         static_assert(false, "Unsupported metric type");
        //     }
        // };

        auto dis_comparator = [](float dis, float simi) {
            if constexpr (metric == MetricType::METRIC_IP) {
                return dis > simi;
            } else if constexpr (metric == MetricType::METRIC_L2) {
                return dis < simi;
            } else {
                static_assert(false, "Unsupported metric type");
            }
        };

        if constexpr (opt_level & OptLevel::OPT_SUBNN_IP) {
            max_radius = candicate2centroid[scan_end - 1];
            max_radius_plus_centroid2query = max_radius + centroid2query;
            inv_two_times_sqrt_max_radius_times_centroid2query = 1 / (2 * sqrt(max_radius * centroid2query));
            inv_sqrt_centroid2query = 1 / sqrt(centroid2query);
            point5_times_inv_sqrt_centroid2query = 0.5 * inv_sqrt_centroid2query;
            if (max_radius + simi[0] >= centroid2query) {
                diff_cos = sqrt(centroid2query - simi[0]) * inv_sqrt_centroid2query;
            } else {
                diff_cos =
                    (max_radius_plus_centroid2query - simi[0]) * inv_two_times_sqrt_max_radius_times_centroid2query;
            }
            diff_sin = sqrt(1 - diff_cos * diff_cos);
        }
        if constexpr (opt_level & OptLevel::OPT_SUBNN_L2) {
            sqrt_simi = sqrt(simi[0]);
        }
        for (size_t i = scan_begin; i < scan_end; i++) {
            if constexpr ((opt_level & OptLevel::OPT_SUBNN_IP) || (opt_level & OptLevel::OPT_SUBNN_L2)) {
                _mm_prefetch((char*)(if_skip + i + 1), _MM_HINT_T0);
                if (if_skip[i]) {
                    // assert(false);
                    continue;
                }
            }
            _mm_prefetch((char*)(codes + (i + 1) * d), _MM_HINT_T0);
            const float* candicate = codes + i * d;
            float dis;
            if constexpr (metric == MetricType::METRIC_L2) {
#ifndef MANUAL_SIMD
                dis = dis_calculator(query, candicate, &d);
#else
                dis = dis_calculator(query, candicate, d);
#endif
                // dis = calculatedEuclideanDistance(query, candicate, query_norm, d);
                // const float candicate_norm = codes_norms[i];
                // dis = calculatedEuclideanDistance(query, candicate, query_norm, candicate_norm, d);
            } else {
#ifndef MANUAL_SIMD
                dis = dis_calculator(query, candicate, &d);
#else
                dis = dis_calculator(query, candicate, d);
#endif
            }

            if (dis_comparator(dis, simi[0])) [[unlikely]] {
                IF_STATS {
                    stats->simi_update_count++;
                }
                idx_t id = ids[i];
                heap_replace_top<metric>(k, simi, idxi, dis, id);

                if constexpr (opt_level & OptLevel::OPT_SUBNN_IP) {
                    if (max_radius + simi[0] >= centroid2query) {
                        diff_cos = sqrt(centroid2query - simi[0]) * inv_sqrt_centroid2query;
                    } else {
                        diff_cos = (max_radius_plus_centroid2query - simi[0]) *
                                   inv_two_times_sqrt_max_radius_times_centroid2query;
                    }
                    diff_sin = sqrt(1 - diff_cos * diff_cos);
                }

                if constexpr (opt_level & OptLevel::OPT_SUBNN_L2) {
                    sqrt_simi = sqrt(simi[0]);
                }
            }

            if constexpr (opt_level & OptLevel::OPT_SUBNN_L2) {
                IF_STATS {
                    stats->check_subnn_L2_count += 1;
                }
                size_t skip_fake_id_begin = i * sub_k;
                size_t skip_fake_id_end = skip_fake_id_begin + sub_k;
                skip_fake_id_begin += 1;
                for (size_t skip_fake_id = skip_fake_id_begin; skip_fake_id < skip_fake_id_end; skip_fake_id++) {
                    float tmp_plus = nearest_L2_dis[skip_fake_id] + sqrt_simi;
                    int64_t skip_true_id = nearest_L2_id[skip_fake_id];
                    if (skip_true_id >= 0 && dis > sqrt_ratio * tmp_plus * tmp_plus) {  // already sqrt nearest_L2_dis
#ifdef CORRECTNESS_CHECK
#ifndef MANUAL_SIMD
                        float true_dis = dis_calculator(query, codes + skip_true_id * d, &d);
#else
                        float true_dis = dis_calculator(query, codes + skip_true_id * d, d);
#endif
#pragma omp critical
                        if (true_dis < simi[0]) {
                            std::cerr << "Error: " << true_dis << " " << dis << " " << nearest_L2_id[skip_fake_id] << std::endl;
                            throw std::runtime_error("SUBNN_L2_NEAREST_ERROR");
                        }
#endif
                        IF_STATS {
                            if (!if_skip[skip_true_id] && i < skip_true_id) { //  
                                // std::ofstream("logs/fuck.txt", std::ios::app) << skip_true_id << " " << skip_fake_id << " " << nearest_L2_dis[skip_fake_id] << " " << sqrt_simi << " " << dis << ", ";
                                stats->skip_subnn_L2_count++;
                            }
                        }
                        if_skip[skip_true_id] = true;
                    } else {
                        // std::ofstream("logs/fuck.txt", std::ios::app) << "["  << skip_true_id << " " << skip_fake_id << " " << nearest_L2_dis[skip_fake_id] << " " << sqrt_simi << " " << dis << " END]";
                        IF_STATS {
                            stats->check_subnn_L2_ele_count += skip_fake_id - skip_fake_id_begin;
                        }
                        break;
                    }
                }
                // std::ofstream("logs/fuck.txt", std::ios::app) << " " << skip_fake_id_end << std::endl;
            }

            if constexpr (opt_level & OptLevel::OPT_SUBNN_IP) {
                if (centroid2query > simi[0]) {
                    float this_cos = (candicate2centroid[i] + centroid2query - dis) *
                                     point5_times_inv_sqrt_centroid2query / sqrt_candicate2centroid[i];
                    float this_sin = sqrt(1 - this_cos * this_cos);

                    float tmpa = diff_cos * this_cos;
                    float tmpb = diff_sin * this_sin;
                    float cut_degree_cos_minus = tmpa + tmpb;
                    float cut_degree_cos_plus = tmpa - tmpb;

                    if (this_cos < diff_cos && this_cos > -diff_cos) {
                        IF_STATS {
                            stats->check_subnn_IP_count += 2;
                        }
                        size_t skip_fake_id_begin = i * sub_k;
                        size_t skip_fake_id_end = skip_fake_id_begin + sub_k;
                        skip_fake_id_begin += 1;
                        for (size_t skip_fake_id = skip_fake_id_begin; skip_fake_id < skip_fake_id_end;
                             skip_fake_id++) {
                            int64_t skip_true_id = nearest_IP_id[skip_fake_id];
                            if (skip_true_id >= 0 && nearest_IP_dis[skip_fake_id] > i_ratio * cut_degree_cos_minus) {
#ifdef CORRECTNESS_CHECK
#ifndef MANUAL_SIMD
                                float true_dis = dis_calculator(query, codes + skip_true_id * d, &d);
#else
                                float true_dis = dis_calculator(query, codes + skip_true_id * d, d);
#endif
#pragma omp critical
                                if (true_dis < simi[0]) {
                                    std::cerr << std::format("Error: query->p2: {} <= {}, nearestesIP: {}, cut: {}", sqrt(true_dis), sqrt(simi[0]), nearest_IP_dis[skip_fake_id], cut_degree_cos_minus) << std::endl;
                                    std::cerr << std::format("query->p1: {}, max_r: {}, query->c: {}", sqrt(dis), sqrt(max_radius), sqrt(centroid2query)) << std::endl;
                                    std::cerr << std::format("c->p1: {}, c->p2: {}", sqrt(candicate2centroid[i]), sqrt(candicate2centroid[skip_true_id])) << std::endl;
#ifndef MANUAL_SIMD
                                    assert(candicate2centroid[i] == dis_calculator(codes + i * d, centroid_code, &d));
                                    assert(candicate2centroid[skip_true_id] == dis_calculator(codes + skip_true_id * d, centroid_code, &d));
#else
                                    assert(candicate2centroid[i] == dis_calculator(codes + i * d, centroid_code, d));
                                    assert(candicate2centroid[skip_true_id] == dis_calculator(codes + skip_true_id * d, centroid_code, d));
#endif
                                    std::cout << (metric == MetricType::METRIC_L2) << std::endl;

                                    output_codes(centroid_code, d);
                                    output_codes(query, d);
                                    output_codes(codes + i * d, d);
                                    output_codes(codes + skip_true_id * d, d);
                                    // throw std::runtime_error("SUBNN_IP_NEAREST_ERROR");
                                    assert(false);
                                }
#endif
                                IF_STATS {
                                    if (!if_skip[skip_true_id] && i < skip_true_id) {
                                        stats->skip_subnn_IP_count++;
                                    }
                                }
                                if_skip[skip_true_id] = true;
                            } else {
                                IF_STATS {
                                    stats->check_subnn_IP_ele_count += skip_fake_id - skip_fake_id_begin;
                                }
                                break;
                            }
                        }

                        skip_fake_id_begin = i * sub_k;
                        skip_fake_id_end = skip_fake_id_begin + sub_k;
                        for (size_t skip_fake_id = skip_fake_id_begin; skip_fake_id < skip_fake_id_end;
                             skip_fake_id++) {
                            size_t skip_true_id = farest_IP_id[skip_fake_id];
                            if (skip_true_id > 0 && farest_IP_dis[skip_fake_id] < cut_degree_cos_plus) {
#ifdef CORRECTNESS_CHECK
                                float true_dis = dis_calculator(query, codes + skip_true_id * d, &d);
#pragma omp critical
                                if (true_dis < simi[0]) {
                                    std::cerr << std::format("Error: query->p2: {} <= {}, farestesIP: {}, cut: {}", sqrt(true_dis), sqrt(simi[0]), farest_IP_dis[skip_fake_id], cut_degree_cos_plus) << std::endl;
                                    std::cerr << std::format("query->p1: {}, max_r: {}, query->c: {}", sqrt(dis), sqrt(max_radius), sqrt(centroid2query)) << std::endl;
                                    std::cerr << std::format("c->p1: {}, c->p2: {}", sqrt(candicate2centroid[i]), sqrt(candicate2centroid[skip_true_id])) << std::endl;
                                    output_codes(centroid_code, d);
                                    output_codes(query, d);
                                    output_codes(codes + i * d, d);
                                    output_codes(codes + skip_true_id * d, d);
                                    // throw std::runtime_error("SUBNN_IP_FAREST_ERROR");
                                    assert(false);
                                }
#endif
                                IF_STATS {
                                    if (!if_skip[skip_true_id] && i < skip_true_id) {
                                        stats->skip_subnn_IP_count++;
                                    }
                                }
                                if_skip[skip_true_id] = true;
                            } else {
                                IF_STATS {
                                    stats->check_subnn_IP_ele_count += skip_fake_id - skip_fake_id_begin;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    };
};  // namespace tribase
}  // namespace tribase