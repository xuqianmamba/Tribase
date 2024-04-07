#pragma once
#include "common.h"
#include "heap.hpp"
#include "stats.h"
#include "utils.h"

namespace tribase {

class IVFScanBase {
   public:
    size_t d;
    size_t k;
    const float* query;

    IVFScanBase(size_t d, size_t k)
        : d(d), k(k) {}

    void set_query(const float* query) {
        this->query = query;
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
                            Stats* stats) = 0;
};

template <MetricType metric, OptLevel opt_level>
class IVFScan : public IVFScanBase {
   public:
    IVFScan(size_t d, size_t k)
        : IVFScanBase(d, k) {}

    void lite_scan_codes(size_t list_size,
                         const float* codes,
                         const size_t* ids,
                         float* simi,
                         idx_t* idxi) override {
        for (size_t i = 0; i < list_size; i++) {
            const float* candicate = codes + i * d;
            float dis = 0;
            if constexpr (metric == MetricType::METRIC_IP) {
                dis = calculatedInnerProduct(query, candicate, d);
                if (dis > simi[0]) {
                    heap_replace_top<metric>(k, simi, idxi, dis, ids[i]);
                }
            } else if constexpr (metric == MetricType::METRIC_L2) {
                dis = calculatedEuclideanDistance(query, candicate, d);
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
                    Stats* stats) override {
        float max_radius;
        float diff_cos, diff_sin;
        float max_radius_plus_centroid2query;
        float inv_two_times_sqrt_max_radius_times_centroid2query;
        float inv_sqrt_centroid2query;
        float point5_times_inv_sqrt_centroid2query;
        float sqrt_simi;

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
            if (if_skip[i]) {
                continue;
            }
            const float* candicate = codes + i * d;
            float dis = 0;

            if constexpr (metric == MetricType::METRIC_IP) {
                dis = calculatedInnerProduct(query, candicate, d);
            } else if constexpr (metric == MetricType::METRIC_L2) {
                dis = calculatedEuclideanDistance(query, candicate, d);
            } else {
                static_assert(false, "Unsupported metric type");
            }

            if constexpr (metric == MetricType::METRIC_L2) {
                if (dis < simi[0]) {
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
            } else {
                if (dis > simi[0]) {
                    heap_replace_top<metric>(k, simi, idxi, dis, ids[i]);
                }
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
                        for (size_t skip_fake_id = skip_fake_id_begin; skip_fake_id < skip_fake_id_end;
                             skip_fake_id++) {
                            size_t skip_true_id = nearest_IP_id[skip_fake_id];
                            if (skip_true_id > 0 && nearest_IP_dis[skip_fake_id] > cut_degree_cos_minus) {
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

            if (opt_level & OptLevel::OPT_SUBNN_L2) {
#ifdef DEBUG
                stats->check_subnn_L2_count += 2;
#endif
                size_t skip_fake_id_begin = i * sub_k;
                size_t skip_fake_id_end = skip_fake_id_begin + sub_k;
                for (size_t skip_fake_id = skip_fake_id_begin; skip_fake_id < skip_fake_id_end; skip_fake_id++) {
                    float tmp_plus = nearest_L2_dis[skip_fake_id] + sqrt_simi;
                    size_t skip_true_id = nearest_L2_id[skip_fake_id];
                    std::cout << dis << " " << tmp_plus * tmp_plus << std::endl;
                    if (skip_true_id > 0 && dis > tmp_plus * tmp_plus) {  // already sqrt nearest_L2_dis
#ifdef DEBUG
                        if (!if_skip[skip_true_id] && i < skip_true_id) {
                            stats->skip_subnn_L2_count++;
                        }
#endif
                        if_skip[skip_true_id] = true;
                    } else {
#ifdef DEBUG
                        stats->check_subnn_L2_ele_count += skip_fake_id - skip_fake_id_begin;
#endif
                        break;
                    }
                }
            }
        }
    };
};
}  // namespace tribase