#include <limits>
#include <queue>
#include <utility>

#include "common.h"

namespace tribase {

template <MetricType metric = MetricType::METRIC_L2>
inline void heap_init(size_t k, float* bh_val, idx_t* bh_ids) {
    static_assert(std::is_signed<idx_t>::value, "idx_t must be signed");
#pragma omp parallel for
    for (size_t i = 0; i < k; i++) {
        if constexpr (metric == MetricType::METRIC_L2) {
            bh_val[i] = std::numeric_limits<float>::max();
        } else {
            bh_val[i] = std::numeric_limits<float>::lowest();
        }
        bh_ids[i] = -1;
    }
}

void init_result(MetricType metric, size_t size, float* dis, idx_t* ids) {
    if (metric == MetricType::METRIC_L2) {
        heap_init<MetricType::METRIC_L2>(size, dis, ids);
    } else {
        heap_init<MetricType::METRIC_IP>(size, dis, ids);
    }
}

template <MetricType metric = MetricType::METRIC_L2>
inline void heap_replace_top(size_t k, float* bh_val, idx_t* bh_ids, float val, idx_t id) {
    bh_ids--;
    bh_val--;
    size_t i = 1, i1, i2;
    constexpr auto is_greater = [&](float a, float b) {
        if constexpr (metric == MetricType::METRIC_L2) {
            return a > b;
        } else {
            return a < b;
        }
    };
    while (1) {
        i1 = i * 2;
        i2 = i1 + 1;
        if (i1 > k)
            break;

        if ((i2 == k + 1) || is_greater(bh_val[i1], bh_val[i2])) {
            if (is_greater(val, bh_val[i1])) {
                break;
            }
            bh_val[i] = bh_val[i1];
            bh_ids[i] = bh_ids[i1];
            i = i1;
        } else {
            if (is_greater(val, bh_val[i2])) {
                break;
            }
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            i = i2;
        }
    }
    bh_val[i] = val;
    bh_ids[i] = id;
}

}  // namespace tribase