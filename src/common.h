#pragma once
#define DEBUG
#include <inttypes.h>

#include <utility>

namespace tribase {
enum MetricType { METRIC_INNER_PRODUCT = 0,
                  METRIC_L2 };

enum OptLevel {
    OPT_NONE = 0b000,
    OPT_TRIANGLE = 0b001,
    OPT_SUBNN_L2 = 0b010,
    OPT_SUBNN_IP = 0b100,
    OPT_TRI_SUBNN_L2 = 0b011,
    OPT_TRI_SUBNN_IP = 0b101,
    OPT_ALL = 0b111
};

using idx_t = int64_t;
using result_t = std::pair<float, idx_t>;

}  // namespace tribase