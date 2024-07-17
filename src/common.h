#pragma once

#ifndef NDEBUG
#define DEBUG
// #define CORRECTNESS_CHECK
#endif

#ifndef NSTATS
#define IF_STATS if (stats)
#else
#define IF_STATS if (false)
#endif

#include <inttypes.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace tribase {
enum MetricType { METRIC_IP = 0,
                  METRIC_L2 };

enum OptLevel {
    OPT_NONE = 0b000,
    OPT_TRIANGLE = 0b001,
    OPT_SUBNN_L2 = 0b010,
    OPT_SUBNN_IP = 0b100,
    OPT_TRI_SUBNN_L2 = 0b011,
    OPT_TRI_SUBNN_IP = 0b101,
    OPT_SUBNN_ONLY = 0b110,
    OPT_ALL = 0b111
};

enum EdgeDevice {
    EDGEDEVIVE_ENABLED,
    EDGEDEVIVE_DISABLED
};

inline bool isLegalOptLevel(int opt) {
    switch (opt) {
        case OPT_NONE:
        case OPT_TRIANGLE:
        case OPT_SUBNN_L2:
        case OPT_SUBNN_IP:
        case OPT_TRI_SUBNN_L2:
        case OPT_TRI_SUBNN_IP:
        case OPT_SUBNN_ONLY:
        case OPT_ALL:
            return true;
        default:
            return false;
    }
}

inline OptLevel str2OptLevel(const std::string& str) {
    try {
        int int_opt = std::stoi(str);
        if (!isLegalOptLevel(int_opt)) {
            throw std::invalid_argument("Invalid optimization level");
        }
        return static_cast<OptLevel>(int_opt);
    } catch (const std::invalid_argument& e) {
        // pass
    }
    if (str == "OPT_NONE") {
        return OptLevel::OPT_NONE;
    } else if (str == "OPT_TRIANGLE") {
        return OptLevel::OPT_TRIANGLE;
    } else if (str == "OPT_SUBNN_L2") {
        return OptLevel::OPT_SUBNN_L2;
    } else if (str == "OPT_SUBNN_IP") {
        return OptLevel::OPT_SUBNN_IP;
    } else if (str == "OPT_TRI_SUBNN_L2") {
        return OptLevel::OPT_TRI_SUBNN_L2;
    } else if (str == "OPT_TRI_SUBNN_IP") {
        return OptLevel::OPT_TRI_SUBNN_IP;
    } else if (str == "OPT_SUBNN_ONLY") {
        return OptLevel::OPT_SUBNN_ONLY;
    } else if (str == "OPT_ALL") {
        return OptLevel::OPT_ALL;
    } else {
        throw std::runtime_error("Invalid optimization level");
    }
}

using idx_t = int64_t;
using result_t = std::pair<float, idx_t>;

}  // namespace tribase