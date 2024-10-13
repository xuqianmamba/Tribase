#pragma once
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include "common.h"
#include "utils.h"

namespace tribase {
class Stats {
   public:
    size_t total_count;

    // triangle part
    size_t skip_triangle_count;
    size_t skip_triangle_large_count;

    // subNN part
    size_t check_subnn_L2_ele_count;
    size_t check_subnn_IP_ele_count;
    size_t check_subnn_L2_count;
    size_t check_subnn_IP_count;
    size_t skip_subnn_L2_count;
    size_t skip_subnn_IP_count;

    size_t simi_update_count;
    size_t dis_calculate_count;

    size_t nlist;
    size_t nprobe;

    double faiss_query_time;
    double query_time;

    OptLevel opt_level;

    double recall;
    double r2;
    float simi_ratio;

    // summary
   private:
    float pruning_triangle;
    float pruning_triangle_large;
    float pruning_subnn_L2;
    float pruning_subnn_IP;
    float check_subnn_L2;
    float check_subnn_IP;

    float simi_update_rate;

    float time_speedup;
    float pruning_speedup;

   public:
    void reset() {
        total_count = 0;
        skip_triangle_count = 0;
        skip_triangle_large_count = 0;
        check_subnn_L2_ele_count = 0;
        check_subnn_IP_ele_count = 0;
        check_subnn_L2_count = 0;
        check_subnn_IP_count = 0;
        skip_subnn_L2_count = 0;
        skip_subnn_IP_count = 0;
        simi_update_count = 0;
    }

    Stats() { reset(); }

    void summary() {
        dis_calculate_count = total_count - skip_triangle_count - skip_triangle_large_count - skip_subnn_L2_count - skip_subnn_IP_count;
        simi_update_rate = dis_calculate_count == 0 ? 0 : 100.0 * simi_update_count / dis_calculate_count;
        pruning_triangle = skip_triangle_count == 0 ? 0 : 100.0 * skip_triangle_count / total_count;
        pruning_triangle_large = skip_triangle_large_count == 0 ? 0 : 100.0 * skip_triangle_large_count / total_count;
        pruning_subnn_L2 = skip_subnn_L2_count == 0 ? 0 : 100.0 * skip_subnn_L2_count / total_count;
        pruning_subnn_IP = skip_subnn_IP_count == 0 ? 0 : 100.0 * skip_subnn_IP_count / total_count;
        check_subnn_IP = check_subnn_IP_count == 0 ? 0 : 1.0 * check_subnn_IP_ele_count / check_subnn_IP_count;
        check_subnn_L2 = check_subnn_L2_count == 0 ? 0 : 1.0 * check_subnn_L2_ele_count / check_subnn_L2_count;

        time_speedup = query_time == 0 ? 0 : 100.0 * faiss_query_time / query_time;
        pruning_speedup = dis_calculate_count == 0 ? 0 : 100.0 * total_count / dis_calculate_count;
    }

    void print() {
        summary();
        std::cout << std::format("nprobe:{} opt_level: {} simi_ratio: {}\n", nprobe, static_cast<int>(opt_level), simi_ratio)
                  << std::format("tri: {}({:.2f}%) tri_large: {}({:.2f}%) subnn_L2: {}({:.2f}%) subnn_IP: {}({:.2f}%)\n", skip_triangle_count, pruning_triangle, skip_triangle_large_count, pruning_triangle_large, skip_subnn_L2_count, pruning_subnn_L2, skip_subnn_IP_count, pruning_subnn_IP)
                  << std::format("simi_update_rate: {:.2f}% check_L2: {} check_IP: {}\n", simi_update_rate, check_subnn_L2, check_subnn_IP)
                  << std::format("time_speedup: {:.2f}% pruning_speedup: {:.2f}% faiss_query_time: {:.6f} query_time: {:.6f}\n", time_speedup, pruning_speedup, faiss_query_time, query_time)
                  << std::format("recall: {} r2: {}\n", recall, r2);
    }

    void toCsv(std::string filename, bool append, std::string dataset = "Unknown") {
        CsvWriter writer(filename,
                         {"dataset", "nlist", "nprobe", "opt_level", "simi_ratio",
                          "tri", "tri_large", "subnn_L2", "subnn_IP", "simi_update_rate",
                          "check_L2", "check_IP",
                          "time_speedup", "pruning_speedup", "query_time",
                          "recall", "r2"},
                         append, false);
        summary();
        writer << dataset << nlist << nprobe << static_cast<int>(opt_level) << simi_ratio
               << skip_triangle_count << skip_triangle_large_count << skip_subnn_L2_count << skip_subnn_IP_count << simi_update_rate
               << check_subnn_L2 << check_subnn_IP
               << time_speedup << pruning_speedup << query_time
               << recall << r2 << std::endl;
    }
};

inline Stats mergeStats(std::vector<Stats>& stats) {
    Stats merged;
    for (auto& s : stats) {
        merged.total_count += s.total_count;
        merged.skip_triangle_count += s.skip_triangle_count;
        merged.skip_triangle_large_count += s.skip_triangle_large_count;
        merged.check_subnn_L2_ele_count += s.check_subnn_L2_ele_count;
        merged.check_subnn_IP_ele_count += s.check_subnn_IP_ele_count;
        merged.check_subnn_L2_count += s.check_subnn_L2_count;
        merged.check_subnn_IP_count += s.check_subnn_IP_count;
        merged.skip_subnn_L2_count += s.skip_subnn_L2_count;
        merged.skip_subnn_IP_count += s.skip_subnn_IP_count;
        merged.simi_update_count += s.simi_update_count;
    }
    return merged;
}
}  // namespace tribase