#pragma once
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include "common.h"

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

    // summary
   private:
    float check_subnn_L2;
    float check_subnn_IP;

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
    }

    Stats() { reset(); }

    void print() {
        check_subnn_IP = check_subnn_IP_count == 0 ? 0 : check_subnn_IP_ele_count / check_subnn_IP_count;
        check_subnn_L2 = check_subnn_L2_count == 0 ? 0 : check_subnn_L2_ele_count / check_subnn_L2_count;

        std::cout << "total: " << total_count << std::endl
                  << "skip_triangle: " << skip_triangle_count << std::endl
                  << "skip_triangle_large: " << skip_triangle_large_count << std::endl
                  << "skip_subnn_L2: " << skip_subnn_L2_count << std::endl
                  << "skip_subnn_IP: " << skip_subnn_IP_count << std::endl
                  << std::fixed << std::setprecision(3) << "check_subnn_L2: " << check_subnn_L2 << std::endl
                  << "check_subnn_IP: " << check_subnn_IP << std::endl;
    }

    void toCsv(std::string_view filename, bool append) {
        std::ofstream ofs;
        if (append)
            ofs.open(filename.data(), std::ios::app);
        else
            ofs.open(filename.data());
        if (!ofs.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        if (!append) {
            ofs << "total,skip_triangle,skip_triangle_large,skip_subnn_L2,skip_subnn_IP,check_subnn_L2,check_subnn_"
                   "IP\n";
        }
        check_subnn_IP = check_subnn_IP_count == 0 ? 0 : check_subnn_IP_ele_count / check_subnn_IP_count;
        check_subnn_L2 = check_subnn_L2_count == 0 ? 0 : check_subnn_L2_ele_count / check_subnn_L2_count;
        ofs << total_count << "," << skip_triangle_count << "," << skip_triangle_large_count << ","
            << skip_subnn_L2_count << "," << skip_subnn_IP_count << "," << std::fixed << std::setprecision(3)
            << check_subnn_L2 << "," << check_subnn_IP << "\n";
    }
};

Stats mergeStats(std::vector<Stats>& stats) {
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
    }
    return merged;
}  // namespace tribase