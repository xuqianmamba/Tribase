#include <memory>
#include "IVFScan.hpp"
#include "common.h"
#include "gtest/gtest.h"

TEST(ScanTest, LiteScan) {
    using namespace tribase;
    [[maybe_unused]] int n = 100;
    [[maybe_unused]] std::unique_ptr<IVFScanBase> scaner = std::make_unique<IVFScan<MetricType::METRIC_L2, OptLevel::OPT_NONE>>(128, 10);
}