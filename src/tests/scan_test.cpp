#include <chrono>
#include <format>
#include <memory>
#include <random>
#include "IVFScan.hpp"
#include "common.h"
#include "gtest/gtest.h"

TEST(ScanTest, LiteScan) {
    using namespace tribase;
    [[maybe_unused]] int nb = 27134, nq = 1000, d = 420, k = 32;
    [[maybe_unused]] std::unique_ptr<IVFScanBase> scaner = std::make_unique<IVFScan<MetricType::METRIC_L2, OptLevel::OPT_NONE, EdgeDevice::EDGEDEVIVE_DISABLED>>(d, k);
    std::unique_ptr<float[]> codes = std::make_unique<float[]>(nb * d);
    std::unique_ptr<size_t[]> code_ids = std::make_unique<size_t[]>(nb);
    std::unique_ptr<float[]> queries = std::make_unique<float[]>(nq * d);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (int i = 0; i < nb * d; i++) {
        codes[i] = dist(gen);
    }
    std::iota(code_ids.get(), code_ids.get() + nb, 0);
    for (int i = 0; i < nq * d; i++) {
        queries[i] = dist(gen);
    }

    std::unique_ptr<float[]> dis = std::make_unique<float[]>(nq * k);
    std::unique_ptr<idx_t[]> ids = std::make_unique<idx_t[]>(nq * k);
    init_result(METRIC_L2, nq * k, dis.get(), ids.get());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nq; i++) {
        scaner->set_query(queries.get() + i * d);
        scaner->lite_scan_codes(nb, codes.get(), code_ids.get(), dis.get() + i * k, ids.get() + i * k);
        sort_result(METRIC_L2, k, dis.get() + i * k, ids.get() + i * k);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    size_t data_size = 1ul * nb * nq * d * sizeof(float);
    // GB/s
    double bandwidth = 1.0 * data_size / time / 1e9;

    std::cout << std::format("Bandwidth: {:.2f} GB/s\tTime: {:.2f} s\n", bandwidth, time);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << std::format("{:d}:{:.2f} ", ids[i * k + j], dis[i * k + j]);
        }
        std::cout << std::endl;
    }
}