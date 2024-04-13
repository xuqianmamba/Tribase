#include "Index.h"
#include <chrono>
#include <format>
#include <memory>
#include <random>
#include "common.h"
#include "gtest/gtest.h"

TEST(IndexTest, NoneTest) {
    using namespace tribase;
    [[maybe_unused]] int nb = 27134, nq = 1000, d = 420, k = 32;
    std::unique_ptr<float[]> codes = std::make_unique<float[]>(nb * d);
    std::unique_ptr<float[]> queries = std::make_unique<float[]>(nq * d);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (int i = 0; i < nb * d; i++) {
        codes[i] = dist(gen);
    }
    for (int i = 0; i < nq * d; i++) {
        queries[i] = dist(gen);
    }

    Index index(d, 1, 1, METRIC_L2, OPT_TRIANGLE, 15, 1, 1, false);
    index.train(nb, codes.get());
    index.add(nb, codes.get());

    std::unique_ptr<float[]> dis = std::make_unique<float[]>(nq * k);
    std::unique_ptr<idx_t[]> ids = std::make_unique<idx_t[]>(nq * k);

    auto start = std::chrono::high_resolution_clock::now();
    index.search(nq, queries.get(), k, dis.get(), ids.get());
    auto end = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;
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