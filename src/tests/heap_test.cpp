#include "heap.hpp"
#include <algorithm>
#include <random>
#include "common.h"
#include "gtest/gtest.h"

TEST(HeapTest, MaxHeap) {  // big top heap
    using namespace tribase;
    int k = 20;
    int n = 100;
    float* simi = new float[k];
    idx_t* idxi = new idx_t[k];

    float* vals = new float[n];
    idx_t* idxs = new idx_t[n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1);
    for (int i = 0; i < n; i++) {
        vals[i] = dis(gen);
    }
    std::iota(idxs, idxs + n, 0);

    [[maybe_unused]] auto output = [&]() {
        for (int j = 0; j < k; j++) {
            std::cout << simi[j] << " ";
        }
        std::cout << std::endl;
    };

    // output();
    heap_init<MetricType::METRIC_L2>(k, simi, idxi);
    for (int i = 0; i < n; i++) {
        if (vals[i] < simi[0]) {
            heap_replace_top<MetricType::METRIC_L2>(k, simi, idxi, vals[i], i);
        }
        // output();
    }

    for (int i = 0; i < k; i++) {
        EXPECT_EQ(vals[idxi[i]], simi[i]);
    }

    std::sort(vals, vals + n);
    std::sort(simi, simi + k);
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(vals[i], simi[i]);
    }
}

TEST(HeapTest, MinHeap) {  // small top heap
    using namespace tribase;
    int k = 20;
    int n = 100;
    float* simi = new float[k];
    idx_t* idxi = new idx_t[k];

    float* vals = new float[n];
    idx_t* idxs = new idx_t[n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1);
    for (int i = 0; i < n; i++) {
        vals[i] = dis(gen);
    }
    std::iota(idxs, idxs + n, 0);

    [[maybe_unused]] auto output = [&]() {
        for (int j = 0; j < k; j++) {
            std::cout << simi[j] << " ";
        }
        std::cout << std::endl;
    };

    // output();
    heap_init<MetricType::METRIC_IP>(k, simi, idxi);
    for (int i = 0; i < n; i++) {
        if (vals[i] > simi[0]) {
            heap_replace_top<MetricType::METRIC_IP>(k, simi, idxi, vals[i], i);
        }
        // output();
    }

    for (int i = 0; i < k; i++) {
        EXPECT_EQ(vals[idxi[i]], simi[i]);
    }

    std::sort(vals, vals + n, std::greater<float>());
    std::sort(simi, simi + k, std::greater<float>());
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(vals[i], simi[i]);
    }
}