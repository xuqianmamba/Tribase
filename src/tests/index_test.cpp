#include "Index.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <chrono>
#include <format>
#include <memory>
#include <random>
#include "common.h"
#include "gtest/gtest.h"

TEST(IndexTest, L2Test) {
    using namespace tribase;
    [[maybe_unused]] size_t nb = 20000, nq = 100, d = 16, k = 10;
    std::unique_ptr<float[]> codes = std::make_unique<float[]>(nb * d);
    std::unique_ptr<float[]> queries = std::make_unique<float[]>(nq * d);
    size_t nlist = sqrt(nb);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (size_t i = 0; i < nb * d; i++) {
        codes[i] = dist(gen);
    }
    for (size_t i = 0; i < nq * d; i++) {
        queries[i] = dist(gen);
    }

    Index index(d, nlist, nlist, METRIC_L2, OPT_ALL, 15, 1, 1, false);
    index.train(nb, codes.get());
    index.add(nb, codes.get());

    faiss::IndexFlatL2 quantizer(d);  // the other index
    faiss::IndexIVFFlat index_faiss(&quantizer, d, nlist);
    index_faiss.nprobe = nlist;
    index_faiss.train(nb, codes.get());
    index_faiss.add(nb, codes.get());

    std::unique_ptr<float[]> dis = std::make_unique<float[]>(nq * k);
    std::unique_ptr<idx_t[]> ids = std::make_unique<idx_t[]>(nq * k);

    auto start = std::chrono::high_resolution_clock::now();
    index.nprobe = nlist;
    Stats stats = index.search(nq, queries.get(), k, dis.get(), ids.get());
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;

    std::unique_ptr<float[]> g_dis = std::make_unique<float[]>(nq * k);
    std::unique_ptr<idx_t[]> g_ids = std::make_unique<idx_t[]>(nq * k);

    auto start2 = std::chrono::high_resolution_clock::now();
    index_faiss.nprobe = 1;
    index_faiss.search(nq, queries.get(), k, g_dis.get(), g_ids.get());
    auto end2 = std::chrono::high_resolution_clock::now();
    double time2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() / 1e6;

    std::cout << std::format("Time_our:\t{}\nTime_faiss:\t{}\n", time, time2);
    double recall = calculate_recall(ids.get(), dis.get(), g_ids.get(), g_dis.get(), nq, k, MetricType::METRIC_L2);
    stats.recall = recall;
    stats.print();
    ASSERT_FLOAT_EQ(recall, 1);
}

TEST(IndexTest, IPTest) {
    using namespace tribase;
    [[maybe_unused]] size_t nb = 20000, nq = 100, d = 16, k = 10;
    std::unique_ptr<float[]> codes = std::make_unique<float[]>(nb * d);
    std::unique_ptr<float[]> queries = std::make_unique<float[]>(nq * d);
    size_t nlist = sqrt(nb);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (size_t i = 0; i < nb * d; i++) {
        codes[i] = dist(gen);
    }
    for (size_t i = 0; i < nb; i++) {
        float norm = calculatedInnerProduct(codes.get() + i * d, codes.get() + i * d, d);
        if (norm > 0) {
            for (size_t j = 0; j < d; j++) {
                codes[i * d + j] /= sqrt(norm);
            }
        }
    }

    for (size_t i = 0; i < nq * d; i++) {
        queries[i] = dist(gen);
    }
    for (size_t i = 0; i < nq; i++) {
        float norm = calculatedInnerProduct(queries.get() + i * d, queries.get() + i * d, d);
        if (norm > 0) {
            for (size_t j = 0; j < d; j++) {
                queries[i * d + j] /= sqrt(norm);
            }
        }
    }

    Index index(d, nlist, nlist, METRIC_IP, OPT_TRIANGLE, 15, 1, 1, false);
    index.train(nb, codes.get());
    index.add(nb, codes.get());

    faiss::IndexFlatIP quantizer(d);  // the other index
    faiss::IndexIVFFlat index_faiss(&quantizer, d, nlist, faiss::METRIC_INNER_PRODUCT);
    index_faiss.nprobe = nlist;
    index_faiss.train(nb, codes.get());
    index_faiss.add(nb, codes.get());

    std::unique_ptr<float[]> dis = std::make_unique<float[]>(nq * k);
    std::unique_ptr<idx_t[]> ids = std::make_unique<idx_t[]>(nq * k);

    auto start = std::chrono::high_resolution_clock::now();
    index.nprobe = nlist;
    Stats stats = index.search(nq, queries.get(), k, dis.get(), ids.get());
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;

    std::unique_ptr<float[]> g_dis = std::make_unique<float[]>(nq * k);
    std::unique_ptr<idx_t[]> g_ids = std::make_unique<idx_t[]>(nq * k);

    auto start2 = std::chrono::high_resolution_clock::now();
    index_faiss.nprobe = nlist;
    index_faiss.search(nq, queries.get(), k, g_dis.get(), g_ids.get());
    auto end2 = std::chrono::high_resolution_clock::now();
    double time2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() / 1e6;

    std::cout << std::format("Time_our:\t{}\nTime_faiss:\t{}\n", time, time2);
    double recall = calculate_recall(ids.get(), dis.get(), g_ids.get(), g_dis.get(), nq, k, MetricType::METRIC_IP);
    stats.recall = recall;
    stats.print();
    ASSERT_FLOAT_EQ(recall, 1);
}

TEST(IndexTest, DatasetTest) {
    using namespace tribase;
    auto [codes, nb, d] = loadFvecs("../benchmarks/msong/origin/msong_base.fvecs");
    auto [queries, nq, _] = loadFvecs("../benchmarks/msong/origin/msong_query.fvecs");
    size_t k = 100;

    size_t nlist = sqrt(nb);

    for (size_t i = 0; i < nb; i++) {
        float norm = calculatedInnerProduct(codes.get() + i * d, codes.get() + i * d, d);
        if (norm > 0) {
            for (size_t j = 0; j < d; j++) {
                codes[i * d + j] /= sqrt(norm);
            }
        }
    }

    for (size_t i = 0; i < nq; i++) {
        float norm = calculatedInnerProduct(queries.get() + i * d, queries.get() + i * d, d);
        if (norm > 0) {
            for (size_t j = 0; j < d; j++) {
                queries[i * d + j] /= sqrt(norm);
            }
        }
    }

    Index index(d, nlist, nlist, METRIC_IP, OPT_TRIANGLE, 15, 1, 1, false);
    index.train(nb, codes.get());
    index.add(nb, codes.get());

    faiss::IndexFlatIP quantizer(d);  // the other index
    faiss::IndexIVFFlat index_faiss(&quantizer, d, nlist, faiss::METRIC_INNER_PRODUCT);
    index_faiss.nprobe = nlist;
    index_faiss.train(nb, codes.get());
    index_faiss.add(nb, codes.get());

    std::unique_ptr<float[]> dis = std::make_unique<float[]>(nq * k);
    std::unique_ptr<idx_t[]> ids = std::make_unique<idx_t[]>(nq * k);

    auto start = std::chrono::high_resolution_clock::now();
    index.nprobe = nlist;
    Stats stats = index.search(nq, queries.get(), k, dis.get(), ids.get());
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;

    std::unique_ptr<float[]> g_dis = std::make_unique<float[]>(nq * k);
    std::unique_ptr<idx_t[]> g_ids = std::make_unique<idx_t[]>(nq * k);

    auto start2 = std::chrono::high_resolution_clock::now();
    index_faiss.nprobe = nlist;
    index_faiss.search(nq, queries.get(), k, g_dis.get(), g_ids.get());
    auto end2 = std::chrono::high_resolution_clock::now();
    double time2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() / 1e6;

    std::cout << std::format("Time_our:\t{}\nTime_faiss:\t{}\n", time, time2);
    double recall = calculate_recall(ids.get(), dis.get(), g_ids.get(), g_dis.get(), nq, k, MetricType::METRIC_IP);
    stats.recall = recall;
    stats.print();
    ASSERT_FLOAT_EQ(recall, 1);
}