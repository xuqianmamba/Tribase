#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <chrono>
#include <format>
#include <memory>
#include <random>
#include "gtest/gtest.h"
#include "tribase.h"

double test_dim(size_t d = 16) {
    using namespace tribase;
    [[maybe_unused]] size_t nb = 1000000, nq = 1000, k = 10;
    std::unique_ptr<float[]> codes = std::make_unique<float[]>(nb * d);
    std::unique_ptr<float[]> queries = std::make_unique<float[]>(nq * d);
    size_t nlist = sqrt(nb);

#pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0, 1.0);
        int thread_id = omp_get_thread_num();
        int thread_num = omp_get_num_threads();
        for (size_t i = thread_id; i < nb * d; i += thread_num) {
            codes[i] = dist(gen);
        }
        for (size_t i = thread_id; i < nq * d; i += thread_num) {
            queries[i] = dist(gen);
        }
    }

    Index index(d, nlist, nlist, METRIC_L2, OPT_TRIANGLE, 15, 1, 1, false);
    index.train(nb, codes.get());
    index.add(nb, codes.get());

    faiss::IndexFlatL2 quantizer(d);  // the other index
    faiss::IndexIVFFlat index_faiss(&quantizer, d, nlist);
    index_faiss.nprobe = nlist;
    index_faiss.train(nb, codes.get());
    index_faiss.add(nb, codes.get());

    std::unique_ptr<float[]> dis = std::make_unique<float[]>(nq * k);
    std::unique_ptr<idx_t[]> ids = std::make_unique<idx_t[]>(nq * k);

    const int loop = 10;

    auto start = std::chrono::high_resolution_clock::now();
    index.nprobe = nlist;
    Stats stats;
    for (int i = 0; i < loop; i++) {
        stats = index.search(nq, queries.get(), k, dis.get(), ids.get());
    }
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6 / loop;

    std::unique_ptr<float[]> g_dis = std::make_unique<float[]>(nq * k);
    std::unique_ptr<idx_t[]> g_ids = std::make_unique<idx_t[]>(nq * k);

    auto start2 = std::chrono::high_resolution_clock::now();
    index_faiss.nprobe = nlist;
    for (int i = 0; i < loop; i++) {
        index_faiss.search(nq, queries.get(), k, g_dis.get(), g_ids.get());
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    double time2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() / 1e6 / loop;

    std::cout << std::format("Time_our: {:.3f}\tTime_faiss: {:.3f}\tSpeedup: {:.3f}\n", time, time2, time2 / time);
    double recall = calculate_recall(ids.get(), dis.get(), g_ids.get(), g_dis.get(), nq, k, MetricType::METRIC_L2);
    // ASSERT_FLOAT_EQ(recall, 1);
    return time2 / time;
}

TEST(IndexDimTest, L2Test) {
    const int global_loop = 5;
    auto dims = {4, 8, 12, 16, 32, 64, 128};
    for(auto dim: dims){
        std::cout << dim << std::endl;
        double sum = 0;
        for (int i = 0; i < global_loop; i++) {
            sum += test_dim(dim);
        }
        std::cout << sum / global_loop << std::endl;
    }
}