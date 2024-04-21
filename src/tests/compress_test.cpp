#include <assert.h>
#include <gtest/gtest.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdfloat>
#include "../compress/repair_compress.h"
#include "../compress/check.h"
#include "../compress/utils.h"
#include "../utils.h"

using namespace repair_compress;

void generate_random_data(value_t* data, size_t n, size_t dim, value_t MAX = 10, size_t seed = 0) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, MAX);
    for (size_t i = 0; i < n * dim; i++) {
        data[i] = dis(gen);
    }
}

TEST(COMPRESS, BASIC) {
#if 1
    size_t n = 20;
    size_t dim = 8;
    std::unique_ptr<value_t[]> udata = std::make_unique<value_t[]>(n * dim);
    generate_random_data(udata.get(), n, dim, 256);
    value_t* data = udata.get();
#else
    auto [float_udata, n, dim] = loadFvecs("/home/xuqian/Triangle/benchmarks/msong/origin/msong_base.fvecs");
    static_assert(sizeof(value_t) == sizeof(float));
    // value_t* data = reinterpret_cast<value_t*>(float_udata.get());
    std::unique_ptr<value_t[]> udata = std::make_unique<value_t[]>(n * dim);

    // covert2f16(reinterpret_cast<int*>(udata.get()), float_udata.get(), n * dim);

    // covert2fixed(reinterpret_cast<int*>(udata.get()), float_udata.get(), n * dim, true, 6, 0);

    auto [s, lb, rb] = autocovert2fixed(udata.get(), float_udata.get(), n * dim, 16);
    std::cout << "s: " << s << ", lb: " << lb << ", rb: " << rb << std::endl;

    value_t* data = udata.get();
#endif
    std::cout << "n: " << n << ", dim: " << dim << std::endl;

    auto start = std::chrono::steady_clock::now();
    auto [rule, result_vlist, result_elist] = generate_rule(data, n, dim, 4, true);
    auto end = std::chrono::steady_clock::now();

    std::cout << std::format("time: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << std::endl;

    pair_reverse_map_t <value_t> rule_reverse;
    for (const auto& kv : rule) {
        rule_reverse[kv.second] = kv.first;
    }


    bool status = check_compress(data, n, dim, result_vlist, result_elist, rule_reverse);

    if (status) {
        std::cout << "Success: The decompressed data matches the original data.\n";
    } else {
        std::cout << "Failure: The decompressed data does not match the original data.\n";
    }



















    if (n * dim < 256) {
        for (auto [pair, value] : rule) {
            std::cout << pair.first << " " << pair.second << " -> " << value << std::endl;
        }

        for (size_t i = 0; i < result_vlist.size(); i++) {
            std::cout << result_vlist[i] << " ";
        }
        std::cout << std::endl;

        for (size_t i = 0; i < result_elist.size(); i++) {
            std::cout << result_elist[i] << " ";
        }
        std::cout << std::endl;
    }
}

TEST(COMPRESS, msong) {
    using namespace tribase;
    auto [float_udata, n, dim] = loadFvecs("/home/xuqian/Triangle/benchmarks/msong/origin/msong_base.fvecs");
    static_assert(sizeof(value_t) == sizeof(float));
    // value_t* data = reinterpret_cast<value_t*>(float_udata.get());
    std::unique_ptr<value_t[]> udata = std::make_unique<value_t[]>(n * dim);

    // covert2f16(reinterpret_cast<int*>(udata.get()), float_udata.get(), n * dim);

    // covert2fixed(reinterpret_cast<int*>(udata.get()), float_udata.get(), n * dim, true, 6, 0);

    auto [s, lb, rb] = autocovert2fixed(udata.get(), float_udata.get(), n * dim, 16);
    std::cout << "s: " << s << ", lb: " << lb << ", rb: " << rb << std::endl;

    

    value_t* data = udata.get();
    std::cout << "n: " << n << ", dim: " << dim << std::endl;

    auto start = std::chrono::steady_clock::now();
    auto [rule, result_vlist, result_elist] = generate_rule(data, n, dim, 4, true);

    pair_reverse_map_t <value_t> rule_reverse;
    for (const auto& kv : rule) {
        rule_reverse[kv.second] = kv.first;
    }

    bool status = check_compress(data, n, dim, result_vlist, result_elist, rule_reverse);

    if (status) {
        std::cout << "Success: The decompressed data matches the original data.\n";
    } else {
        std::cout << "Failure: The decompressed data does not match the original data.\n";
    }















    auto end = std::chrono::steady_clock::now();

    std::cout << std::format("time: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << std::endl;
}

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <gtest/gtest.h>
#include "../compress/utils.h"
#include "../utils.h"

TEST(COMPRESS, Faiss) {
    using namespace tribase;
    using idx_t = faiss::idx_t;

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    auto [base, nb, d] = loadFvecs("/home/xuqian/Triangle/benchmarks/msong/origin/msong_base.fvecs");
    auto [query, nq, _] = loadFvecs("/home/xuqian/Triangle/benchmarks/msong/origin/msong_query.fvecs");

    float* xb = base.get();
    float* xq = query.get();

    int nlist = sqrt(nb);
    int k = 100;

    faiss::IndexFlatL2 quantizer(d);  // the other index
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    assert(!index.is_trained);
    index.train(nb, xb);
    assert(index.is_trained);
    index.add(nb, xb);

#if 1
    std::unique_ptr<int[]> base16 = std::make_unique<int[]>(nb * d);
    auto [s, lb, rb] = autocovert2fixed(base16.get(), base.get(), nb * d, 16);
    std::unique_ptr<float[]> base32 = std::make_unique<float[]>(nb * d);
    revert4fixed(base32.get(), base16.get(), nb * d, s, lb, rb);
#else
    std::unique_ptr<int[]> base16 = std::make_unique<int[]>(nb * d);
    covert2f16(base16.get(), base.get(), nb * d);
    std::unique_ptr<float[]> base32 = std::make_unique<float[]>(nb * d);
    covert2f32(base32.get(), base16.get(), nb * d);
#endif

    faiss::IndexFlatL2 quantizer2(d);  // the other index
    faiss::IndexIVFFlat index2(&quantizer2, d, nlist);

    assert(!index2.is_trained);
    index2.train(nb, base32.get());
    assert(index2.is_trained);
    index2.add(nb, base32.get());

    {  // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        idx_t* GI = new idx_t[k * nq];
        float* GD = new float[k * nq];

        idx_t* I32 = new idx_t[k * nq];
        float* D32 = new float[k * nq];

        index.nprobe = index.nlist;
        index.search(nq, xq, k, GD, GI);

        index.nprobe = 1;
        index.search(nq, xq, k, D, I);

        auto recall = tribase::calculate_recall(I, D, GI, GD, nq, k, tribase::MetricType::METRIC_L2);
        std::cout << "recall: " << recall << std::endl;

        index2.nprobe = 1;
        index2.search(nq, xq, k, D32, I32);
        calculate_true_distance(xb, xq, I32, D32, nq, k, d, nb);

        auto recall32 = tribase::calculate_recall(I32, D32, GI, GD, nq, k, tribase::MetricType::METRIC_L2);
        std::cout << "recall32: " << recall32 << std::endl;

        index2.nprobe = index2.nlist;
        index2.search(nq, xq, k, D32, I32);
        calculate_true_distance(xb, xq, I32, D32, nq, k, d, nb);

        auto full_recall32 = tribase::calculate_recall(I32, D32, GI, GD, nq, k, tribase::MetricType::METRIC_L2);
        std::cout << "full_recall32: " << full_recall32 << std::endl;

        delete[] I;
        delete[] D;
        delete[] GI;
        delete[] GD;
        delete[] I32;
        delete[] D32;
    }
}