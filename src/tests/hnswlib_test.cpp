#include "hnswlib/hnswlib.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <argparse/argparse.hpp>
#include <filesystem>
#include "tribase.h"

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("hnswlibtest");
    program.add_argument("--benchmarks_path").help("benchmarks path").default_value(std::string("./benchmarks"));
    program.add_argument("--dataset").help("dataset name").default_value(std::string("sift10k"));
    program.add_argument("--k").help("knn").default_value(size_t(1)).action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--tag").help("csv tag").default_value(std::string(""));
    program.add_argument("--pure_hnsw").help("use pure hnsw").default_value(false).implicit_value(true);
    program.add_argument("--build_only").help("build only").default_value(false).implicit_value(true);
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    std::string csv_tag = program.get<std::string>("tag");
    if (csv_tag.empty()) {
        csv_tag = std::format("{:%m-%d-%H-%M}", std::chrono::system_clock::now());
    }
    std::string benchmarks_path = program.get<std::string>("benchmarks_path");
    std::string dataset = program.get<std::string>("dataset");
    std::string base_path = std::format("{}/{}/origin/{}_base.fvecs", benchmarks_path, dataset, dataset);
    std::string query_path = std::format("{}/{}/origin/{}_query.fvecs", benchmarks_path, dataset, dataset);

    bool pure_hnsw = program.get<bool>("pure_hnsw");
    bool build_only = program.get<bool>("build_only");

    std::string index_path = std::format("{}/{}/index/hnsw_index_tri={}.index", benchmarks_path, dataset, !pure_hnsw);

    using namespace tribase;
    CsvWriter csv_add_time(std::format("logs/hnswlib_add_time_{}.csv", csv_tag), {"dataset", "pure_hnsw",
                                                                             "ef_construction",
                                                                             "time"});

    auto [codes, nb, d] = loadFvecs(base_path);
    auto [queries, nq, _] = loadFvecs(query_path);
    size_t k = program.get<size_t>("k");
    size_t gt_k = 100;

    size_t nlist = sqrt(nb);

    std::string groundtruth_path = std::format("{}/{}/result/groundtruth_{}.bin", benchmarks_path, dataset, gt_k);

    std::unique_ptr<float[]> dis = std::make_unique<float[]>(nq * k);
    std::unique_ptr<idx_t[]> ids = std::make_unique<idx_t[]>(nq * k);

    int dim = d;
    int max_elements = nb;      // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 500;  // Controls index search speed/build speed tradeoff
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction, 100UL, false, !pure_hnsw);

    float* data = codes.get();
    float* query_data = queries.get();

    auto start_add = std::chrono::high_resolution_clock::now();
    alg_hnsw->addPoint(data, 0);
    std::cout << "Adding..." << std::endl;
#pragma omp parallel for
    for (int i = 1; i < nb; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }
    auto end_add = std::chrono::high_resolution_clock::now();
    double time_add = std::chrono::duration_cast<std::chrono::microseconds>(end_add - start_add).count() / 1e6;
    csv_add_time << dataset << pure_hnsw << ef_construction << time_add << std::endl;  // log

    if (build_only) {
        alg_hnsw->saveIndex(index_path);
        CsvWriter csv_index_size(std::format("logs/hnswlib_index_size_{}.csv", csv_tag), {"dataset", "tri", "size(MB)"});
        // std::cout << std::format("Index ({}) size: {} MB", index_path, std::filesystem::file_size(index_path) / 1024 / 1024) << std::endl;
        csv_index_size << dataset << !pure_hnsw << 1.0 * std::filesystem::file_size(index_path) / 1024 / 1024 << std::endl;
        return 0;
    }
    CsvWriter csv_recall(std::format("logs/hnswlib_recall_{}.csv", csv_tag), {"dataset", "k", "ef", "tri_ef", "time", "min_time", "recall"});

    Index index(d, nlist, 0, METRIC_L2, OPT_TRIANGLE, 15, 1, 1, false);
    index.train(nb, codes.get());
    index.add(nb, codes.get());
    auto start = std::chrono::high_resolution_clock::now();
    index.nprobe = nlist;
    Stats stats = index.search(nq, queries.get(), k, dis.get(), ids.get());
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;
    
    // loadResults(groundtruth_path,ids.get(), dis.get(), nq, k);

    std::vector<size_t> efs;
    // = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
    for (size_t ef = k; ef < 30; ef++) {
        efs.push_back(ef);
    }
    for (size_t ef = std::max((size_t)30, k); ef < 100; ef += 10) {
        efs.push_back(ef);
    }
    // for (size_t ef = std::max((size_t)100, k); ef < 1000; ef += 100) {
    //     efs.push_back(ef);
    // }
    std::cout << "Testing..." << std::endl;
    bool stop_flag = false;
    for (size_t ef : efs) {
        const int loop = 5;
        std::unique_ptr<float[]> hnswlib_dis = std::make_unique<float[]>(nq * k);
        std::unique_ptr<idx_t[]> hnswlib_ids = std::make_unique<idx_t[]>(nq * k);
        alg_hnsw->setEf(ef);

        //         auto start3 = std::chrono::high_resolution_clock::now();
        //         for (int t = 0; t < loop; t++) {
        // #pragma omp parallel for
        //             for (int i = 0; i < nq; i++) {
        //                 auto ret = alg_hnsw->searchKnn(query_data + i * d, k);
        //                 size_t sz = k - 1;
        //                 while (!ret.empty()) {
        //                     hnswlib_dis[i * k + sz] = ret.top().first;
        //                     hnswlib_ids[i * k + sz] = ret.top().second;
        //                     ret.pop();
        //                 }
        //             }
        //         }
        //         auto end3 = std::chrono::high_resolution_clock::now();
        //         double time3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3).count() / 1e6 / loop;

        //         double hnswlib_recall = calculate_recall(hnswlib_ids.get(), hnswlib_dis.get(), ids.get(), dis.get(), nq, k, MetricType::METRIC_L2);

        for (int64_t delta = 1; ef >= delta; delta += 4 * (delta >= 30) + 1) {
            if (pure_hnsw) {
                delta = ef;
            }
            size_t tri_ef = ef - delta;
            auto start4 = std::chrono::high_resolution_clock::now();
            double min_time = 1e9;
            for (int t = 0; t < loop; t++) {
                auto start5 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
                for (int i = 0; i < nq; i++) {
                    auto ret = alg_hnsw->searchKnn(query_data + i * d, k, nullptr, tri_ef);
                    size_t sz = k - 1;
                    while (!ret.empty()) {
                        hnswlib_dis[i * k + sz] = ret.top().first;
                        hnswlib_ids[i * k + sz] = ret.top().second;
                        ret.pop();
                    }
                }
                auto end5 = std::chrono::high_resolution_clock::now();
                double time5 = std::chrono::duration_cast<std::chrono::microseconds>(end5 - start5).count() / 1e6;
                min_time = std::min(min_time, time5);
            }
            auto end4 = std::chrono::high_resolution_clock::now();
            double time4 = std::chrono::duration_cast<std::chrono::microseconds>(end4 - start4).count() / 1e6 / loop;
            double hnswlib_recall2 = calculate_recall(hnswlib_ids.get(), hnswlib_dis.get(), ids.get(), dis.get(), nq, k, MetricType::METRIC_L2);
            csv_recall << dataset << k << ef << tri_ef << time4 << min_time << hnswlib_recall2 << std::endl;  // log
            if (hnswlib_recall2 >= 1 - 1e-7 && tri_ef == 0) {
                stop_flag = true;
                break;
            }
        }
        if (stop_flag) {
            break;
        }
    }
}

// int main(int argc, char* argv[]) {
//     argparse::ArgumentParser program("hnswlibtest");
//     program.add_argument("--benchmarks_path").help("benchmarks path").default_value(std::string("../benchmarks"));
//     program.add_argument("--dataset").help("dataset name").default_value(std::string("msong"));
//     program.add_argument("--k").help("knn").default_value(size_t(100)).action([](const std::string& value) -> size_t { return std::stoul(value); });
//     try {
//         program.parse_args(argc, argv);
//     } catch (const std::runtime_error& err) {
//         std::cerr << err.what() << std::endl;
//         std::cerr << program;
//         return 1;
//     }
//     std::string benchmarks_path = program.get<std::string>("benchmarks_path");
//     std::string dataset = program.get<std::string>("dataset");
//     std::string base_path = std::format("{}/{}/origin/{}_base.fvecs", benchmarks_path, dataset, dataset);
//     std::string query_path = std::format("{}/{}/origin/{}_query.fvecs", benchmarks_path, dataset, dataset);

//     using namespace tribase;
//     auto [codes, nb, d] = loadFvecs(base_path);
//     auto [queries, nq, _] = loadFvecs(query_path);
//     size_t k = program.get<size_t>("k");
//     size_t gt_k = 100;

//     size_t nlist = sqrt(nb);

//     std::string groundtruth_path = std::format("{}/{}/result/groundtruth_{}.bin", benchmarks_path, dataset, gt_k);
//     std::unique_ptr<idx_t[]> ground_truth_I = std::make_unique<idx_t[]>(k * nq);
//     std::unique_ptr<float[]> ground_truth_D = std::make_unique<float[]>(k * nq);
//     loadResults(groundtruth_path, ground_truth_I.get(), ground_truth_D.get(), nq, k);

//     Index index(d, nlist, 0, METRIC_L2, OPT_TRIANGLE, 15, 1, 1, false);
//     index.train(nb, codes.get());
//     index.add(nb, codes.get());

//     faiss::IndexFlatL2 quantizer(d);  // the other index
//     faiss::IndexIVFFlat index_faiss(&quantizer, d, nlist);
//     index_faiss.nprobe = nlist;
//     index_faiss.train(nb, codes.get());
//     index_faiss.add(nb, codes.get());

//     std::unique_ptr<float[]> dis = std::make_unique<float[]>(nq * k);
//     std::unique_ptr<idx_t[]> ids = std::make_unique<idx_t[]>(nq * k);

//     auto start = std::chrono::high_resolution_clock::now();
//     index.nprobe = nlist;
//     Stats stats = index.search(nq, queries.get(), k, dis.get(), ids.get());
//     auto end = std::chrono::high_resolution_clock::now();
//     double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;

//     std::unique_ptr<float[]> faiss_dis = std::make_unique<float[]>(nq * k);
//     std::unique_ptr<idx_t[]> faiss_ids = std::make_unique<idx_t[]>(nq * k);

//     auto start2 = std::chrono::high_resolution_clock::now();
//     index_faiss.nprobe = nlist;
//     index_faiss.search(nq, queries.get(), k, faiss_dis.get(), faiss_ids.get());
//     auto end2 = std::chrono::high_resolution_clock::now();
//     double time2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() / 1e6;

//     std::cout << std::format("Time_our:\t{}\nTime_faiss:\t{}\n", time, time2);
//     double recall = calculate_recall(ids.get(), dis.get(), faiss_ids.get(), faiss_dis.get(), nq, k, MetricType::METRIC_L2);
//     // result_info(ids.get(), dis.get(), ground_truth_I.get(), ground_truth_D.get(), nq, k, MetricType::METRIC_L2);
//     // double faiss_recall = calculate_recall(faiss_ids.get(), faiss_dis.get(), ground_truth_I.get(), ground_truth_D.get(), nq, k, MetricType::METRIC_L2);
//     double faiss_recall = 1;
//     // result_info(faiss_ids.get(), faiss_dis.get(), ground_truth_I.get(), ground_truth_D.get(), nq, k, MetricType::METRIC_L2);
//     std::cout << recall << " " << faiss_recall << std::endl;

//     stats.recall = recall;
//     stats.print();
//     {
//         int dim = d;
//         int max_elements = nb;      // Maximum number of elements, should be known beforehand
//         int M = 16;                 // Tightly connected with internal dimensionality of the data
//                                     // strongly affects the memory consumption
//         int ef_construction = 500;  // Controls index search speed/build speed tradeoff
//         hnswlib::L2Space space(dim);
//         hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

//         float* data = codes.get();
//         float* query_data = queries.get();
//         alg_hnsw->addPoint(data, 0);
// #pragma omp parallel for
//         for (int i = 1; i < nb; i++) {
//             alg_hnsw->addPoint(data + i * dim, i);
//         }

//         std::vector<size_t> efs;
//         // = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
//         for (size_t ef = k; ef < 30; ef++) {
//             efs.push_back(ef);
//         }
//         for (size_t ef = std::max((size_t)30, k); ef < 100; ef += 10) {
//             efs.push_back(ef);
//         }
//         for (size_t ef = std::max((size_t)100, k); ef < 1000; ef += 100) {
//             efs.push_back(ef);
//         }
//         for (size_t ef : efs) {
//             std::unique_ptr<float[]> hnswlib_dis = std::make_unique<float[]>(nq * k);
//             std::unique_ptr<idx_t[]> hnswlib_ids = std::make_unique<idx_t[]>(nq * k);
//             alg_hnsw->setEf(ef);

//             auto start3 = std::chrono::high_resolution_clock::now();
// #pragma omp parallel for
//             for (int i = 0; i < nq; i++) {
//                 auto ret = alg_hnsw->searchKnn(query_data + i * d, k);
//                 size_t sz = k - 1;
//                 while (!ret.empty()) {
//                     hnswlib_dis[i * k + sz] = ret.top().first;
//                     hnswlib_ids[i * k + sz] = ret.top().second;
//                     ret.pop();
//                 }
//             }
//             auto end3 = std::chrono::high_resolution_clock::now();
//             double time3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3).count() / 1e6;
//             std::cout << std::format("Time_our:\t{}\nTime_faiss:\t{}\nTime_hnswlib:\t{}\n", time, time2, time3);

//             double hnswlib_recall = calculate_recall(hnswlib_ids.get(), hnswlib_dis.get(), faiss_ids.get(), faiss_dis.get(), nq, k, MetricType::METRIC_L2);
//             // std::cout << "ef:" << ef << "\t" << "hnswlib_recall:" << hnswlib_recall << std::endl;
//             std::cout << std::format("ef: {}\thnswlib_recall: {}\n", ef, hnswlib_recall);
//         }
//     }
// }