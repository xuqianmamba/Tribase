#include <argparse/argparse.hpp>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include "tribase.h"
#include "utils.h"
using namespace tribase;

bool str_lower_equal(const std::string& a, const std::string& b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("tribase");
    program.add_argument("--benchmarks_path").help("benchmarks path").default_value(std::string("/home/xuqian/Triangle/benchmarks"));
    program.add_argument("--dataset").help("dataset name").default_value(std::string("sift10k"));
    program.add_argument("--input_format").help("format of the dataset").default_value(std::string("fvecs"));
    program.add_argument("--output_format").help("format of the output").default_value(std::string("bin"));
    program.add_argument("--k").help("number of nearest neighbors").default_value(100ul).action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--nprobes").default_value(std::vector<size_t>({1ul})).nargs(0, 100).help("number of clusters to search").scan<'u', size_t>();
    program.add_argument("--opt_levels").default_value(std::vector<std::string>({"OPT_NONE", "OPT_TRIANGLE", "OPT_TRI_SUBNN_L2", "OPT_TRI_SUBNN_IP", "OPT_ALL"})).nargs(0, 10).help("optimization levels");
    program.add_argument("--train_only").default_value(false).implicit_value(true).help("train only");
    program.add_argument("--cache").default_value(false).implicit_value(true).help("use cached index");
    program.add_argument("--high_precision_subNN_index").default_value(false).implicit_value(true).help("use high precision subNN index");
    program.add_argument("--metric").default_value("l2").help("metric type");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::vector<size_t> nprobes = program.get<std::vector<size_t>>("nprobes");
    std::vector<std::string> opt_levels_str = program.get<std::vector<std::string>>("opt_levels");
    size_t k = program.get<size_t>("k");

    std::vector<OptLevel> opt_levels;
    for (const auto& opt_level_str : opt_levels_str) {
        opt_levels.push_back(str2OptLevel(opt_level_str));
    }
    OptLevel added_opt_levels = OptLevel::OPT_NONE;
    for (const OptLevel& opt_level : opt_levels) {
        added_opt_levels = static_cast<OptLevel>(static_cast<int>(added_opt_levels) | static_cast<int>(opt_level));
    }
    std::cout << std::format("Added optimization levels: {}", static_cast<int>(added_opt_levels)) << std::endl;

    std::string benchmarks_path = program.get<std::string>("benchmarks_path");
    std::string dataset = program.get<std::string>("dataset");
    std::string input_format = program.get<std::string>("input_format");
    std::string output_format = program.get<std::string>("output_format");
    std::string metric_str = program.get<std::string>("metric");
    MetricType metric;

    if (str_lower_equal(metric_str, "l2")) {
        metric = MetricType::METRIC_L2;
    } else if (str_lower_equal(metric_str, "ip")) {
        metric = MetricType::METRIC_IP;
    } else {
        throw std::runtime_error("Invalid metric type");
    }

    bool train_only = program.get<bool>("train_only");
    bool cache = program.get<bool>("cache");
    bool high_precision_subNN_index = program.get<bool>("high_precision_subNN_index");

    std::string base_path = std::format("{}/{}/origin/{}_base.{}", benchmarks_path, dataset, dataset, input_format);
    std::string query_path = std::format("{}/{}/origin/{}_query.{}", benchmarks_path, dataset, dataset, input_format);
    std::string groundtruth_path = std::format("{}/{}/result/groundtruth_{}.{}", benchmarks_path, dataset, k, output_format);

    size_t nb, d;
    std::unique_ptr<float[]> base = nullptr;
    std::tie(nb, d) = loadFvecsInfo(base_path);
    size_t nlist = static_cast<size_t>(std::sqrt(nb));

    std::string index_path = std::format("{}/{}/index/index_nlist_{}_opt_{}_{}.index", benchmarks_path, dataset, nlist, static_cast<int>(added_opt_levels), high_precision_subNN_index ? "high_precision" : "low_precision");
    Index index;

    if (std::filesystem::exists(index_path) && cache) {
        std::cout << std::format("Loading index from {}", index_path) << std::endl;
        index.load_index(index_path);
        std::cout << std::format("Index loaded") << std::endl;
    } else {
        std::tie(base, nb, d) = loadFvecs(base_path);
        nlist = static_cast<size_t>(std::sqrt(nb));
        if (high_precision_subNN_index) {
            index = Index(d, nlist, 0, metric, added_opt_levels, 15, 1, 1, true);
        } else {
            index = Index(d, nlist, 0, metric, added_opt_levels, 15, 20, 5, true);
        }
        index.train(nb, base.get());
        index.add(nb, base.get());
        std::cout << std::format("Index trained") << std::endl;
        index.save_index(index_path);
        std::cout << std::format("Index saved to {}", index_path) << std::endl;
    }

    if (train_only) {
        return 0;
    }

    auto [query, nq, _] = loadFvecs(query_path);

    std::unique_ptr<idx_t[]> ground_truth_I = std::make_unique<idx_t[]>(k * nq);
    std::unique_ptr<float[]> ground_truth_D = std::make_unique<float[]>(k * nq);

    if (!std::filesystem::exists(groundtruth_path)) {
        std::cout << std::format("Groundtruth file {} does not exist", groundtruth_path) << std::endl;
        if (base == nullptr) {
            std::tie(base, nb, d) = loadFvecs(base_path);
        }
        faiss::IndexFlatL2 quantizer(d);
        faiss::IndexIVFFlat index2(&quantizer, d, nlist);
        index2.nprobe = nlist;
        std::cout << std::format("Training groundtruth index") << std::endl;
        index2.train(nb, base.get());
        std::cout << std::format("Adding vectors to groundtruth index") << std::endl;
        index2.add(nb, base.get());
        std::cout << std::format("Searching groundtruth index") << std::endl;
        index2.search(nq, query.get(), k, ground_truth_D.get(), ground_truth_I.get());
        writeResultsToFile(ground_truth_I.get(), ground_truth_D.get(), nq, k, groundtruth_path);
        // throw std::runtime_error(std::format("Groundtruth file {} does not exist", groundtruth_path));
        std::cout << std::format("Groundtruth file {} created", groundtruth_path) << std::endl;
    } else {
        loadResults(groundtruth_path, ground_truth_I.get(), ground_truth_D.get(), nq, k);
    }

    if (nprobes.back() == 0) {
        nprobes.back() = nlist;
    }

    for (size_t nprobe : nprobes) {
        index.nprobe = nprobe;
        for (const OptLevel& opt_level : opt_levels) {
            index.opt_level = opt_level;
            std::string output_path = std::format("{}/{}/result/result_nlist_{}_nprobe_{}_opt_{}_k_{}.{}",
                                                  benchmarks_path, dataset, nlist, nprobe, static_cast<int>(opt_level), k, output_format);
            std::unique_ptr<float[]> distances = std::make_unique<float[]>(nq * k);
            std::unique_ptr<idx_t[]> labels = std::make_unique<idx_t[]>(nq * k);
            Stopwatch stopwatch;
            Stats ststs = index.search(nq, query.get(), k, distances.get(), labels.get());
            std::cout << std::format("Search ime: {} s", stopwatch.elapsedSeconds()) << std::endl;
            ststs.print();
            writeResultsToFile(labels.get(), distances.get(), nq, k, output_path);
            float recall = calculate_recall(labels.get(), distances.get(), ground_truth_I.get(), ground_truth_D.get(), nq, k, metric);
            std::cout << std::format("Recall: {}", recall) << std::endl;
        }
    }
}