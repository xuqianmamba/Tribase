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

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("tribase");
    program.add_argument("--benchmarks_path").help("benchmarks path").default_value(std::string("/home/xuqian/Triangle/benchmarks"));
    program.add_argument("--dataset").help("dataset name").default_value(std::string("sift10k"));
    program.add_argument("--format").help("format of the dataset").default_value(std::string("fvecs"));
    program.add_argument("--k").help("number of nearest neighbors").default_value(100ul).action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--nprobes").default_value(std::vector<size_t>({1ul})).nargs(0, 100).help("number of clusters to search").scan<'u', size_t>();
    program.add_argument("--opt_levels").default_value(std::vector<std::string>({"OPT_NONE", "OPT_TRIANGLE", "OPT_SUBNN_L2", "OPT_SUBNN_IP", "OPT_ALL"})).nargs(0, 10).help("optimization levels");
    program.add_argument("--train_only").default_value(false).implicit_value(true).help("train only");
    program.add_argument("--cache").default_value(false).implicit_value(true).help("use cached index");
    program.add_argument("--high_precision_subNN_index").default_value(false).implicit_value(true).help("use high precision subNN index");

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

    std::string benchmarks_path = program.get<std::string>("benchmarks_path");
    std::string dataset = program.get<std::string>("dataset");
    std::string format = program.get<std::string>("format");
    bool train_only = program.get<bool>("train_only");
    bool cache = program.get<bool>("cache");
    bool high_precision_subNN_index = program.get<bool>("high_precision_subNN_index");

    std::string base_path = std::format("{}/{}/origin/{}_base.{}", benchmarks_path, dataset, dataset, format);
    std::string query_path = std::format("{}/{}/origin/{}_query.{}", benchmarks_path, dataset, dataset, format);
    std::string groundtruth_path = std::format("{}/{}/result/groundtruth_{}.txt", benchmarks_path, dataset, k);

    size_t nb, d;
    std::unique_ptr<float[]> base = nullptr;
    std::tie(nb, d) = loadFvecsInfo(base_path);
    size_t nlist = static_cast<size_t>(std::sqrt(nb));

    std::string index_path = std::format("{}/{}/index/index_nlist_{}_opt_{}_{}.index", benchmarks_path, dataset, nlist, static_cast<int>(added_opt_levels), high_precision_subNN_index ? "high_precision" : "low_precision");
    Index index;

    if (std::filesystem::exists(index_path) && cache) {
        std::cout << std::format("Loading index from {}", index_path) << std::endl;
        index.load_index(index_path);
    } else {
        std::tie(base, nb, d) = loadFvecs(base_path);
        nlist = static_cast<size_t>(std::sqrt(nb));
        if (high_precision_subNN_index) {
            index = Index(d, nlist, 0, MetricType::METRIC_L2, added_opt_levels, 50, 1, 1, true);
        } else {
            index = Index(d, nlist, 0, MetricType::METRIC_L2, added_opt_levels, 50, 20, 5, true);
        }
        index.train(nb, base.get());
        index.add(nb, base.get());
        index.save_index(index_path);
        std::cout << std::format("Index saved to {}", index_path) << std::endl;
    }

    if (train_only) {
        return 0;
    }

    auto [query, nq, _] = loadFvecs(query_path);

    if (!std::filesystem::exists(groundtruth_path)) {
        if (base == nullptr) {
            std::tie(base, nb, d) = loadFvecs(base_path);
        }
        faiss::IndexFlatL2 quantizer(d);
        faiss::IndexIVFFlat index2(&quantizer, d, nlist);
        index2.nprobe = nlist;
        index2.train(nb, base.get());
        index2.add(nb, base.get());
        std::unique_ptr<idx_t[]> I(new idx_t[k * nq]);
        std::unique_ptr<float[]> D(new float[k * nq]);
        index2.search(nq, query.get(), k, D.get(), I.get());
        writeResultsToFile(I.get(), D.get(), nq, k, groundtruth_path);
        // throw std::runtime_error(std::format("Groundtruth file {} does not exist", groundtruth_path));
    }

    if (nprobes.back() == 0) {
        nprobes.back() = nlist;
    }

    for (size_t nprobe : nprobes) {
        index.nprobe = nprobe;
        for (const OptLevel& opt_level : opt_levels) {
            index.opt_level = opt_level;
            std::string output_path = std::format("{}/{}/result/result_nlist_{}_nprobe_{}_opt_{}_k_{}.txt", benchmarks_path, dataset, nlist, nprobe, static_cast<int>(opt_level), k);
            std::unique_ptr<float[]> distances = std::make_unique<float[]>(nq * k);
            std::unique_ptr<idx_t[]> labels = std::make_unique<idx_t[]>(nq * k);
            Stats ststs = index.search(nq, query.get(), k, distances.get(), labels.get());
            // ststs.print();
            writeResultsToFile(labels.get(), distances.get(), nq, k, output_path);
        }
    }
}