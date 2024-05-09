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
#include <faiss/index_io.h>
#include "tribase.h"
#include "utils.h"

using namespace tribase;

bool str_lower_equal(const std::string& a, const std::string& b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("tribase");
    program.add_argument("--benchmarks_path").help("benchmarks path").default_value(std::string("/home/xuqian/Triangle/benchmarks"));
    program.add_argument("--dataset").help("dataset name").default_value(std::string("msong"));
    program.add_argument("--input_format").help("format of the dataset").default_value(std::string("fvecs"));
    program.add_argument("--output_format").help("format of the output").default_value(std::string("bin"));
    program.add_argument("--k").help("number of nearest neighbors").default_value(100ul).action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--nprobes").default_value(std::vector<size_t>({0ul})).nargs(0, 100).help("number of clusters to search").scan<'u', size_t>();
    program.add_argument("--opt_levels").default_value(std::vector<std::string>({"OPT_NONE", "OPT_TRIANGLE", "OPT_TRI_SUBNN_L2", "OPT_TRI_SUBNN_IP", "OPT_ALL"})).nargs(0, 10).help("optimization levels");
    // program.add_argument("--opt_levels").default_value(std::vector<std::string>({"OPT_TRI_SUBNN_L2"})).nargs(0, 10).help("optimization levels");
    program.add_argument("--train_only").default_value(false).implicit_value(true).help("train only");
    program.add_argument("--cache").default_value(false).implicit_value(true).help("use cached index");
    program.add_argument("--sub_nprobe_ratio").default_value(1.0f).action([](const std::string& value) -> float { return std::stof(value); });
    program.add_argument("--metric").default_value("l2").help("metric type");
    program.add_argument("--run_faiss").default_value(false).implicit_value(true).help("run faiss");
    program.add_argument("--loop").default_value(1ul).action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--nlist").default_value(0ul).action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--verbose").default_value(false).implicit_value(true).help("verbose");
    // program.add_argument("--mini").default_value(false).implicit_value(true).help("1/100 datasets");
    program.add_argument("--ratios").default_value(std::vector<float>({1.0f})).nargs(0, 100).help("ratio of the number of subNNs to the number of clusters").scan<'f', float>();
    // program.add_argument("--ratio").default_value(1.0f).action([](const std::string& value) -> float { return std::stof(value); });

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::vector<size_t> nprobes = program.get<std::vector<size_t>>("nprobes");
    std::vector<std::string> opt_levels_str = program.get<std::vector<std::string>>("opt_levels");
    std::vector<float> ratios = program.get<std::vector<float>>("ratios");

    size_t k = program.get<size_t>("k");

    std::vector<OptLevel> opt_levels;
    for (const auto& opt_level_str : opt_levels_str) {
        opt_levels.push_back(str2OptLevel(opt_level_str));
    }
    OptLevel added_opt_levels = OptLevel::OPT_NONE;

    std::string benchmarks_path = program.get<std::string>("benchmarks_path");
    std::string dataset = program.get<std::string>("dataset");
    std::string input_format = program.get<std::string>("input_format");
    std::string output_format = program.get<std::string>("output_format");
    std::string metric_str = program.get<std::string>("metric");
    bool run_faiss = program.get<bool>("run_faiss");
    MetricType metric;
    size_t loop = program.get<size_t>("loop");
    size_t nlist = program.get<size_t>("nlist");
    bool verbose = program.get<bool>("verbose");

    if (str_lower_equal(metric_str, "l2")) {
        metric = MetricType::METRIC_L2;
    } else if (str_lower_equal(metric_str, "ip")) {
        metric = MetricType::METRIC_IP;
    } else {
        throw std::runtime_error("Invalid metric type");
    }

    bool train_only = program.get<bool>("train_only");
    bool cache = program.get<bool>("cache");
    float sub_nprobe_ratio = program.get<float>("sub_nprobe_ratio");

    std::string base_path = std::format("{}/{}/origin/{}_base.{}", benchmarks_path, dataset, dataset, input_format);
    std::string query_path = std::format("{}/{}/origin/{}_query.{}", benchmarks_path, dataset, dataset, input_format);
    std::string groundtruth_path = std::format("{}/{}/result/groundtruth_{}.{}", benchmarks_path, dataset, k, output_format);
    std::string log_path = std::format("{}/{}/result/log.csv", benchmarks_path, dataset);

    size_t nb, d;
    std::unique_ptr<float[]> base = nullptr;
    std::tie(nb, d) = loadFvecsInfo(base_path);
    if (nlist == 0) {
        nlist = static_cast<size_t>(std::sqrt(nb));
    }
    size_t sub_nlist = std::sqrt(nb / nlist);
    size_t sub_nprobe = std::max(static_cast<size_t>(sub_nlist * sub_nprobe_ratio), 1ul);
    if (verbose) {
        std::cout << std::format("sub_nlist: {} sub_nprobe: {}", sub_nlist, sub_nprobe) << std::endl;
    }

    for (const OptLevel& opt_level : opt_levels) {
        added_opt_levels = static_cast<OptLevel>(static_cast<int>(added_opt_levels) | static_cast<int>(opt_level));
    }
    if (verbose) {
        std::cout << std::format("Added optimization levels: {}", static_cast<int>(added_opt_levels)) << std::endl;
    }
    // nprobes.clear();
    // for (size_t val = 1; val <= nlist / 2; val *= 2) {
    //     nprobes.push_back(val);
    // }
    // nprobes.push_back(nlist);

    auto get_index_path = [&]() {
        int target = static_cast<int>(added_opt_levels);
        for (int i = 0; i < 8; i++) {
            // target is a subset of i
            if ((target & i) == target) {
                std::string index_path = std::format("{}/{}/index/index_nlist_{}_opt_{}_subNprobeRatio_{}.index", benchmarks_path, dataset, nlist, i, sub_nprobe_ratio);
                if (std::filesystem::exists(index_path)) {
                    return index_path;
                }
            }
        }
        return std::format("{}/{}/index/index_nlist_{}_opt_{}_subNprobeRatio_{}.index", benchmarks_path, dataset, nlist, target, sub_nprobe_ratio);
    };

    auto get_faiss_index_path = [&]() {
        return std::format("{}/{}/index/faiss_index_nlist_{}.index", benchmarks_path, dataset, nlist);
    };

    std::string index_path = get_index_path();
    std::string faiss_index_path = get_faiss_index_path();
    prepareDirectory(faiss_index_path);

    Index index;

    if (std::filesystem::exists(index_path) && cache) {
        if (verbose) {
            std::cout << std::format("Loading index from {}", index_path) << std::endl;
        }
        index.load_index(index_path);
        if (verbose) {
            std::cout << std::format("Index loaded") << std::endl;
        }
    } else {
        std::tie(base, nb, d) = loadFvecs(base_path);
        nlist = static_cast<size_t>(std::sqrt(nb));
        index = Index(d, nlist, 0, metric, added_opt_levels, 15, sub_nlist, sub_nprobe, verbose);
        index.train(nb, base.get());
        index.add(nb, base.get());
        if (verbose) {
            std::cout << std::format("Index trained") << std::endl;
        }
        index.save_index(index_path);
        if (verbose) {
            std::cout << std::format("Index saved to {}", index_path) << std::endl;
        }
    }

    if (train_only) {
        return 0;
    }

    auto [query, nq, _] = loadFvecs(query_path);

    std::unique_ptr<idx_t[]> ground_truth_I = std::make_unique<idx_t[]>(k * nq);
    std::unique_ptr<float[]> ground_truth_D = std::make_unique<float[]>(k * nq);

    std::string faiss_time_path = std::format("{}/{}/result/faiss_result_nlist_{}.txt", benchmarks_path, dataset, nlist);
    std::vector<double> faiss_time(nprobes.size(), 0.0);
    std::ifstream faiss_time_input(faiss_time_path);
    if (faiss_time_input.is_open()) {
        size_t nprobe;
        double time;
        float recall, r2;
        while (faiss_time_input >> nprobe >> time >> recall >> r2) {
            auto it = std::find(nprobes.begin(), nprobes.end(), nprobe);
            if (it != nprobes.end()) {
                faiss_time[std::distance(nprobes.begin(), it)] = time;
            }
        }
    }

    faiss::IndexFlatL2 quantizer(d);
    std::unique_ptr<faiss::IndexIVFFlat> index_faiss = std::make_unique<faiss::IndexIVFFlat>(&quantizer, d, nlist);

    if (!std::filesystem::exists(groundtruth_path)) {
        double faiss_groundtruth_time = 0.0;
        if (verbose) {
            std::cout << std::format("Groundtruth file {} does not exist", groundtruth_path) << std::endl;
        }

        if (!std::filesystem::exists(faiss_index_path)) {
            if (verbose) {
                std::cout << std::format("Training Faiss index") << std::endl;
            }
            if (base == nullptr) {
                std::tie(base, nb, d) = loadFvecs(base_path);
            }
            index_faiss->train(nb, base.get());
            if (verbose) {
                std::cout << std::format("Adding vectors to Faiss index") << std::endl;
            }
            index_faiss->add(nb, base.get());
            ::faiss::write_index(index_faiss.get(), faiss_index_path.c_str());
        } else {
            index_faiss.reset(dynamic_cast<faiss::IndexIVFFlat*>(::faiss::read_index(faiss_index_path.c_str())));
        }

        if (verbose) {
            std::cout << std::format("Searching Faiss index") << std::endl;
        }
        index_faiss->nprobe = nlist;
        Stopwatch stopwatch;
        index_faiss->search(nq, query.get(), k, ground_truth_D.get(), ground_truth_I.get());
        faiss_groundtruth_time = stopwatch.elapsedSeconds();
        writeResultsToFile(ground_truth_I.get(), ground_truth_D.get(), nq, k, groundtruth_path);
        if (verbose) {
            std::cout << std::format("Groundtruth file {} created using {} s", groundtruth_path, faiss_groundtruth_time) << std::endl;
        }
        if (nprobes.back() == nlist) {
            faiss_time.back() = faiss_groundtruth_time;
        }
    } else {
        if (verbose) {
            std::cout << std::format("Loading groundtruth file {}", groundtruth_path) << std::endl;
        }
        loadResults(groundtruth_path, ground_truth_I.get(), ground_truth_D.get(), nq, k);
        if (verbose) {
            std::cout << std::format("Groundtruth file loaded") << std::endl;
        }
    }

    if (nprobes.back() == 0) {
        nprobes.back() = nlist;
    }

    if (run_faiss) {
        if (verbose) {
            std::cout << std::format("Running Faiss") << std::endl;
        }
        if (!index_faiss->is_trained) {
            if (!std::filesystem::exists(faiss_index_path)) {
                if (verbose) {
                    std::cout << std::format("Training Faiss index") << std::endl;
                }
                if (base == nullptr) {
                    std::tie(base, nb, d) = loadFvecs(base_path);
                }
                index_faiss->train(nb, base.get());
                if (verbose) {
                    std::cout << std::format("Adding vectors to Faiss index") << std::endl;
                }
                index_faiss->add(nb, base.get());
                ::faiss::write_index(index_faiss.get(), faiss_index_path.c_str());
            } else {
                index_faiss.reset(dynamic_cast<faiss::IndexIVFFlat*>(::faiss::read_index(faiss_index_path.c_str())));
            }
        }
        std::ofstream faiss_time_output(faiss_time_path);
        std::unique_ptr<float[]> tmp_faiss_dis = std::make_unique<float[]>(k * nq);
        std::unique_ptr<idx_t[]> tmp_faiss_labels = std::make_unique<idx_t[]>(k * nq);
        for (size_t i = 0; i < nprobes.size(); i++) {
            index_faiss->nprobe = nprobes[i];
            if (loop > 1) {
                index_faiss->search(nq, query.get(), k, tmp_faiss_dis.get(), tmp_faiss_labels.get());
            }
            Stopwatch stopwatch;
            for (size_t j = 0; j < loop; j++) {
                index_faiss->search(nq, query.get(), k, tmp_faiss_dis.get(), tmp_faiss_labels.get());
            }
            float recall = calculate_recall(tmp_faiss_labels.get(), tmp_faiss_dis.get(), ground_truth_I.get(), ground_truth_D.get(), nq, k, metric);
            float r2 = calculate_r2(tmp_faiss_labels.get(), tmp_faiss_dis.get(), ground_truth_I.get(), ground_truth_D.get(), nq, k, metric);
            faiss_time[i] = stopwatch.elapsedSeconds() / loop;
            std::cout << std::format("Faiss nprobe: {} time: {} recall: {} r2: {}", nprobes[i], faiss_time[i], recall, r2) << std::endl;
            faiss_time_output << std::format("{} {} {} {}\n", nprobes[i], faiss_time[i], recall, r2);
        }
        return 0;
    }

    for (size_t i = 0; i < nprobes.size(); i++) {
        size_t nprobe = nprobes[i];
        double f_time = faiss_time[i];
        index.nprobe = nprobe;
        for (const OptLevel& opt_level : opt_levels) {
            index.opt_level = opt_level;
            for (float ratio : ratios) {
                std::string output_path = std::format("{}/{}/result/result_nlist_{}_nprobe_{}_opt_{}_k_{}_ratio_{}.{}",
                                                      benchmarks_path, dataset, nlist, nprobe, static_cast<int>(opt_level), k, ratio, output_format);
                std::unique_ptr<float[]> distances = std::make_unique<float[]>(nq * k);
                std::unique_ptr<idx_t[]> labels = std::make_unique<idx_t[]>(nq * k);
                if (loop > 1) {
                    index.search(nq, query.get(), k, distances.get(), labels.get(), ratio);
                }
                Stopwatch stopwatch;
                Stats stats;
                for (size_t j = 0; j < loop; j++) {
                    stats = index.search(nq, query.get(), k, distances.get(), labels.get(), ratio);
                }
                float recall = calculate_recall(labels.get(), distances.get(), ground_truth_I.get(), ground_truth_D.get(), nq, k, metric);
                float r2 = calculate_r2(labels.get(), distances.get(), ground_truth_I.get(), ground_truth_D.get(), nq, k, metric);
                double search_time = stopwatch.elapsedSeconds() / loop;
                stats.simi_ratio = ratio;
                stats.nprobe = nprobe;
                stats.query_time = search_time;
                stats.faiss_query_time = f_time;
                stats.opt_level = opt_level;
                stats.recall = recall;
                stats.r2 = r2;
                stats.print();
                stats.toCsv(log_path, true);
                // writeResultsToFile(labels.get(), distances.get(), nq, k, output_path);
                // std::ofstream output("truth.bin", std::ios::binary);
                // output.write(reinterpret_cast<const char*>(&nq), 4);
                // output.write(reinterpret_cast<const char*>(&k), 4);
                // for (size_t i = 0; i < nq; i++) {
                //     for (size_t j = 0; j < k; j++) {
                //         output.write(reinterpret_cast<const char*>(&ground_truth_I[i * k + j]), 4);
                //     }
                // }
            }
        }
    }
}