#include <argparse/argparse.hpp>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include "tribase.h"
#include "utils.h"
using namespace tribase;
int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("tribase");
    program.add_argument("--base_file").help("base file path").default_value(std::string("../benchmarks/sift10k/origin/sift10k_base.fvecs"));
    program.add_argument("--query_file").help("query file path").default_value(std::string("../benchmarks/sift10k/origin/sift10k_query.fvecs"));
    program.add_argument("--nlist").help("number of clusters").default_value(0ul).action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--nprobe").help("number of clusters to search").default_value(1ul).action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--k").help("number of nearest neighbors").default_value(100ul).action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--opt_level").help("optimization level").default_value(OptLevel::OPT_ALL).action([](const std::string& value) -> OptLevel { return str2OptLevel(value); });
    program.add_argument("--output_file").help("output file path").default_value(std::string("../benchmarks/sift10k/result/result.txt"));

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    [[maybe_unused]] std::string base_file = program.get<std::string>("base_file");
    [[maybe_unused]] std::string query_file = program.get<std::string>("query_file");
    [[maybe_unused]] size_t nlist = program.get<size_t>("nlist");
    [[maybe_unused]] size_t nprobe = program.get<size_t>("nprobe");
    [[maybe_unused]] size_t k = program.get<size_t>("k");
    [[maybe_unused]] OptLevel opt_level = program.get<OptLevel>("opt_level");
    [[maybe_unused]] std::string output_file = program.get<std::string>("output_file");

    auto [base, nb, d] = loadFvecs(base_file);
    auto [query, nq, _] = loadFvecs(query_file);

    if (nlist == 0) {
        nlist = static_cast<size_t>(std::sqrt(nb));
        nprobe = (nlist + 9) / 10;
    }

    // Index index(d, nlist, nprobe, MetricType::METRIC_L2, OptLevel::OPT_SUBNN_IP);
    Index index(d, nlist, nprobe, MetricType::METRIC_L2, opt_level);
    std::unique_ptr<float[]> distances(new float[nq * k]);
    std::unique_ptr<idx_t[]> labels(new idx_t[nq * k]);

    Stopwatch sw;
    index.train(nb, base.get());
    std::cout << "train time: " << sw.elapsedSeconds(true) << "s" << std::endl;
    index.add(nb, base.get());
    std::cout << "add time: " << sw.elapsedSeconds(true) << "s" << std::endl;
    index.search(nq, query.get(), k, distances.get(), labels.get());
    std::cout << "search time: " << sw.elapsedSeconds(true) << "s" << std::endl;
    writeResultsToFile(labels.get(), distances.get(), nq, k, output_file);
    std::cout << "write time: " << sw.elapsedSeconds(true) << "s" << std::endl;

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat index2(&quantizer, d, nlist);
    index2.nprobe = nprobe;
    std::unique_ptr<idx_t[]> I(new idx_t[k * nq]);
    std::unique_ptr<float[]> D(new float[k * nq]);
    Stopwatch sw2;
    index2.train(nb, base.get());
    std::cout << "train time: " << sw2.elapsedSeconds(true) << "s" << std::endl;
    index2.add(nb, base.get());
    std::cout << "add time: " << sw2.elapsedSeconds(true) << "s" << std::endl;
    index2.search(nq, query.get(), k, D.get(), I.get());
    std::cout << "search time: " << sw2.elapsedSeconds(true) << "s" << std::endl;
    writeResultsToFile(I.get(), D.get(), nq, k, output_file + ".faiss");
}