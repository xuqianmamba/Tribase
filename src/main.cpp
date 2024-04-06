#include <argparse/argparse.hpp>
#include <format>
#include <iostream>
#include <memory>
#include "tribase.h"
#include "utils.h"
using namespace tribase;
int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("tribase");
    program.add_argument("--base_file").help("base file path").default_value(std::string("../src/tests/iris.fvecs"));
    program.add_argument("--query_file").help("query file path").default_value(std::string("../src/tests/iris.fvecs"));
    program.add_argument("--nlist").help("number of clusters").default_value(3ul).action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--nprobe").help("number of clusters to search").default_value(3ul).action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--k").help("number of nearest neighbors").default_value(3ul).action([](const std::string& value) -> size_t { return std::stoul(value); });

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

    auto [base, nb, d] = loadFvecs(base_file);
    auto [query, nq, _] = loadFvecs(query_file);

    // Index index(d, nlist, nprobe, MetricType::METRIC_L2, OptLevel::OPT_SUBNN_IP);
    Index index(d, nlist, nprobe, MetricType::METRIC_L2, OptLevel::OPT_NONE);
    index.train(nb, base.get());
    index.add(nb, base.get());

    std::unique_ptr<float[]> distances(new float[nq * k]);
    std::unique_ptr<idx_t[]> labels(new idx_t[nq * k]);

    Stopwatch sw;
    index.search(nq, query.get(), k, distances.get(), labels.get());
    std::cout << "search time: " << sw.elapsedSeconds() << "s" << std::endl;

    // for (size_t i = 0; i < nq; i++) {
    //     for (size_t j = 0; j < k; j++) {
    //         std::cout << std::format("({},{})", distances[i * k + j], labels[i * k + j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }
}