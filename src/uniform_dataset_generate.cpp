#include <argparse/argparse.hpp>
#include <omp.h>
#include <chrono>
#include <format>
#include <memory>
#include <random>
#include <fstream>
#include <filesystem>
#include <format>

void dump_fvecs(const std::string& filename, const float* data, size_t n, size_t d) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    for (size_t i = 0; i < n; ++i) {
        ofs.write(reinterpret_cast<const char*>(&d), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&data[i * d]), d * sizeof(float));
    }
}

int main(int argc, char* argv[]){
    argparse::ArgumentParser program("uniform_dataset_generate");
    program.add_argument("--benchmarks_path").help("benchmarks path").default_value(std::string("./benchmarks"));
    program.add_argument("--dataset").help("dataset name").default_value(std::string("8d256s"));
    program.add_argument("--d").default_value(8ul).action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--nb").default_value(256ul).action([](const std::string& value) -> size_t { return std::stoul(value); });
    program.add_argument("--nq").default_value(128ul).action([](const std::string& value) -> size_t { return std::stoul(value); });

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::string benchmarks_path = program.get<std::string>("benchmarks_path");
    std::string dataset = program.get<std::string>("dataset");
    size_t d = program.get<size_t>("--d");
    size_t nb = program.get<size_t>("--nb");
    size_t nq = program.get<size_t>("--nq");
    std::filesystem::path dataset_path = std::format("{}/{}/origin", benchmarks_path, dataset);

    std::filesystem::create_directories(dataset_path);

    std::unique_ptr<float[]> codes = std::make_unique<float[]>(nb * d);
    std::unique_ptr<float[]> queries = std::make_unique<float[]>(nq * d);

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

    dump_fvecs(std::format("{}/{}/origin/{}_base.fvecs", benchmarks_path, dataset, dataset), codes.get(), nb, d);
    dump_fvecs(std::format("{}/{}/origin/{}_query.fvecs", benchmarks_path, dataset, dataset), queries.get(), nq, d);
}