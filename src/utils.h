#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#include "common.h"

namespace tribase {

inline std::pair<size_t, int> loadFvecsInfo(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    // 读取向量的维度
    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));

    // 移动到文件末尾获取文件大小，计算向量数量
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    size_t n = fileSize / (4 + d * 4);  // 减去初始的维度信息

    return {n, d};
}

inline std::tuple<std::unique_ptr<float[]>, size_t, int> loadFvecs(const std::string& filePath, std::pair<int, int> bounds = {1, 0}) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));

    int vecSizeof = 4 + d * 4;  // int + d * float

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    size_t bmax = fileSize / vecSizeof;

    size_t a = bounds.first;
    size_t b = (bounds.second == 0) ? bmax : bounds.second;

    assert(a >= 1 && b <= bmax && b >= a);

    size_t n = b - a + 1;
    std::unique_ptr<float[]> vectors = std::make_unique<float[]>(n * d);

    file.seekg((a - 1) * vecSizeof, std::ios::beg);

    for (size_t i = 0; i < n; ++i) {
        file.seekg(4, std::ios::cur);
        file.read(reinterpret_cast<char*>(vectors.get() + i * d), d * sizeof(float));
    }

    return std::make_tuple(std::move(vectors), n, d);
}

inline std::pair<size_t, int> loadBvecsInfo(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    size_t n = fileSize / (4 + d);  // 减去初始的维度信息

    return {n, d};
}

inline std::tuple<std::unique_ptr<uint8_t[]>, size_t, int> loadBvecs(const std::string& filePath, std::pair<int, int> bounds = {1, 0}) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));
    int vecSizeof = 4 + d;  // int + d * uint8_t

    size_t bmax = (fileSize - 4) / vecSizeof;

    size_t a = bounds.first;
    size_t b = (bounds.second == 0) ? bmax : bounds.second;

    assert(a >= 1 && b <= bmax && b >= a);

    size_t n = b - a + 1;
    std::unique_ptr<uint8_t[]> vectors = std::make_unique<uint8_t[]>(n * d);

    file.seekg(4 + (a - 1) * vecSizeof, std::ios::beg);

    for (size_t i = 0; i < n; ++i) {
        file.seekg(4, std::ios::cur);
        file.read(reinterpret_cast<char*>(vectors.get() + i * d), d * sizeof(uint8_t));
    }

    return std::make_tuple(std::move(vectors), n, d);
}

// A class for measuring execution time
class Stopwatch {
   public:
    // Constructor initializes the start time
    Stopwatch()
        : start(std::chrono::high_resolution_clock::now()) {}

    // Resets the start time to the current time
    inline void reset() { start = std::chrono::high_resolution_clock::now(); }

    // Returns the elapsed time in milliseconds since the stopwatch was started or last reset
    inline double elapsedMilliseconds(bool isReset = false) {
        auto end = std::chrono::high_resolution_clock::now();
        auto ret = std::chrono::duration<double, std::milli>(end - start).count();
        if (isReset) {
            reset();
        }
        return ret;
    }

    inline double elapsedSeconds(bool isReset = false) {
        auto end = std::chrono::high_resolution_clock::now();
        auto ret = std::chrono::duration<double>(end - start).count();
        if (isReset) {
            reset();
        }
        return ret;
    }

   private:
    // The start time
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

// inline float calculatedEuclideanDistance(const float* vec1, const float* vec2, size_t size) {
//     float distance = 0.0;
//     for (size_t i = 0; i < size; ++i) {
//         float diff = vec1[i] - vec2[i];
//         distance += diff * diff;
//     }
//     return distance;
// }

inline float calculatedEuclideanDistance(const float* vec1, const float* vec2, size_t size) {
    Eigen::Map<const Eigen::VectorXf> v1(vec1, size);
    Eigen::Map<const Eigen::VectorXf> v2(vec2, size);
    return (v1 - v2).squaredNorm();
}

// float calculatedEuclideanDistance(const float* vec1, const float* vec2, size_t size);

// Calculates the inner product between two vectors
inline float calculatedInnerProduct(const float* vec1, const float* vec2, size_t size) {
    float distance = 0.0;
    // Calculate the squared difference for each dimension
    for (size_t i = 0; i < size; ++i) {
        distance += vec1[i] * vec2[i];
    }
    return distance;
}

// Calculates the magnitude (length) of a vector
inline float vectorMagnitude(const float* vec, size_t size) {
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

// Calculates the cosine similarity between two vectors
inline float calculateCosineSimilarity(const float* vec1, const float* vec2, size_t size) {
    float dotProduct = 0.0;
    for (size_t i = 0; i < size; ++i) {
        dotProduct += vec1[i] * vec2[i];
    }

    float magnitude1 = vectorMagnitude(vec1, size);
    float magnitude2 = vectorMagnitude(vec2, size);

    if (magnitude1 == 0 || magnitude2 == 0) {
        throw std::invalid_argument("One or both vectors are zero vectors.");
    }

    return dotProduct / (magnitude1 * magnitude2);
}

inline void prepareDirectory(const std::string& filePath) {
    std::filesystem::path path(filePath);
    if (!std::filesystem::exists(path.parent_path())) {
        std::filesystem::create_directories(path.parent_path());
    }
}

inline void writeResultsToFile(const idx_t* labels, const float* distances, size_t nq, size_t k, std::string filePath) {
    prepareDirectory(filePath);

    if (filePath.ends_with("txt")) {
        std::ofstream outFile(filePath);
        if (!outFile.is_open()) {
            std::cerr << std::format("Failed to open file: {}", filePath) << std::endl;
            return;
        }
        for (size_t i = 0; i < nq; ++i) {
            for (size_t j = 0; j < k; ++j) {
                outFile << labels[i * k + j] << " " << std::fixed << std::setprecision(6) << distances[i * k + j];
                if (j < k - 1) {
                    outFile << " ";
                }
            }
            outFile << "\n";
        }
        outFile.close();
    } else {
        std::ofstream outFile(filePath, std::ios::binary);
        if (!outFile.is_open()) {
            std::cerr << std::format("Failed to open file: {}", filePath) << std::endl;
            return;
        }
        outFile.write(reinterpret_cast<const char*>(labels), nq * k * sizeof(idx_t));
        outFile.write(reinterpret_cast<const char*>(distances), nq * k * sizeof(float));
        outFile.close();
    }
}

inline void loadResults(const std::string& filePath, idx_t* labels, float* distances, size_t nq, size_t k) {
    if (filePath.ends_with("txt")) {
        std::ifstream inFile(filePath);
        if (!inFile.is_open()) {
            std::cerr << std::format("Failed to open file: {}", filePath) << std::endl;
            return;
        }
        for (size_t i = 0; i < nq; ++i) {
            for (size_t j = 0; j < k; ++j) {
                inFile >> labels[i * k + j] >> distances[i * k + j];
            }
        }
        inFile.close();
        // throw std::invalid_argument("Reading txt files is not supported.");
    } else {
        std::ifstream inFile(filePath, std::ios::binary);
        if (!inFile.is_open()) {
            std::cerr << std::format("Failed to open file: {}", filePath) << std::endl;
            return;
        }
        inFile.read(reinterpret_cast<char*>(labels), nq * k * sizeof(idx_t));
        inFile.read(reinterpret_cast<char*>(distances), nq * k * sizeof(float));
        inFile.close();
    }
}

inline float relative_error(float x, float y) {
    if (x == 0) {
        return std::abs(y);
    } else {
        return std::abs((x - y) / x);
    }
}

inline float calculate_recall(const idx_t* I, const float* D, const idx_t* GT, const float* GD, size_t nq, size_t k, MetricType metric, size_t gt_k = 0) {
    if (gt_k == 0) {
        gt_k = k;
    }
    size_t correct = 0;
    if (k > gt_k) {
        throw std::invalid_argument("k should be less than or equal to gt_k.");
    }
    // for (size_t i = 0; i < nq; ++i) {
    //     std::unordered_set<idx_t> groundtruth(GT + i * gt_k, GT + i * gt_k + k);
    //     for (size_t j = 0; j < k; ++j) {
    //         if (I[i * k + j] == -1) {
    //             break;
    //         }
    //         if (groundtruth.find(I[i * k + j]) != groundtruth.end()) {
    //             correct++;
    //         }
    //     }
    // }
    if (metric == MetricType::METRIC_L2) {
        for (size_t i = 0; i < nq; ++i) {
            float topK = std::numeric_limits<float>::max();
            size_t ii = k - 1;
            while (GD[i * gt_k + ii] == -1) {
                ii--;
            }
            topK = GD[i * gt_k + ii];
            for (size_t j = 0; j < k; ++j) {
                if (I[i * k + j] == -1) {
                    break;
                }
                if (D[i * k + j] <= topK || relative_error(D[i * k + j], topK) < 1e-5) {
                    correct++;
                } else {
                    std::cerr << std::format("D[{}, {}]= {} > topK= {}", i, j, D[i * k + j], topK) << std::endl;
                    assert(false);
                }
            }
        }
    } else {
        for (size_t i = 0; i < nq; ++i) {
            float topK = std::numeric_limits<float>::lowest();
            size_t ii = k - 1;
            while (GD[i * gt_k + ii] == -1) {
                ii--;
            }
            topK = GD[i * gt_k + ii];
            for (size_t j = 0; j < k; ++j) {
                if (I[i * k + j] == -1) {
                    break;
                }
                if (D[i * k + j] >= topK || relative_error(D[i * k + j], topK) < 1e-5) {
                    correct++;
                } else {
                    std::cerr << std::format("D[{}, {}]= {} < topK= {}", i, j, D[i * k + j], topK) << std::endl;
                    assert(false);
                }
            }
        }
    }
    return static_cast<float>(correct) / (nq * k);
}

inline void output_codes(const float* code, size_t d) {
    if (code) {
        for (size_t i = 0; i < d; ++i) {
            std::cerr << code[i] << ",";
        }
    }
    std::cerr << std::endl;
}

}  // namespace tribase