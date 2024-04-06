#pragma once

#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace tribase {

inline std::tuple<std::unique_ptr<float[]>, size_t, int> loadFvecs(const std::string& filePath, std::pair<int, int> bounds = {1, 0}) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    // 读取向量的维度
    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));

    // 计算每个向量的大小（包括维度信息）
    int vecSizeof = 4 + d * 4;  // int + d * float

    // 移动到文件末尾获取文件大小，计算向量数量
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    size_t bmax = fileSize / vecSizeof;

    size_t a = bounds.first;
    size_t b = (bounds.second == 0) ? bmax : bounds.second;

    assert(a >= 1 && b <= bmax && b >= a);

    size_t n = b - a + 1;  // 实际读取的向量数量
    std::unique_ptr<float[]> vectors = std::make_unique<float[]>(n * d);

    // 移动到起始位置
    file.seekg((a - 1) * vecSizeof, std::ios::beg);

    // 读取向量
    for (size_t i = 0; i < n; ++i) {
        file.seekg(4, std::ios::cur);  // 跳过向量维度
        file.read(reinterpret_cast<char*>(vectors.get() + i * d), d * sizeof(float));
    }

    return std::make_tuple(std::move(vectors), n, d);
}

inline std::tuple<std::unique_ptr<uint8_t[]>, size_t, int> loadBvecs(const std::string& filePath, std::pair<int, int> bounds = {1, 0}) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    // 移动到文件末尾获取文件大小
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    // 回到文件开头
    file.seekg(0, std::ios::beg);

    // 读取向量的维度
    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));
    // 计算每个向量的大小（包括维度信息）
    int vecSizeof = 4 + d;  // int + d * uint8_t

    // 计算向量数量
    size_t bmax = (fileSize - 4) / vecSizeof;  // 减去初始的维度信息

    size_t a = bounds.first;
    size_t b = (bounds.second == 0) ? bmax : bounds.second;

    assert(a >= 1 && b <= bmax && b >= a);

    size_t n = b - a + 1;  // 实际读取的向量数量
    std::unique_ptr<uint8_t[]> vectors = std::make_unique<uint8_t[]>(n * d);

    // 移动到起始位置
    file.seekg(4 + (a - 1) * vecSizeof, std::ios::beg);  // 跳过初始的维度信息

    // 读取向量
    for (size_t i = 0; i < n; ++i) {
        file.seekg(4, std::ios::cur);  // 跳过每个向量前的维度信息
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
    void reset() { start = std::chrono::high_resolution_clock::now(); }

    // Returns the elapsed time in milliseconds since the stopwatch was started or last reset
    double elapsedMilliseconds(bool isReset = false) {
        auto end = std::chrono::high_resolution_clock::now();
        auto ret = std::chrono::duration<double, std::milli>(end - start).count();
        if (isReset) {
            reset();
        }
        return ret;
    }

    double elapsedSeconds(bool isReset = false) {
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

// Calculates the Euclidean distance between two vectors
inline float calculatedEuclideanDistance(const float* vec1, const float* vec2, size_t size) {
    float distance = 0.0;
    // Calculate the squared difference for each dimension
    for (size_t i = 0; i < size; ++i) {
        float diff = vec1[i] - vec2[i];
        distance += diff * diff;
    }

    return sqrt(distance);
}

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

}  // namespace tribase