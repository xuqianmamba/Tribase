#pragma once

#include <mkl.h>
// #include <mkl_cblas.h>
#include <immintrin.h>  // 包含AVX2和其他SIMD指令集的头文件
#include <Eigen/Dense>
#include <algorithm>  // 包含std::fill_n
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
// #include "avx512.h"
#include "common.h"
#include "faiss/faiss/utils/distances.h"
#include "platform_macros.h"

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

// V0
//  inline float calculatedEuclideanDistance(const float* vec1, const float* vec2, size_t size) {
//      // 计算 vec1 和 vec2 的 L2 范数的平方
//      float norm1 = cblas_snrm2(size, vec1, 1);
//      float norm2 = cblas_snrm2(size, vec2, 1);
//      float norm1Sq = norm1 * norm1;
//      float norm2Sq = norm2 * norm2;

//     // 计算 vec1 和 vec2 的点积
//     float dotProduct = cblas_sdot(size, vec1, 1, vec2, 1);

//     // 使用前面的公式计算欧氏距离的平方
//     float distanceSq = norm1Sq + norm2Sq - 2 * dotProduct;

//     return distanceSq;
// }

// V1
// inline float calculatedEuclideanDistance(const float* vec1, const float* vec2, size_t size) {
//     float diff[size];
//     cblas_scopy(size, vec1, 1, diff, 1);
//     cblas_saxpy(size, -1.0, vec2, 1, diff, 1);
//     float distanceSq = cblas_sdot(size, diff, 1, diff, 1);

//     return distanceSq;
// }

// V2
inline float calculatedEuclideanDistance(const float* vec1, const float* vec2, size_t size) {
    float dotProduct12 = cblas_sdot(size, vec1, 1, vec2, 1);
    float dotProduct11 = cblas_sdot(size, vec1, 1, vec1, 1);
    float dotProduct22 = cblas_sdot(size, vec2, 1, vec2, 1);
    return dotProduct11 + dotProduct22 - 2 * dotProduct12;
}

inline float calculatedEuclideanDistance(const float* vec1, const float* vec2, float norm1, size_t size) {
    float dotProduct22 = cblas_sdot(size, vec2, 1, vec2, 1);
    return norm1 + dotProduct22 - 2 * cblas_sdot(size, vec1, 1, vec2, 1);
}

inline float calculatedEuclideanDistance(const float* vec1, const float* vec2, float norm1, float norm2, size_t size) {
    return norm1 + norm2 - 2 * cblas_sdot(size, vec1, 1, vec2, 1);
}

// V3
// inline float calculatedEuclideanDistance(const float* vec1, const float* vec2, size_t size) {
//     Eigen::Map<const Eigen::VectorXf> v1(vec1, size);
//     Eigen::Map<const Eigen::VectorXf> v2(vec2, size);
//     return (v1 - v2).squaredNorm();
// }

// V4
//  inline float calculatedEuclideanDistance(const float* vec1, const float* vec2, size_t size) {
//      float distance = 0.0;
//      for (size_t i = 0; i < size; ++i) {
//          float diff = vec1[i] - vec2[i];
//          distance += diff * diff;
//      }
//      return distance;
//  }

// V5

// inline float calculatedEuclideanDistance(const float* x, const float* y, size_t d) {
//     size_t newSize = (d + 15) / 16 * 16; // 计算新大小为16的倍数
//     float* newX = new float[newSize];
//     float* newY = new float[newSize];
//     std::copy(x, x + d, newX); // 复制原数组
//     std::copy(y, y + d, newY); // 复制原数组
//     std::fill_n(newX + d, newSize - d, 0.0f); // 用0填充剩余的部分
//     std::fill_n(newY + d, newSize - d, 0.0f); // 用0填充剩余的部分

//     __m512 sum = _mm512_setzero_ps(); // 初始化累加器为0
//     for (size_t i = 0; i < newSize; i += 16) {
//         __m512 mx = _mm512_loadu_ps(newX + i); // 加载x的一部分
//         __m512 my = _mm512_loadu_ps(newY + i); // 加载y的一部分
//         __m512 diff = _mm512_sub_ps(mx, my); // 计算差异
//         __m512 sqr = _mm512_mul_ps(diff, diff); // 计算平方
//         sum = _mm512_add_ps(sum, sqr); // 累加平方
//     }

//     // 将累加器中的值合并
//     __m256 low256 = _mm512_castps512_ps256(sum);
//     __m256 high256 = _mm512_extractf32x8_ps(sum, 1);
//     __m256 sum256 = _mm256_add_ps(low256, high256);
//     __m128 low128 = _mm256_castps256_ps128(sum256);
//     __m128 high128 = _mm256_extractf128_ps(sum256, 1);
//     __m128 sum128 = _mm_add_ps(low128, high128);
//     sum128 = _mm_hadd_ps(sum128, sum128);
//     sum128 = _mm_hadd_ps(sum128, sum128);
//     float finalSum = _mm_cvtss_f32(sum128); // 将累加器中的值转换为float

//     delete[] newX; // 释放临时数组
//     delete[] newY; // 释放临时数组

//     return finalSum; // 返回最终的L2距离的平方
// }

// V6

// inline float calculatedEuclideanDistance(const float* x, const float* y, size_t d) {
//     size_t newSize = (d + 7) / 8 * 8; // 计算新大小为8的倍数
//     float* newX = new float[newSize];
//     float* newY = new float[newSize];
//     std::copy(x, x + d, newX); // 复制原数组
//     std::copy(y, y + d, newY); // 复制原数组
//     std::fill_n(newX + d, newSize - d, 0.0f); // 用0填充剩余的部分
//     std::fill_n(newY + d, newSize - d, 0.0f); // 用0填充剩余的部分

//     __m256 sum = _mm256_setzero_ps(); // 初始化累加器为0
//     for (size_t i = 0; i < newSize; i += 8) {
//         __m256 mx = _mm256_loadu_ps(newX + i); // 加载x的一部分
//         __m256 my = _mm256_loadu_ps(newY + i); // 加载y的一部分
//         __m256 diff = _mm256_sub_ps(mx, my); // 计算差异
//         __m256 sqr = _mm256_mul_ps(diff, diff); // 计算平方
//         sum = _mm256_add_ps(sum, sqr); // 累加平方
//     }

//     // 将累加器中的值合并
//     __m128 low128 = _mm256_castps256_ps128(sum);
//     __m128 high128 = _mm256_extractf128_ps(sum, 1);
//     __m128 sum128 = _mm_add_ps(low128, high128);
//     sum128 = _mm_hadd_ps(sum128, sum128);
//     sum128 = _mm_hadd_ps(sum128, sum128);
//     float finalSum = _mm_cvtss_f32(sum128); // 将累加器中的值转换为float

//     delete[] newX; // 释放临时数组
//     delete[] newY; // 释放临时数组

//     return finalSum; // 返回最终的L2距离的平方
// }

// V7

// inline float calculatedEuclideanDistance(const float* x, const float* y, size_t d) {
//     __m256 sum = _mm256_setzero_ps(); // 初始化累加器为0
//     size_t i;
//     for (i = 0; i <= d - 8; i += 8) {
//         __m256 mx = _mm256_loadu_ps(x + i); // 加载x的一部分
//         __m256 my = _mm256_loadu_ps(y + i); // 加载y的一部分
//         __m256 diff = _mm256_sub_ps(mx, my); // 计算差异
//         __m256 sqr = _mm256_mul_ps(diff, diff); // 计算平方
//         sum = _mm256_add_ps(sum, sqr); // 累加平方
//     }

//     // 处理剩余的元素
//     float residual = 0.0;
//     for (; i < d; ++i) {
//         float diff = x[i] - y[i];
//         residual += diff * diff;
//     }

//     // 将累加器中的值合并
//     __m128 low128 = _mm256_castps256_ps128(sum);
//     __m128 high128 = _mm256_extractf128_ps(sum, 1);
//     __m128 sum128 = _mm_add_ps(low128, high128);
//     sum128 = _mm_hadd_ps(sum128, sum128);
//     sum128 = _mm_hadd_ps(sum128, sum128);
//     float finalSum = _mm_cvtss_f32(sum128) + residual; // 将累加器中的值与处理剩余元素的结果相加

//     return finalSum; // 返回最终的L2距离的平方
// }

// V8
//  inline float calculatedEuclideanDistance(const float* x, const float* y, size_t d) {
//      __m256 sum = _mm256_setzero_ps(); // 初始化累加器为0
//      size_t i;
//      for (i = 0; i <= d - 8; i += 8) {
//          __m256 mx = _mm256_loadu_ps(x + i); // 假设x是对齐的
//          __m256 my = _mm256_loadu_ps(y + i); // 假设y是对齐的
//          __m256 diff = _mm256_sub_ps(mx, my);
//          // 使用FMA指令集合并乘法和累加
//          sum = _mm256_fmadd_ps(diff, diff, sum);
//      }

//     float residual = 0.0;
//     for (; i < d; ++i) {
//         float diff = x[i] - y[i];
//         residual += diff * diff;
//     }

//     // 减少水平加法的使用
//     float finalSum = residual;
//     for (int j = 0; j < 8; ++j) {
//         finalSum += ((float*)&sum)[j];
//     }

//     return finalSum; // 返回最终的L2距离的平方
// }

// V9
//  TRIBASE_IMPRECISE_FUNCTION_BEGIN
//  inline float calculatedEuclideanDistance(const float* x, const float* y, size_t d) {
//      size_t i;
//      float res = 0;
//      TRIBASE_IMPRECISE_LOOP
//      for (i = 0; i < d; i++) {
//          const float tmp = x[i] - y[i];
//          res += tmp * tmp;
//      }
//      return res;
//  }
//  TRIBASE_IMPRECISE_FUNCTION_END

// V9
// TRIBASE_IMPRECISE_FUNCTION_BEGIN
// inline float calculatedEuclideanDistance(const float* x, const float* y, size_t d) {
//     size_t i;
//     float res = 0;
//     TRIBASE_IMPRECISE_LOOP
//     for (i = 0; i < d; i++) {
//         const float tmp = x[i] - y[i];
//         res += tmp * tmp;
//     }
//     return res;
// }
// TRIBASE_IMPRECISE_FUNCTION_END

// V10
//  inline float calculatedEuclideanDistance(const float* x, const float* y, size_t d) {
//      // 直接调用FAISS的fvec_L2sqr函数
//      return faiss::fvec_L2sqr(x, y, d);
//  }
//  float calculatedEuclideanDistance(const float* vec1, const float* vec2, size_t size);

// Calculates the inner product between two vectors
inline float calculatedInnerProduct(const float* vec1, const float* vec2, size_t size) {
    return cblas_sdot(size, vec1, 1, vec2, 1);
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

#define FEPS 1e-4

inline float calculate_recall(const idx_t* I, const float* D, const idx_t* GT, const float* GD, size_t nq, size_t k, MetricType metric, size_t gt_k = 0) {
    if (gt_k == 0) {
        gt_k = k;
    }
    size_t true_correct = 0;
    size_t correct = 0;
    if (k > gt_k) {
        throw std::invalid_argument("k should be less than or equal to gt_k.");
    }
    for (size_t i = 0; i < nq; ++i) {
        std::unordered_set<idx_t> groundtruth(GT + i * gt_k, GT + i * gt_k + k);
        for (size_t j = 0; j < k; ++j) {
            if (I[i * k + j] == -1) {
                break;
            }
            if (groundtruth.find(I[i * k + j]) != groundtruth.end()) {
                true_correct++;
            }
        }
    }
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
                if (D[i * k + j] <= topK || relative_error(D[i * k + j], topK) < FEPS) {
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
                if (D[i * k + j] >= topK || relative_error(D[i * k + j], topK) < FEPS) {
                    correct++;
                } else {
                    std::cerr << std::format("D[{}, {}]= {} < topK= {}", i, j, D[i * k + j], topK) << std::endl;
                    assert(false);
                }
            }
        }
    }
    assert(1.0 * true_correct / correct > 0.99);
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