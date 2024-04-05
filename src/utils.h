#ifndef UTILS_H
#define UTILS_H

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tribase {

// A class for measuring execution time
class Stopwatch {
   public:
    // Constructor initializes the start time
    Stopwatch()
        : start(std::chrono::high_resolution_clock::now()) {}

    // Resets the start time to the current time
    void reset() { start = std::chrono::high_resolution_clock::now(); }

    // Returns the elapsed time in milliseconds since the stopwatch was started or last reset
    double elapsedMilliseconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

   private:
    // The start time
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

// Calculates the Euclidean distance between two vectors
float calculatedEuclideanDistance(const float* vec1, const float* vec2, size_t size) {
    float distance = 0.0;
    // Calculate the squared difference for each dimension
    for (size_t i = 0; i < size; ++i) {
        float diff = vec1[i] - vec2[i];
        distance += diff * diff;
    }

    return sqrt(distance);
}

// Calculates the inner product between two vectors
float calculatedInnerProduct(const float* vec1, const float* vec2, size_t size) {
    float distance = 0.0;
    // Calculate the squared difference for each dimension
    for (size_t i = 0; i < size; ++i) {
        distance += vec1[i] * vec2[i];
    }
    return distance;
}

// Calculates the magnitude (length) of a vector
float vectorMagnitude(const float* vec, size_t size) {
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

// Calculates the cosine similarity between two vectors
float calculateCosineSimilarity(const float* vec1, const float* vec2, size_t size) {
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

#endif  // UTILS_H