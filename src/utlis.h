#ifndef UTILS_H
#define UTILS_H

#include <chrono>
#include <cmath>
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
    void reset() {
        start = std::chrono::high_resolution_clock::now();
    }

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
float calculatedEuclideanDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    // Ensure the vectors are of the same size
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    float distance = 0.0;
    // Calculate the squared difference for each dimension
    for (size_t i = 0; i < vec1.size(); ++i) {
        float diff = vec1[i] - vec2[i];
        distance += diff * diff;
    }

    return sqrt(distance);
}

// Calculates the magnitude (length) of a vector
float vectorMagnitude(const std::vector<float>& vec) {
    float sum = 0.0;
    for (float val : vec) {
        sum += val * val;
    }
    return sqrt(sum);
}

// Calculates the cosine similarity between two vectors
float calculateCosineSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    // Ensure the vectors are of the same size
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    float dotProduct = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        dotProduct += vec1[i] * vec2[i];
    }

    float magnitude1 = vectorMagnitude(vec1);
    float magnitude2 = vectorMagnitude(vec2);

    if (magnitude1 == 0 || magnitude2 == 0) {
        throw std::invalid_argument("One or both vectors are zero vectors.");
    }

    return dotProduct / (magnitude1 * magnitude2);
}

}  // namespace tribase

#endif  // UTILS_H