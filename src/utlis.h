#ifndef UTILS_H
#define UTILS_H

#include <chrono>
#include <vector>
#include <cmath>

namespace tribase {

// A class for measuring execution time
class Stopwatch {
public:
    // Constructor initializes the start time
    Stopwatch() : start(std::chrono::high_resolution_clock::now()) {}

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

} // namespace tribase

#endif // UTILS_H