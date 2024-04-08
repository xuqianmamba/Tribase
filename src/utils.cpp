#include "utils.h"
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    // _Pragma("float_control(precise, off, push)")

#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END \
    // _Pragma("float_control(pop)")

#define FAISS_PRAGMA_IMPRECISE_LOOP \
    // _Pragma("clang loop vectorize(enable) interleave(enable)")

namespace tribase {
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float calculatedEuclideanDistance(const float* vec1, const float* vec2, size_t size) {
    float distance = 0.0;
    // Calculate the squared difference for each dimension
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < size; ++i) {
        float diff = vec1[i] - vec2[i];
        distance += diff * diff;
    }
    return distance;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END
}  // namespace tribase