#ifndef INDEX_FLAT_WITH_DISTANCE_H
#define INDEX_FLAT_WITH_DISTANCE_H

#include <faiss/IndexFlat.h>
#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/Index.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/Heap.h>
#include <faiss/IndexIVFFlat.h>

namespace faiss {

class IndexFlatwithDistance : public IndexFlat {
public:
    // 新的成员变量，用于存储距离
    float* distances;
    // std::vector<float> distances;

    IndexFlatwithDistance(idx_t d) : IndexFlat(d) {}

    void assign(idx_t n, const float* x, idx_t* labels, idx_t k, float* dis2nearest_center);

};

} // namespace faiss

#endif