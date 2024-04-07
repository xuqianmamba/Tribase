#pragma once
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
#include <faiss/IndexIVF.h>
#include <faiss/IndexFlat.h>

namespace faiss{
class InvertedListswithDistance : public faiss::ArrayInvertedLists {
public:
    std::vector<std::vector<float>> distances;
    std::vector<std::vector<float>> sqrt_distances;
    size_t ref_vector_total;
    std::vector<std::vector<std::vector<float>>> mcos;
    std::vector<std::vector<float>> nearest_dis;  // inner product distance
    std::vector<std::vector<idx_t>> nearest_idx;  // inner product distance index
    std::vector<std::vector<float>> nearest_dis_l2;  // inner product distance
    std::vector<std::vector<idx_t>> nearest_idx_l2;  // inner product distance index

    InvertedListswithDistance(size_t nlist, size_t code_size);

    const float* get_distances(size_t list_no) const;
    const float* get_sqrt_distances(size_t list_no) const;

    const std::vector<std::vector<float>> get_mcos(size_t list_no) const{
        return mcos[list_no];
    }

    const float* get_single_mcos(size_t list_no, size_t offset) const{
        return mcos[list_no][offset].data();
    }

    const float* get_nearest_dis(size_t list_no) const {
        return nearest_dis[list_no].data();
    }

    const idx_t* get_nearest_idx_l2(size_t list_no) const {
        return nearest_idx_l2[list_no].data();
    }

    const idx_t* get_nearest_idx(size_t list_no) const {
        return nearest_idx[list_no].data();
    }

    const float* get_nearest_dis_l2(size_t list_no) const {
        return nearest_dis_l2[list_no].data();
    }

    float get_single_distance(size_t list_no, size_t offset) const;

    size_t add_entry(size_t list_no, idx_t theid, const uint8_t* code, float distance);

    size_t add_entries(size_t list_no, size_t n_entry, const idx_t* ids, const uint8_t* code, const float* distance);

    void resize(size_t list_no, size_t new_size) override;

    void release_distances(size_t list_no, const float* distances) const;
    
    void sort_all_lists();

    ~InvertedListswithDistance();
};
}