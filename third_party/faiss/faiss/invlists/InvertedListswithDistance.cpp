#include "InvertedListswithDistance.h"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <numeric>

namespace faiss {

faiss::InvertedListswithDistance::InvertedListswithDistance(size_t nlist, size_t code_size)
    : faiss::ArrayInvertedLists(nlist, code_size) {
    distances.resize(nlist);
    sqrt_distances.resize(nlist);
    mcos.resize(nlist);
    nearest_dis.resize(nlist);
    nearest_idx.resize(nlist);
    nearest_dis_l2.resize(nlist);
    nearest_idx_l2.resize(nlist);
    ref_vector_total = 16;
}

const float* InvertedListswithDistance::get_distances(size_t list_no) const {
    assert(list_no < nlist);
    return distances[list_no].data();
}

const float* InvertedListswithDistance::get_sqrt_distances(size_t list_no) const {
    assert(list_no < nlist);
    return sqrt_distances[list_no].data();
}

float InvertedListswithDistance::get_single_distance(size_t list_no, size_t offset) const {
    // assert(offset < list_size(list_no));
    offset = std::min(offset, list_size(list_no) - 1);
    const float* dists = get_distances(list_no);
    float dist = dists[offset];
    // 如果你有一个release_distances方法，你应该在这里调用它
    release_distances(list_no, dists);
    return dist;
}

size_t InvertedListswithDistance::add_entries(size_t list_no, size_t n_entry, const idx_t* ids, const uint8_t* code, const float* distance) {
    size_t o = faiss::ArrayInvertedLists::add_entries(list_no, n_entry, ids, code);
    distances[list_no].resize(o + n_entry);
    memcpy(&distances[list_no][o], distance, sizeof(distance[0]) * n_entry);
    return o;
}

size_t InvertedListswithDistance::add_entry(size_t list_no, idx_t theid, const uint8_t* code, float distance) {
    return add_entries(list_no, 1, &theid, code, &distance);
}

void InvertedListswithDistance::resize(size_t list_no, size_t new_size) {
    distances[list_no].resize(new_size);
    faiss::ArrayInvertedLists::resize(list_no, new_size);
}

void InvertedListswithDistance::release_distances(size_t list_no, const float* distances) const {
    // 什么都不做
}

void InvertedListswithDistance::sort_all_lists() {
    // std::cout<<"in sort_all_lists!"<<std::endl;
    // for (size_t i = 0; i < nlist; ++i) {
    //     std::cout << "Before sorting list " << i << ": ";
    //     for (size_t j = 0; j < std::min(size_t(10), distances[i].size()); ++j) {
    //         std::cout << distances[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "Before sorting id " << i << ": ";
    //     for (size_t j = 0; j < std::min(size_t(10), ids[i].size()); ++j) {
    //         std::cout << ids[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    for (size_t i = 0; i < nlist; ++i) {
        // 创建一个索引数组
        std::vector<size_t> idx(distances[i].size());
        std::iota(idx.begin(), idx.end(), 0);

        // 根据距离对索引数组进行排序
        std::sort(idx.begin(), idx.end(),
                  [this, i](size_t a, size_t b) { return distances[i][a] < distances[i][b]; });

        // 使用索引数组对距离、ID和编码进行排序
        std::vector<float> sorted_distances(distances[i].size());
        std::vector<idx_t> sorted_ids(ids[i].size());
        std::vector<uint8_t> sorted_codes(codes[i].size());
        for (size_t j = 0; j < idx.size(); ++j) {
            sorted_distances[j] = distances[i][idx[j]];
            // sorted_distances[j] = sqrt(distances[i][idx[j]]); // sqrt debug
            sorted_ids[j] = ids[i][idx[j]];
            memcpy(&sorted_codes[j * code_size], &codes[i][idx[j] * code_size], code_size);
        }
        // 更新距离、ID和编码
        distances[i] = std::move(sorted_distances);
        ids[i] = std::move(sorted_ids);
        codes[i] = std::move(sorted_codes);
    }
    // 打印排序后的数组
    // for (size_t i = 0; i < nlist; ++i) {
    //     std::cout << "After sorting list " << i << ": ";
    //     for (size_t j = 0; j < std::min(size_t(10), distances[i].size()); ++j) {
    //         std::cout << distances[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    //  for (size_t i = 0; i < nlist; ++i) {
    //     std::cout << "After sorting ids " << i << ": ";
    //     for (size_t j = 0; j < std::min(size_t(10), ids[i].size()); ++j) {
    //         std::cout << ids[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
}

InvertedListswithDistance::~InvertedListswithDistance() {}
}  // namespace faiss