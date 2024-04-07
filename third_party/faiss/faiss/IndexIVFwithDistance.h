#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/Index.h>
#include <faiss/IndexFlatwithDistance.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/invlists/InvertedListswithDistance.h>
#include <faiss/utils/Heap.h>
#include <mutex>

namespace faiss {

// std::mutex mtx;

class IndexIVFwithDistance : public IndexIVFFlat {
   public:
    // 新的成员变量，用于存储InvertedListswithDistance对象
    std::vector<float> estimates;
    size_t total_skip_count = 0;
    size_t total_skip_count_large = 0;
    size_t total_skip_cos_count = 0;
    size_t total_count = 0;
    size_t total_compute_cos_skip_count=0;
    size_t total_compute_cos_skip_count_true=0;
    size_t total_compute_L2_skip_count_true=0;
    size_t sub_k;
    size_t sub_nlist;
    size_t sub_nprobe;
    float sub_sample;
    float log_interval;
    size_t total_used_sub_k_cos;
    size_t total_used_sub_k_l2;
    size_t total_used_sub_k_cos_count;
    size_t total_used_sub_k_l2_count;

    bool enable_triangle;
    bool enable_intersection;
    bool enable_sub_knn_cos;
    bool enable_sub_knn_l2;
    bool enable_lite_sub_knn;
    int opt_level;

    InvertedListswithDistance* invlistswithdist;
    // IndexFlatwithDistance* quantizer;

    IndexIVFwithDistance(
        IndexFlatL2* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric,
        size_t sub_k = 20,
        size_t sub_nlist = 20,
        size_t sub_nprobe = 1,
        float sub_sample = 0.1);
    ~IndexIVFwithDistance();

    void assign(idx_t n, const float* x, idx_t* labels, idx_t k, float* dis2nearest_center) const;

    void add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        const float* dis2nearest_center);

    void add_with_ids(idx_t n, const float* x, const idx_t* xids);

    void add(idx_t n, const float* x);

    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params = nullptr);

    void search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params = nullptr,
        IndexIVFStats* stats = nullptr,
        idx_t global_begin = 0);

    inline float* get_centroid_codes(idx_t list_no) {
        auto flat_quantizer = dynamic_cast<faiss::IndexFlatL2*>(quantizer);
        auto codes = flat_quantizer->get_xb();
        auto d = flat_quantizer->d;
        return codes + list_no * d;
    }

    // void search_preassigned(
    //     idx_t n,
    //     const float* x,
    //     idx_t k,
    //     const idx_t* assign,
    //     const float* centroid_dis,
    //     float* distances,
    //     idx_t* labels,
    //     bool store_pairs,
    //     const IVFSearchParameters* params,
    //     IndexIVFStats* stats) const;

    // void range_search_preassigned(
    //     idx_t nx,
    //     const float* x,
    //     float radius,
    //     const idx_t* keys,
    //     const float* coarse_dis,
    //     RangeSearchResult* result,
    //     bool store_pairs,
    //     const IVFSearchParameters* params,
    //     IndexIVFStats* stats) const;

    // void encode_vectors(
    //     idx_t n,
    //     const float* x,
    //     const idx_t* list_nos,
    //     uint8_t* codes,
    //     bool include_listnos) const;
};

// class Level1QuantizerwithDistance : public Level1Quantizer {
//     /// quantizer that maps vectors to inverted lists
//     IndexIVFwithDistance* quantizer = nullptr;

//     Level1QuantizerwithDistance(IndexIVFwithDistance* quantizer, size_t nlist)
//     : Level1Quantizer(static_cast<Index*>(quantizer), nlist), quantizer(quantizer) {
//         // ... other initialization code ...
//     }

//     /// number of inverted lists
//     size_t nlist = 0;

//     /**
//      * = 0: use the quantizer as index in a kmeans training
//      * = 1: just pass on the training set to the train() of the quantizer
//      * = 2: kmeans training on a flat index + add the centroids to the quantizer
//      */
//     char quantizer_trains_alone = 0;
//     bool own_fields = false; ///< whether object owns the quantizer

//     ClusteringParameters cp; ///< to override default clustering params
//     /// to override index used during clustering
//     Index* clustering_index = nullptr;

//     /// Trains the quantizer and calls train_residual to train sub-quantizers
//     void train_q1(
//             size_t n,
//             const float* x,
//             bool verbose,
//             MetricType metric_type);

//     /// compute the number of bytes required to store list ids
//     size_t coarse_code_size() const;
//     void encode_listno(idx_t list_no, uint8_t* code) const;
//     idx_t decode_listno(const uint8_t* code) const;

//     Level1QuantizerwithDistance();

//     ~Level1QuantizerwithDistance();
// };

}  // namespace faiss