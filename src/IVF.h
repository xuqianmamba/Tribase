#ifndef IVF_H
#define IVF_H

#include <fstream>
#include <memory>
#include "common.h"

namespace tribase {
class IVF {
   public:
    size_t list_size;
    size_t d;
    size_t sub_k;
    OptLevel opt_level;

    std::unique_ptr<size_t[]> candidate_id;
    std::unique_ptr<float[]> candidate_codes;

    std::unique_ptr<float[]> candidate_norms;
    std::unique_ptr<float[]> candidate2centroid;
    std::unique_ptr<float[]> sqrt_candidate2centroid;
    std::unique_ptr<idx_t[]> sub_nearest_L2_id;
    std::unique_ptr<float[]> sub_nearest_L2_dis;
    std::unique_ptr<idx_t[]> sub_nearest_IP_id;
    std::unique_ptr<float[]> sub_nearest_IP_dis;
    std::unique_ptr<idx_t[]> sub_farest_IP_id;
    std::unique_ptr<float[]> sub_farest_IP_dis;

   public:
    // Constructor
    IVF(size_t listSize = 0, size_t d = 0, size_t subK = 0, OptLevel optLevel = OptLevel::OPT_ALL);

    // Destructor
    ~IVF();

    // Copy constructor and assignment operator are disabled
    IVF(const IVF&) = delete;
    IVF& operator=(const IVF&) = delete;

    // Move constructor and assignment operator
    IVF(IVF&&) noexcept;
    IVF& operator=(IVF&&) noexcept;

    void reset(size_t listSize, size_t d, size_t subK, OptLevel optLevel = OptLevel::OPT_ALL);

    void save_IVF(std::ostream& os) const;
    void load_IVF(std::istream& is);

    // Additional methods to manipulate the data can be added here
    size_t get_list_size() const { return list_size; }
    size_t get_d() const { return d; }
    size_t get_sub_k() const { return sub_k; }
    const float* get_candidate_codes() const { return candidate_codes.get(); }
    const float* get_candidate_codes(size_t i) const { return candidate_codes.get() + i * d; }
    const size_t* get_candidate_id() const { return candidate_id.get(); }
    const size_t get_candidate_id(size_t i) const { return candidate_id[i]; }
    const float* get_candidate_norms() const { return candidate_norms.get(); }
    const float* get_candidate2centroid() const { return candidate2centroid.get(); }
    const float get_candidate2centroid(size_t i) const { return candidate2centroid[i]; }
    const float* get_sqrt_candidate2centroid() const { return sqrt_candidate2centroid.get(); }
    const float get_sqrt_candidate2centroid(size_t i) const { return sqrt_candidate2centroid[i]; }
    const idx_t* get_sub_nearest_L2_id() const { return sub_nearest_L2_id.get(); }
    const idx_t get_sub_nearest_L2_id(size_t i, size_t k) const { return sub_nearest_L2_id[i * sub_k + k]; }
    const float* get_sub_nearest_L2_dis() const { return sub_nearest_L2_dis.get(); }
    const float get_sub_nearest_L2_dis(size_t i, size_t k) const { return sub_nearest_L2_dis[i * sub_k + k]; }
    const idx_t* get_sub_nearest_IP_id() const { return sub_nearest_IP_id.get(); }
    const idx_t get_sub_nearest_IP_id(size_t i, size_t k) const { return sub_nearest_IP_id[i * sub_k + k]; }
    const float* get_sub_nearest_IP_dis() const { return sub_nearest_IP_dis.get(); }
    const float get_sub_nearest_IP_dis(size_t i, size_t k) const { return sub_nearest_IP_dis[i * sub_k + k]; }
    const idx_t* get_sub_farest_IP_id() const { return sub_farest_IP_id.get(); }
    const idx_t get_sub_farest_IP_id(size_t i, size_t k) const { return sub_farest_IP_id[i * sub_k + k]; }
    const float* get_sub_farest_IP_dis() const { return sub_farest_IP_dis.get(); }
    const float get_sub_farest_IP_dis(size_t i, size_t k) const { return sub_farest_IP_dis[i * sub_k + k]; }
};
}  // namespace tribase

#endif  // IVF_H
