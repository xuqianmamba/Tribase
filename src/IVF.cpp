#include "IVF.h"

namespace tribase {
IVF::IVF(size_t listSize, size_t d, size_t subK)
    : list_size(listSize),
      d(d),
      sub_k(subK),
      candidate_id(std::make_unique<size_t[]>(listSize)),
      candidate_codes(std::make_unique<float[]>(listSize * d)),
      candidate2centroid(std::make_unique<float[]>(listSize)),
      sqrt_candidate2centroid(std::make_unique<float[]>(listSize)),
      sub_nearest_L2_id(std::make_unique<idx_t[]>(listSize * subK)),
      sub_nearest_L2_dis(std::make_unique<float[]>(listSize * subK)),
      sub_nearest_IP_id(std::make_unique<idx_t[]>(listSize * subK)),
      sub_nearest_IP_dis(std::make_unique<float[]>(listSize * subK)),
      sub_farest_IP_id(std::make_unique<idx_t[]>(listSize * subK)),
      sub_farest_IP_dis(std::make_unique<float[]>(listSize * subK)) {}

IVF::~IVF() {}

IVF::IVF(IVF&& other) noexcept {
    list_size = other.list_size;
    sub_k = other.sub_k;
    candidate_id = std::move(other.candidate_id);
    candidate_codes = std::move(other.candidate_codes);
    candidate2centroid = std::move(other.candidate2centroid);
    sqrt_candidate2centroid = std::move(other.sqrt_candidate2centroid);
    sub_nearest_L2_id = std::move(other.sub_nearest_L2_id);
    sub_nearest_L2_dis = std::move(other.sub_nearest_L2_dis);
    sub_nearest_IP_id = std::move(other.sub_nearest_IP_id);
    sub_nearest_IP_dis = std::move(other.sub_nearest_IP_dis);
    sub_farest_IP_id = std::move(other.sub_farest_IP_id);
    sub_farest_IP_dis = std::move(other.sub_farest_IP_dis);
}

IVF& IVF::operator=(IVF&& other) noexcept {
    if (this != &other) {
        list_size = other.list_size;
        sub_k = other.sub_k;
        candidate_id = std::move(other.candidate_id);
        candidate_codes = std::move(other.candidate_codes);
        candidate2centroid = std::move(other.candidate2centroid);
        sqrt_candidate2centroid = std::move(other.sqrt_candidate2centroid);
        sub_nearest_L2_id = std::move(other.sub_nearest_L2_id);
        sub_nearest_L2_dis = std::move(other.sub_nearest_L2_dis);
        sub_nearest_IP_id = std::move(other.sub_nearest_IP_id);
        sub_nearest_IP_dis = std::move(other.sub_nearest_IP_dis);
        sub_farest_IP_id = std::move(other.sub_farest_IP_id);
        sub_farest_IP_dis = std::move(other.sub_farest_IP_dis);
    }
    return *this;
}

void IVF::reset(size_t listSize, size_t d, size_t subK) {
    this->list_size = listSize;
    this->d = d;
    this->sub_k = subK;
    candidate_id = std::make_unique<size_t[]>(listSize);
    candidate_codes = std::make_unique<float[]>(listSize * d);
    candidate2centroid = std::make_unique<float[]>(listSize);
    sqrt_candidate2centroid = std::make_unique<float[]>(listSize);
    sub_nearest_L2_id = std::make_unique<idx_t[]>(listSize * subK);
    sub_nearest_L2_dis = std::make_unique<float[]>(listSize * subK);
    sub_nearest_IP_id = std::make_unique<idx_t[]>(listSize * subK);
    sub_nearest_IP_dis = std::make_unique<float[]>(listSize * subK);
    sub_farest_IP_id = std::make_unique<idx_t[]>(listSize * subK);
    sub_farest_IP_dis = std::make_unique<float[]>(listSize * subK);
}

}  // namespace tribase