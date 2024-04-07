#include "IVF.h"

namespace tribase {
IVF::IVF(size_t listSize, size_t d, size_t subK, OptLevel optLevel) {
    reset(listSize, d, subK, optLevel);
}

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

void IVF::reset(size_t listSize, size_t d, size_t subK, OptLevel optLevel) {
    this->list_size = listSize;
    this->d = d;
    this->sub_k = subK;
    candidate_id = std::make_unique<size_t[]>(listSize);
    candidate_codes = std::make_unique<float[]>(listSize * d);
    if ((optLevel & OptLevel::OPT_TRIANGLE) || (optLevel & OptLevel::OPT_SUBNN_IP)) {
        candidate2centroid = std::make_unique<float[]>(listSize);
        sqrt_candidate2centroid = std::make_unique<float[]>(listSize);
    }
    if (optLevel & OptLevel::OPT_SUBNN_L2) {
        sub_nearest_L2_id = std::make_unique<idx_t[]>(listSize * subK);
        sub_nearest_L2_dis = std::make_unique<float[]>(listSize * subK);
    }
    if (optLevel & OptLevel::OPT_SUBNN_IP) {
        sub_nearest_IP_id = std::make_unique<idx_t[]>(listSize * subK);
        sub_nearest_IP_dis = std::make_unique<float[]>(listSize * subK);
        sub_farest_IP_id = std::make_unique<idx_t[]>(listSize * subK);
        sub_farest_IP_dis = std::make_unique<float[]>(listSize * subK);
    }
}

void IVF::save_IVF(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&list_size), sizeof(size_t));
    os.write(reinterpret_cast<const char*>(&d), sizeof(size_t));
    os.write(reinterpret_cast<const char*>(&sub_k), sizeof(size_t));
    os.write(reinterpret_cast<const char*>(&opt_level), sizeof(OptLevel));
    os.write(reinterpret_cast<const char*>(candidate_id.get()), list_size * sizeof(size_t));
    os.write(reinterpret_cast<const char*>(candidate_codes.get()), list_size * d * sizeof(float));
    if ((opt_level & OptLevel::OPT_TRIANGLE) || (opt_level & OptLevel::OPT_SUBNN_IP)) {
        os.write(reinterpret_cast<const char*>(candidate2centroid.get()), list_size * sizeof(float));
        os.write(reinterpret_cast<const char*>(sqrt_candidate2centroid.get()), list_size * sizeof(float));
    }
    if (opt_level & OptLevel::OPT_SUBNN_L2) {
        os.write(reinterpret_cast<const char*>(sub_nearest_L2_id.get()), list_size * sub_k * sizeof(idx_t));
        os.write(reinterpret_cast<const char*>(sub_nearest_L2_dis.get()), list_size * sub_k * sizeof(float));
    }
    if (opt_level & OptLevel::OPT_SUBNN_IP) {
        os.write(reinterpret_cast<const char*>(sub_nearest_IP_id.get()), list_size * sub_k * sizeof(idx_t));
        os.write(reinterpret_cast<const char*>(sub_nearest_IP_dis.get()), list_size * sub_k * sizeof(float));
        os.write(reinterpret_cast<const char*>(sub_farest_IP_id.get()), list_size * sub_k * sizeof(idx_t));
        os.write(reinterpret_cast<const char*>(sub_farest_IP_dis.get()), list_size * sub_k * sizeof(float));
    }
}

void IVF::load_IVF(std::istream& is) {
    is.read(reinterpret_cast<char*>(&list_size), sizeof(size_t));
    is.read(reinterpret_cast<char*>(&d), sizeof(size_t));
    is.read(reinterpret_cast<char*>(&sub_k), sizeof(size_t));
    is.read(reinterpret_cast<char*>(&opt_level), sizeof(OptLevel));
    candidate_id = std::make_unique<size_t[]>(list_size);
    candidate_codes = std::make_unique<float[]>(list_size * d);
    is.read(reinterpret_cast<char*>(candidate_id.get()), list_size * sizeof(size_t));
    is.read(reinterpret_cast<char*>(candidate_codes.get()), list_size * d * sizeof(float));
    if ((opt_level & OptLevel::OPT_TRIANGLE) || (opt_level & OptLevel::OPT_SUBNN_IP)) {
        candidate2centroid = std::make_unique<float[]>(list_size);
        sqrt_candidate2centroid = std::make_unique<float[]>(list_size);
        is.read(reinterpret_cast<char*>(candidate2centroid.get()), list_size * sizeof(float));
        is.read(reinterpret_cast<char*>(sqrt_candidate2centroid.get()), list_size * sizeof(float));
    }
    if (opt_level & OptLevel::OPT_SUBNN_L2) {
        sub_nearest_L2_id = std::make_unique<idx_t[]>(list_size * sub_k);
        sub_nearest_L2_dis = std::make_unique<float[]>(list_size * sub_k);
        is.read(reinterpret_cast<char*>(sub_nearest_L2_id.get()), list_size * sub_k * sizeof(idx_t));
        is.read(reinterpret_cast<char*>(sub_nearest_L2_dis.get()), list_size * sub_k * sizeof(float));
    }
    if (opt_level & OptLevel::OPT_SUBNN_IP) {
        sub_nearest_IP_id = std::make_unique<idx_t[]>(list_size * sub_k);
        sub_nearest_IP_dis = std::make_unique<float[]>(list_size * sub_k);
        sub_farest_IP_id = std::make_unique<idx_t[]>(list_size * sub_k);
        sub_farest_IP_dis = std::make_unique<float[]>(list_size * sub_k);
        is.read(reinterpret_cast<char*>(sub_nearest_IP_id.get()), list_size * sub_k * sizeof(idx_t));
        is.read(reinterpret_cast<char*>(sub_nearest_IP_dis.get()), list_size * sub_k * sizeof(float));
        is.read(reinterpret_cast<char*>(sub_farest_IP_id.get()), list_size * sub_k * sizeof(idx_t));
        is.read(reinterpret_cast<char*>(sub_farest_IP_dis.get()), list_size * sub_k * sizeof(float));
    }
}

}  // namespace tribase