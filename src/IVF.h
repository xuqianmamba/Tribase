#ifndef IVF_H
#define IVF_H

#include <memory>

class IVF {
private:
    size_t list_size;
    size_t sub_k;
    std::unique_ptr<size_t[]> candidate_id;
    std::unique_ptr<float[]> candidate_codes;
    std::unique_ptr<float[]> candidate2centroid;
    std::unique_ptr<float[]> sqrt_candidate2centroid;
    std::unique_ptr<size_t[]> sub_nearest_L2_id;
    std::unique_ptr<float[]> sub_nearest_L2_dis;
    std::unique_ptr<size_t[]> sub_nearest_IP_id;
    std::unique_ptr<float[]> sub_nearest_IP_dis;
    std::unique_ptr<size_t[]> sub_farest_IP_id;
    std::unique_ptr<float[]> sub_farest_IP_dis;

public:
    // Constructor
    IVF(size_t listSize, size_t subK);

    // Destructor
    ~IVF();

    // Copy constructor and assignment operator are disabled
    IVF(const IVF&) = delete;
    IVF& operator=(const IVF&) = delete;

    // Move constructor and assignment operator
    IVF(IVF&&) noexcept;
    IVF& operator=(IVF&&) noexcept;

    // Additional methods to manipulate the data can be added here
};

#endif  // IVF_H
