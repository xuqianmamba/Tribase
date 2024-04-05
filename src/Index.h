#ifndef INDEX_H
#define INDEX_H

#include <memory>
#include "Clustering.h"
#include "common.h"

namespace tribase {

class IVF {};  // IVF类的简单声明，您需要根据实际情况来定义它

class Index {
   public:
    Index(size_t d, size_t nlist, size_t nprobe, MetricType metric = MetricType::METRIC_L2);

    void train(size_t n, std::unique_ptr<float[]>& codes);

    void search(size_t n, const float* queries);

    // 其他查询方法的声明

   private:
    MetricType metric;
    size_t sub_k;
    size_t d;
    size_t nlist;
    size_t nprobe;
    std::unique_ptr<IVF[]> lists;
    std::unique_ptr<float[]> centroid_codes;
};

}  // namespace tribase

#endif  // INDEX_H