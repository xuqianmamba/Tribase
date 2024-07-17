#include "IndexFlatwithDistance.h"

#include <vector>
#include <iostream>

#include <faiss/IndexIVF.h>

#include <omp.h>
#include <cstdint>
#include <memory>
#include <mutex>

#include <iostream>
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <limits>
#include <memory>

#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>


namespace faiss {

void IndexFlatwithDistance::assign(idx_t n, const float* x, idx_t* labels, idx_t k, float* dis2nearest_center) {
    // std::cout<<"IndexFlatwithDistance::assign"<<std::endl;
    // std::cout<<"distances.resized"<<std::endl;
    // // 打印x数组的前100个值
    
    // std::cout << "x: " << x << std::endl;
    // std::cout << "labels: " << labels << std::endl;
    // // std::cout << "distances: " << distances.data() << std::endl;
    // std::cout << "this: " << this << std::endl;

    // std::cout<<"begin search"<<std::endl;
    IndexFlat::search(n, x, k, distances, labels, nullptr);
    // std::cout<<"end search"<<std::endl;

    #pragma omp parallel for
    for (idx_t i = 0; i < n; ++i) {
        dis2nearest_center[i] = distances[i * k];
    }

}
}