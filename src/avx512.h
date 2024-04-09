#pragma once

#include <immintrin.h>
#include <cmath>
#include <cstddef>

namespace tribase {

// 简化的512位向量类，只处理浮点数
class simd512f {
public:
    __m512 data;

    simd512f() : data(_mm512_setzero_ps()) {}

    explicit simd512f(float val) : data(_mm512_set1_ps(val)) {}

    explicit simd512f(const float* ptr) : data(_mm512_loadu_ps(ptr)) {}

    // 加载未对齐的数据
    void loadu(const float* ptr) {
        data = _mm512_loadu_ps(ptr);
    }

    // 存储未对齐的数据
    void storeu(float* ptr) const {
        _mm512_storeu_ps(ptr, data);
    }

    // 计算两个向量的差的平方，并累加
    simd512f& accumulate_square_diff(const simd512f& other) {
        data = _mm512_fmadd_ps(_mm512_sub_ps(data, other.data), _mm512_sub_ps(data, other.data), data);
        return *this;
    }

    // 水平加法，将向量中的所有元素相加
    float horizontal_sum() const {
        __m256 low = _mm512_castps512_ps256(data);
        __m256 high = _mm512_extractf32x8_ps(data, 1);
        __m256 sum256 = _mm256_add_ps(low, high);
        __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
        sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 0x55));
        return _mm_cvtss_f32(sum128);
    }
};

// 计算两个向量之间的L2距离的平方
// float calculateL2Distance(const float* x, const float* y, size_t d) {
//     simd512f sum;
//     size_t i;
//     for (i = 0; i <= d - 16; i += 16) {
//         simd512f vx(x + i), vy(y + i);
//         sum.accumulate_square_diff(vy);
//     }

//     float residual = 0.0f;
//     for (; i < d; ++i) {
//         float diff = x[i] - y[i];
//         residual += diff * diff;
//     }

//     return sum.horizontal_sum() + residual;
// }

} // namespace simple