#pragma once
#include <cmath>
#include <limits>
#include <memory>

template <typename T>
concept HasQSqrtIeeeFastFloat = std::numeric_limits<float>::is_iec559 and sizeof(float) == 4;

// Standard float systems.
template <size_t Iterations = 1>
    requires(HasQSqrtIeeeFastFloat<float>)
[[gnu::always_inline, nodiscard]] constexpr float quick_rsqrt(float number) noexcept {
    return []<size_t... I>(float number, std::index_sequence<I...>) noexcept [[gnu::always_inline]] {
        constexpr float threehalfs = 1.5f;
        constexpr float half = 0.5f;
        auto i = std::bit_cast<int32_t>(number);
        auto x2 = number * half;
        i = 0x5f3759df - (i >> 1);
        auto y = std::bit_cast<float>(i);
        (
            (y = y * (threehalfs - (x2 * y * y)), I),
            ...);
        return y;
    }(number, std::make_index_sequence<Iterations>());
}

// Fallback for non-standard float systems.
template <size_t Iterations = 1>
    requires(not HasQSqrtIeeeFastFloat<float>)
[[gnu::always_inline, nodiscard]] constexpr float quick_rsqrt(float number) noexcept {
    return 1.0f / std::sqrt(number);
}