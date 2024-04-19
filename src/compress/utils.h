#pragma once
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdfloat>
#include <string>
#include <tuple>

inline void covert2f16(int* target, float* source, size_t size) {
    for (size_t i = 0; i < size; i++) {
        // target[i] = static_cast<std::float16_t>(source[i]);
        std::float16_t f16 = static_cast<std::float16_t>(source[i]);
        std::memcpy(target + i, &f16, sizeof(std::float16_t));
    }
}

inline void covert2f32(float* target, int* source, size_t size) {
    for (size_t i = 0; i < size; i++) {
        std::float16_t f16;
        std::memcpy(&f16, source + i, sizeof(std::float16_t));
        target[i] = f16;
    }
}

inline void covert2fixed(int* fixed_array, const float* float_array, size_t array_size, bool sign_bit, int left_digits, int right_digits) {
    int total_digits = left_digits + right_digits + sign_bit;
    int max_value;
    if (sign_bit) {
        max_value = (1 << (total_digits - 1)) - 1;
    } else {
        max_value = (1 << total_digits) - 1;
    }

    float p = pow(2, right_digits);
    for (size_t i = 0; i < array_size; ++i) {
        float scaled_float = abs(float_array[i] * p);
        int fixed_number = static_cast<int>(round(scaled_float));

        if (fixed_number > max_value) {
            std::cerr << std::format("Fixed point overflow occurred: {} -> {} > {}", float_array[i], fixed_number, max_value) << std::endl;
            throw std::overflow_error("Fixed point overflow occurred");
        }

        if (sign_bit && float_array[i] < 0)
            fixed_number |= 1 << (total_digits - 1);  // 将符号位设置为1

        fixed_array[i] = fixed_number;
    }
}

inline void revert4fixed(float* float_array, const int* fixed_array, size_t array_size, bool sign_bit, int left_digits, int right_digits) {
    int total_digits = left_digits + right_digits + sign_bit;
    float p = pow(2, right_digits);
    for (size_t i = 0; i < array_size; ++i) {
        int fixed_number = fixed_array[i];
        if (sign_bit && fixed_number & (1 << (total_digits - 1))) {
            fixed_number &= ~(1 << (total_digits - 1));
            float_array[i] = -fixed_number / p;
        } else {
            float_array[i] = fixed_number / p;
        }
    }
}

inline std::tuple<bool, int, int> autocovert2fixed(int* fixed_array, const float* float_array, size_t array_size, int total_digits = -1, int right_digits = -1) {
    float mx = std::numeric_limits<float>::min();
    float closest = std::numeric_limits<float>::max();
    bool sign_bit;

#pragma omp parallel for reduction(max : mx) reduction(min : closest)
    for (size_t i = 0; i < array_size; ++i) {
        if (std::abs(float_array[i]) < closest && float_array[i] != 0) {
            closest = std::abs(float_array[i]);
        }
        if (abs(float_array[i]) > mx) {
            mx = abs(float_array[i]);
        }
        if (float_array[i] < 0) {
            sign_bit = true;
        }
    }

    int left_digits = 0;

    while (mx >= 1) {
        mx /= 2;
        left_digits++;
    }

    if (right_digits < 0 && total_digits < 0) {
        throw std::invalid_argument("Either right_digits or total_digits must be specified");
    }

    if (right_digits < 0) {
        right_digits = 0;
        while (closest < 1) {
            closest *= 2;
            right_digits++;
        }
    } else {
        // right_digits = right_digits; // use the specified value
    }

    if (total_digits < 0) {
        total_digits = left_digits + right_digits + sign_bit;
    } else {
        right_digits = total_digits - left_digits - sign_bit;  // use the specified value
    }

    int max_value;

    if (sign_bit) {
        max_value = (1 << (total_digits - 1)) - 1;
    } else {
        max_value = (1 << total_digits) - 1;
    }

    float p = pow(2, right_digits);
#pragma omp parallel for
    for (size_t i = 0; i < array_size; ++i) {
        float scaled_float = abs(float_array[i] * p);
        int fixed_number = static_cast<int>(round(scaled_float));

        if (fixed_number > max_value) {
            std::cerr << std::format("Fixed point overflow occurred: {} -> {} > {}", float_array[i], fixed_number, max_value) << std::endl;
            throw std::overflow_error("Fixed point overflow occurred");
        }

        if (sign_bit && float_array[i] < 0)
            fixed_number |= 1 << (total_digits - 1);  // 将符号位设置为1

        fixed_array[i] = fixed_number;
    }
    return std::make_tuple(sign_bit, left_digits, right_digits);
}