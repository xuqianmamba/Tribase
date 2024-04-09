#pragma once

// basic int types and size_t
#include <cstdint>
#include <cstdio>

#if defined(__GNUC__) || defined(__clang__)
#define TRIBASE_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#define TRIBASE_DEPRECATED(msg)
#endif // GCC or Clang

// 自定义的宏来控制浮点运算的精度和优化行为
#if defined(_MSC_VER)
#define TRIBASE_IMPRECISE_LOOP
#define TRIBASE_IMPRECISE_FUNCTION_BEGIN \
    __pragma(float_control(precise, off, push))
#define TRIBASE_IMPRECISE_FUNCTION_END __pragma(float_control(pop))
#elif defined(__clang__)
#define TRIBASE_IMPRECISE_LOOP \
    _Pragma("clang loop vectorize(enable) interleave(enable)")

#if defined(__x86_64__) && (defined(__clang_major__) && (__clang_major__ > 10))
#define TRIBASE_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("float_control(precise, off, push)")
#define TRIBASE_IMPRECISE_FUNCTION_END _Pragma("float_control(pop)")
#else
#define TRIBASE_IMPRECISE_FUNCTION_BEGIN
#define TRIBASE_IMPRECISE_FUNCTION_END
#endif
#elif defined(__GNUC__)
#define TRIBASE_IMPRECISE_LOOP
#define TRIBASE_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("GCC push_options") \
    _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define TRIBASE_IMPRECISE_FUNCTION_END \
    _Pragma("GCC pop_options")
#else
#define TRIBASE_IMPRECISE_LOOP
#define TRIBASE_IMPRECISE_FUNCTION_BEGIN
#define TRIBASE_IMPRECISE_FUNCTION_END
#endif