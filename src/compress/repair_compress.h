#pragma once
#include <unordered_map>
#include <utility>
#include <vector>

namespace repair_compress {
using value_t = int;
using cnt_t = int;

template <typename T>
struct pair_hash {
    std::size_t operator()(const std::pair<T, T>& pair) const {
        return std::hash<T>()(pair.first) * 998244353 + std::hash<T>()(pair.second);
    }
};

using pair_value_t = std::pair<value_t, value_t>;
using pair_value_t_hash = pair_hash<value_t>;

using pair_pos_t = std::pair<cnt_t, cnt_t>;
using pair_pos_t_hash = pair_hash<cnt_t>;

// pair_value_t -> T map
template <typename T>
using pair_map_t = std::unordered_map<pair_value_t, T, pair_value_t_hash>;

std::tuple<pair_map_t<value_t>, std::vector<cnt_t>, std::vector<value_t>> generate_rule(value_t* raw_data, size_t n, size_t dim, cnt_t threshold = 3, bool verbose = false);
}  // namespace repair_compress