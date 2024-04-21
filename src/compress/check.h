#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <utility>
#include <cassert>


namespace repair_compress {

using value_t = int;
using cnt_t = int;

using pair_value_t = std::pair<value_t, value_t>;

template <typename T>
using pair_reverse_map_t = std::unordered_map<T, pair_value_t>;

bool check_compress(
    const value_t* original_data, 
    size_t n, 
    int dim, 
    const std::vector<cnt_t>& result_vlist,
    const std::vector<value_t>& result_elist,
    const pair_reverse_map_t<value_t>& rule
) ;
}// namespace repair_compress