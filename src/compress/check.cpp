#include "check.h"


namespace repair_compress {

bool check_compress(
    const value_t* original_data, 
    size_t n, 
    int dim, 
    const std::vector<cnt_t>& result_vlist,
    const std::vector<value_t>& result_elist,
    const pair_reverse_map_t<value_t>& rule
) {

    std::vector<value_t> restored_data; 
    
    auto restore_value = [&](const value_t& val, auto& restore_value_ref) -> void {
        auto it = rule.find(val);
        if (it != rule.end()) {
            if (rule.find(it->second.first) != rule.end()) {
                restore_value_ref(it->second.first, restore_value_ref);
            } else {
                restored_data.push_back(it->second.first);
            }
            if (rule.find(it->second.second) != rule.end()) {
                restore_value_ref(it->second.second, restore_value_ref);
            } else {
                restored_data.push_back(it->second.second);
            }
        }  else {
            restored_data.push_back(val);
        }
    };

    for (auto val : result_elist) {
        restore_value(val, restore_value);
    }


    if (restored_data.size() != n * dim) {
        std::cerr << "Error: Size mismatch between restored and original data. "
                << "Original size: " << n * dim << ", Restored size: " << restored_data.size() << std::endl;
        return false;
    }

    for (size_t i = 0; i < n; ++i) {
        if (original_data[i] != restored_data[i]) {
            std::cerr << "Error: Data mismatch at index " << i << std::endl;
            return false;
        }
    }

    return true; 
}

}