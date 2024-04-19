#include "repair_compress.h"

#include <omp.h>
#include <algorithm>
#include <boost/heap/fibonacci_heap.hpp>
#include <chrono>
#include <format>
#include <iostream>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// #define NDEBUG

namespace repair_compress {

const int LOG_INTERVAL = 1000;

// store the raw pos for convenience of deletion
struct list_item_t {
    pair_pos_t pos;
    value_t value;
    list_item_t(cnt_t begin, cnt_t end, value_t value)
        : pos(begin, end), value(value) {}

    bool operator==(const list_item_t& other) const {
        return pos == other.pos && value == other.value;
    }
};

using list_t = std::list<list_item_t>;

struct list_iter_with_pos {
    pair_pos_t pos;
    list_t::iterator it;

    bool operator==(const list_iter_with_pos& other) const {
        return pos == other.pos;
    }
};

struct list_iter_with_pos_hash {
    std::size_t operator()(const list_iter_with_pos& item) const {
        return pair_hash<cnt_t>()(item.pos);
    }
};

// store the pair for convenience of update
struct pair_count_item_t {
    pair_value_t pair;
    cnt_t count;

    pair_count_item_t(const pair_value_t& pair, cnt_t count)
        : pair(pair), count(count) {}

    bool operator<(const pair_count_item_t& other) const {
        return count < other.count;
    }

    static pair_count_item_t decreased_item(const pair_count_item_t& item) {
        return pair_count_item_t(item.pair, item.count - 1);
    }

    static pair_count_item_t increased_item(const pair_count_item_t& item) {
        return pair_count_item_t(item.pair, item.count + 1);
    }
};

using heap_t = boost::heap::fibonacci_heap<pair_count_item_t>;
// using heap_t = boost::heap::pairing_heap<pair_count_item_t>;

std::tuple<pair_map_t<value_t>, std::vector<cnt_t>, std::vector<value_t>> generate_rule(value_t* raw_data, size_t n, size_t dim, cnt_t threshold, bool verbose) {
    value_t max_value = *std::max_element(raw_data, raw_data + n * dim);
    value_t max_sep = max_value + 1;
    value_t max_pair_value = max_value + n + 1;

    if (max_value > std::numeric_limits<value_t>::max() - n - 100) {
        throw std::runtime_error("There is not enough numbers (<100) to store auxiliary value");
    } else {
        if (verbose) {
            std::cout << std::format("There is {} numbers for new pair", std::numeric_limits<value_t>::max() - max_value - n) << std::endl;
        }
    }

    list_t data;

    // vector sep vector sep ... sep vector sep
    {
        size_t index = 0;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < dim; j++) {
                data.emplace_back(list_item_t(index, index + 1, raw_data[i * dim + j]));
                index++;
            }
            data.emplace_back(list_item_t(index, index + 1, max_sep++));
            index++;
        }

        if (verbose) {
            std::cout << "Add sep done" << std::endl;
        }
    }

    pair_map_t<value_t> pair_rule;  // pair -> pair_value

    pair_map_t<cnt_t> pair_count;  // pair -> count (only used for init, not used later)

    pair_map_t<std::unordered_set<list_iter_with_pos, list_iter_with_pos_hash>> pair_header;  // pair -> all left side variable (list_iter_with_pos) it appears

    // init, scan every pair, count and record position

    std::unordered_map<value_t, cnt_t> value_count;

    for (auto it = data.begin(); it != data.end(); it++) {
        value_count[it->value]++;
    }
    if (verbose) {
        std::cout << "Count done" << std::endl;
    }

    for (list_t::iterator it1 = data.begin(),
                          it2 = std::next(data.begin());
         it2 != data.end();
         it1++, it2++) {
        if (value_count[it1->value] < threshold || value_count[it2->value] < threshold) {
            continue;
        }
        pair_value_t pair = std::make_pair(it1->value, it2->value);
        pair_count[pair]++;
        pair_header[pair].insert({it1->pos, it1});
    }
    if (verbose) {
        std::cout << "One pass done" << std::endl;
    }

    heap_t heap;
    pair_map_t<heap_t::handle_type> pair_handle;

    for (auto [pair, count] : pair_count) {
        if (count >= threshold) {
            pair_handle[pair] = heap.push(pair_count_item_t({pair, count}));
        } else {
            pair_header[pair].clear();
        }
    }

    auto lst_log_time = std::chrono::steady_clock::now();

    while (true) {
        if (heap.empty()) {
            break;
        }
        auto [cur_pair, cur_count] = heap.top();
        heap.pop();
        if (cur_count < threshold) {
            break;
        }

        // generate new pair rule
        pair_rule[cur_pair] = max_pair_value++;

        if (verbose) {
            auto cur_time = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - lst_log_time).count() > LOG_INTERVAL) {
                lst_log_time = cur_time;
                std::cout << std::format("memory usage: {:.2f} KB\tdisk usage: {:.2f} MB ({:.2f}%)\tcurrent_pair_count: {}",
                                         static_cast<double>(pair_rule.size()) * (3 * sizeof(value_t)) / 1024,
                                         static_cast<double>(data.size() - n) * sizeof(value_t) / 1024 / 1024,
                                         100 * static_cast<double>(data.size() - n) / n / dim,
                                         cur_count)
                          << std::endl;
            }
        }

#ifndef NDEBUG
        std::cout << cur_pair.first << " " << cur_pair.second << " --> " << pair_rule[cur_pair] << std::endl;
#endif

        std::unordered_set<pair_pos_t, pair_pos_t_hash> invalid_pos;
        // loop1: update old pair count, erase old pair position
        for (auto it_wrapper : pair_header[cur_pair]) {
            auto it = it_wrapper.it;
            if (invalid_pos.find(it_wrapper.pos) != invalid_pos.end()) {
                continue;
            }

#ifndef NDEBUG
            std::cout << std::format("replace {},{} in {} [{}, {}) with {}",
                                     it->pos.first, it->pos.second,
                                     it->pos.first / (dim + 1), it->pos.first % (dim + 1), it->pos.second % (dim + 1),
                                     pair_rule[cur_pair])
                      << std::endl;
#endif

            if (it != data.begin()) {
                auto it0 = std::prev(it);  // it0 it it2
                pair_value_t old_pair = std::make_pair(it0->value, it->value);
                if (cur_pair != old_pair) {
                    if (pair_handle.find(old_pair) != pair_handle.end()) {
                        heap.decrease(pair_handle[old_pair], pair_count_item_t::decreased_item(*pair_handle[old_pair]));
                        pair_header[old_pair].erase({it0->pos, it0});
                    }
                } else {                           // pair has been pop
                    invalid_pos.insert(it0->pos);  // instead of erase, mark it as invalid to avoid iterator invalidation
                }
#ifndef NDEBUG
                std::cout << "erase " << old_pair.first << " " << old_pair.second << std::endl;
#endif
            }

            if (std::next(it) != data.end()) {
                auto it2 = std::next(it);  // it it2 it3
                auto it3 = std::next(it2);
                pair_value_t old_pair = std::make_pair(it2->value, it3->value);
                if (cur_pair != old_pair) {
                    if (pair_handle.find(old_pair) != pair_handle.end()) {
                        heap.decrease(pair_handle[old_pair], pair_count_item_t::decreased_item(*pair_handle[old_pair]));
                        pair_header[old_pair].erase({it2->pos, it2});
                    }
                } else {                           // pair has been pop
                    invalid_pos.insert(it2->pos);  // instead of erase, mark it as invalid to avoid iterator invalidation
                }
#ifndef NDEBUG
                std::cout << "erase " << old_pair.first << " " << old_pair.second << std::endl;
#endif
            }
        }
#ifndef NDEBUG
        for (auto invalid : invalid_pos) {
            std::cout << "invalid " << invalid.first << " " << invalid.second << std::endl;
        }
#endif

        // loop2: transform old pair to new pair
        for (auto it_wrapper : pair_header[cur_pair]) {
            auto it = it_wrapper.it;
            if (invalid_pos.find(it_wrapper.pos) != invalid_pos.end()) {
                continue;
            }
            auto it2 = std::next(it);
            it->value = pair_rule[cur_pair];
            it->pos.second = it2->pos.second;
            data.erase(it2);
        }

        pair_map_t<cnt_t> new_pair_count;
        // loop3: update new pair count, insert new pair position
        for (auto it_wrapper : pair_header[cur_pair]) {
            auto it = it_wrapper.it;
            if (invalid_pos.find(it_wrapper.pos) != invalid_pos.end()) {
                continue;
            }

            if (it != data.begin()) {
                auto it0 = std::prev(it);  // it0 it
                pair_value_t new_pair = std::make_pair(it0->value, it->value);
                new_pair_count[new_pair]++;
                pair_header[new_pair].insert({it0->pos, it0});
#ifndef NDEBUG
                // std::cout << "insert " << new_pair.first << " " << new_pair.second << std::endl;
#endif
            }

            if (it != data.end()) {
                auto it2 = std::next(it);  // it it2
                pair_value_t new_pair = std::make_pair(it->value, it2->value);
                new_pair_count[new_pair]++;
                pair_header[new_pair].insert({it->pos, it});

#ifndef NDEBUG
                // std::cout << "insert " << new_pair.first << " " << new_pair.second << std::endl;
#endif
            }
        }

        for (auto [pair, count] : new_pair_count) {
            pair_handle[pair] = heap.push(pair_count_item_t({pair, count}));
        }

        // optional clear
        pair_header[cur_pair].clear();
        pair_handle.erase(cur_pair);
    }

    std::vector<value_t> result_elist;
    std::vector<cnt_t> result_vlist = {0};
    for (auto it : data) {
        if (it.value <= max_value || it.value > max_sep) {
            result_elist.push_back(it.value);
        } else {
            result_vlist.push_back(result_elist.size());
        }
    }
    if (verbose) {
        std::cout << std::format("memory usage: {:.2f} KB\tdisk usage: {:.2f} MB ({:.2f}%)",
                                 static_cast<double>(pair_rule.size()) * (3 * sizeof(value_t)) / 1024,
                                 static_cast<double>(result_elist.size()) * sizeof(value_t) / 1024 / 1024,
                                 100 * static_cast<double>(result_elist.size()) / n / dim)
                  << std::endl;
    }
    return std::make_tuple(pair_rule, result_vlist, result_elist);
}
}  // namespace repair_compress
