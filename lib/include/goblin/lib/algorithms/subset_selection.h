#pragma once
#ifndef _GOBLIN_LIB_SUBSET_SELECTION_H
#define _GOBLIN_LIB_SUBSET_SELECTION_H

#include <algorithm>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "goblin/lib/assert.h"
#include "goblin/lib/types.h"

namespace goblin {

template <typename DST>
inline std::tuple<std::vector<usize>, std::vector<usize>>
greedy_scattered_subset_selection(DST&& distance, usize pool_size, usize target_size, std::vector<usize>&& selection) {
    __goblin_runtime_assert(target_size <= pool_size);

    std::vector<usize> remaining(pool_size);
    std::iota(remaining.begin(), remaining.end(), 0);

    std::sort(selection.begin(), selection.end(), std::greater{});
    for (auto i : selection) {
        __goblin_runtime_assert(i < pool_size);
        std::swap(remaining[i], remaining.back());
        remaining.pop_back();
    }

    using dist_t = std::remove_cvref_t<decltype(std::declval<DST>()(0, 0))>;
    std::vector<dist_t> dist_to_selection(remaining.size(), std::numeric_limits<dist_t>::max());
    for (usize s = 0; s < selection.size() - 1; s++) {
        for (usize i = 0; i < remaining.size(); i++) {
            dist_to_selection[i] = std::min(dist_to_selection[i], distance(selection[s], remaining[i]));
        }
    }

    while (selection.size() < target_size) {
        // update distance to selection and get furthest
        usize furthest = 0;
        for (usize i = 0; i < remaining.size(); i++) {
            dist_to_selection[i] = std::min(dist_to_selection[i], distance(selection.back(), remaining[i]));
            if (dist_to_selection[i] > dist_to_selection[furthest]) {
                furthest = i;
            }
        }

        // add furthest element to the selection
        std::swap(dist_to_selection[furthest], dist_to_selection.back());
        std::swap(remaining[furthest], remaining.back());
        selection.push_back(remaining.back());
        remaining.pop_back();
    }

    return std::make_tuple(selection, remaining);
};

/// Given a list of remaining options and an initial solution,
/// the furthest remaining solution is repeatedly selected until the target size
/// is reached
template <typename DST>
inline std::tuple<std::vector<usize>, std::vector<usize>> greedy_scattered_subset_selection(DST&& distance,
                                                                                            usize pool_size,
                                                                                            usize target_size,
                                                                                            usize initial) {
    return greedy_scattered_subset_selection(distance, pool_size, target_size, std::move(std::vector<usize>{initial}));
};

};  // namespace goblin

#endif /* _GOBLIN_LIB_SUBSET_SELECTION_H */
