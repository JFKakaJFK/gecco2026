#pragma once
#ifndef _GOBLIN_LIB_ALGORITHMS_MO_H
#define _GOBLIN_LIB_ALGORITHMS_MO_H

#include <algorithm>
#include <functional>
#include <limits>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "goblin/lib/archive.h"
#include "goblin/lib/assert.h"
#include "goblin/lib/ordering.h"

namespace goblin {

template <typename D>
inline std::tuple<std::vector<usize>, std::vector<std::set<usize>>> non_dominated_sorting(D&& cmp, usize pool_size) {
    std::vector<std::set<usize>> dominations(pool_size);
    std::vector<usize> domination_count(pool_size, 0);
    std::vector<usize> rank(pool_size);
    std::vector<std::set<usize>> fronts(1);

    for (usize i = 0; i < pool_size; i++) {
        for (usize j = 0; j < i; j++) {
            Ordering o = cmp(i, j);
            if (o == Ordering::Better) {
                dominations[i].insert(j);
                domination_count[j]++;
            } else if (o == Ordering::Worse) {
                dominations[j].insert(i);
                domination_count[i]++;
            }
        }
        if (domination_count[i] == 0) {
            rank[i] = 0;
            fronts.back().insert(i);
        }
    }

    while (!fronts.back().empty()) {
        std::set<usize> front;
        for (usize p : fronts.back()) {
            for (usize q : dominations[p]) {
                domination_count[q]--;
                if (domination_count[q] == 0) {
                    rank[q] = fronts.size();
                    front.insert(q);
                }
            }
        }
        fronts.push_back(front);
    }

    return std::make_tuple(rank, fronts);
};

/// 2D Hypervolume
///
/// Uses an O(n * log(n)) sweeping line approach
/// 1. Points are sorted in one dimension
/// 2. The areas of the resulting rectangles are added together
template <typename SD, typename D2R>
inline CType hypervolume2D_impl(SD&& signed_distance, D2R&& distance_to_reference, usize num_points) {
    if (num_points == 0) {
        return 0.0;
    } else if (num_points == 1) {
        return std::abs(distance_to_reference(0, 0) * distance_to_reference(0, 1));
    }

    std::vector<usize> points(num_points);
    std::iota(points.begin(), points.end(), 0);

    std::sort(points.begin(), points.end(),
              [&](const usize lhs, const usize rhs) { return signed_distance(lhs, rhs, 1) <= 0.0; });

    CType hv = 0.0;
    CType w = distance_to_reference(points[0], 0);
    for (usize i = 1; i < points.size(); i++) {
        hv += signed_distance(points[i], points[i - 1], 1) * w;
        w = std::max(w, distance_to_reference(points[i], 0));
    }
    return hv + distance_to_reference(points.back(), 1) * w;
};

template <typename SolutionSetLike>
inline CType hypervolume2D_dispatch(const SolutionSetLike& solutions,
                                    const FitnessBase& fitness,
                                    const QualityBase& reference_point) {
    return hypervolume2D_impl(
        [&](const usize lhs, const usize rhs, const usize objective) {
            auto dist = fitness.distance(solutions[lhs].quality(), solutions[rhs].quality(), objective);
            auto lhs_better =
                fitness.cmp(solutions[lhs].quality(), solutions[rhs].quality(), objective) == Ordering::Better;
            return lhs_better ? -dist : dist;
        },
        [&](const usize idx, const usize objective) {
            return fitness.distance(solutions[idx].quality(), reference_point, objective);
        },
        solutions.size());
};

inline CType hypervolume2D(const ArchiveBase& solutions,
                           const FitnessBase& fitness,
                           const QualityBase& reference_point) {
    return hypervolume2D_dispatch(solutions, fitness, reference_point);
};
inline CType hypervolume2D(const SolutionSetBase& solutions,
                           const FitnessBase& fitness,
                           const QualityBase& reference_point) {
    return hypervolume2D_dispatch(solutions, fitness, reference_point);
};

inline CType hypervolume2D(const Arr2D<CType> points, const Array<CType> reference_point) {
    return hypervolume2D_impl([&](const usize lhs, const usize rhs,
                                  const usize objective) { return points(lhs, objective) - points(rhs, objective); },
                              [&](const usize idx, const usize objective) {
                                  return std::abs(points(idx, objective) - reference_point(objective));
                              },
                              points.rows());
};

};  // namespace goblin

#endif /* _GOBLIN_LIB_ALGORITHMS_MO_H */
