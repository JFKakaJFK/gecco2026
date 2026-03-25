#pragma once
#ifndef _GOBLIN_LIB_ALGORITHMS_MO_H
#define _GOBLIN_LIB_ALGORITHMS_MO_H

#include <algorithm>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <set>
#include <functional>
#include <variant>

#include "goblin/lib/assert.h"
#include "goblin/lib/ordering.h"
#include "goblin/lib/archive.h"

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

/// pointwise hypervolume contribution as per doi.org/10.1109/DOCS63458.2024.10704327
template <typename S2R, typename P2R>
inline CType pHVC_impl(S2R&& signed_solution_to_ref,
                       P2R&& signed_point_to_ref,
                       usize num_solutions,
                       usize num_objectives) {
  if (num_solutions < 1) {
    // contribution = hypervolume of the single point
    CType hv = 1.0;
    for (usize j = 0; j < num_objectives; j++) {
      hv *= signed_point_to_ref(j);
    }
    return hv;
  }

  CType contribution = std::numeric_limits<CType>::infinity();
  for (usize i = 0; i < num_solutions; i++) {
    CType r2p_volume = 1.0, r2worst_volume = 1.0;
    for (usize j = 0; j < num_objectives; j++) {
      CType r2p_j = signed_point_to_ref(j);
      CType r2s_j = signed_solution_to_ref(i, j);
      r2p_volume *= r2p_j;
      // ref_j - max(s_j, p_j) = min(dist_j(r, s), dist_j(r, p))
      r2worst_volume *= std::min(r2p_j, r2s_j);
    }

    contribution = std::min(contribution, r2p_volume - r2worst_volume);
  }
  return contribution;
};

template <typename SolutionSetLike>
inline CType pHVC_dispatch(const SolutionSetLike& solutions,
                           const FitnessBase& fitness,
                           const std::span<const usize> indices,
                           const QualityBase& point,
                           const QualityBase& reference_point) {
  return pHVC_impl(
      [&](const usize idx, const usize objective) {
        CType dist = fitness.distance(solutions[indices[idx]].quality(), reference_point, objective);
        bool solution_is_worse =
            fitness.cmp(solutions[indices[idx]].quality(), reference_point, objective) == Ordering::Worse;
        return solution_is_worse ? -dist : dist;
      },
      [&](const usize objective) {
        CType dist = fitness.distance(point, reference_point, objective);
        bool point_is_worse = fitness.cmp(point, reference_point, objective) == Ordering::Worse;
        return point_is_worse ? -dist : dist;
      },
      indices.size(), fitness.num_objectives());
};

inline CType pHVC(const SolutionSetBase& solutions,
                  const FitnessBase& fitness,
                  const std::span<const usize> indices,
                  const QualityBase& point,
                  const QualityBase& reference_point) {
  return pHVC_dispatch(solutions, fitness, indices, point, reference_point);
};
inline CType pHVC(const ArchiveBase& solutions,
                  const FitnessBase& fitness,
                  const std::span<const usize> indices,
                  const QualityBase& point,
                  const QualityBase& reference_point) {
  return pHVC_dispatch(solutions, fitness, indices, point, reference_point);
};
inline CType pHVC(const SolutionSetBase& solutions,
                  const FitnessBase& fitness,
                  const QualityBase& point,
                  const QualityBase& reference_point) {
  std::vector<usize> indices(solutions.size());
  std::iota(indices.begin(), indices.end(), 0);
  return pHVC_dispatch(solutions, fitness, indices, point, reference_point);
};
inline CType pHVC(const ArchiveBase& solutions,
                  const FitnessBase& fitness,
                  const QualityBase& point,
                  const QualityBase& reference_point) {
  std::vector<usize> indices(solutions.size());
  std::iota(indices.begin(), indices.end(), 0);
  return pHVC_dispatch(solutions, fitness, indices, point, reference_point);
};
inline CType pHVC(const Arr2D<CType> solutions, const Array<CType> point, const Array<CType> reference_point) {
  return pHVC_impl(
      [&](const usize idx, const usize objective) { return reference_point(objective) - solutions(idx, objective); },
      [&](const usize objective) { return reference_point(objective) - point(objective); }, solutions.rows(),
      solutions.cols());
};

};  // namespace goblin

#endif /* _GOBLIN_LIB_ALGORITHMS_MO_H */
