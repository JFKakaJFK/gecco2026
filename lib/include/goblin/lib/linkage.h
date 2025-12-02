#pragma once
#ifndef _GOBLIN_LIB_LINKAGE_H
#define _GOBLIN_LIB_LINKAGE_H

#include <algorithm>
#include <print>
#include <string_view>
#include <vector>
#include <optional>
#include <stdexcept>

#include "goblin/lib/assert.h"
#include "goblin/lib/instance.h"
#include "goblin/lib/solution.h"
#include "goblin/lib/types.h"

namespace goblin {
enum class VariableSet : u8 { Discrete = 0b01, Continuous = 0b10, Mixed = 0b11 };

inline constexpr bool operator&(VariableSet lhs, VariableSet rhs) noexcept {
  return static_cast<bool>(static_cast<u8>(lhs) & static_cast<u8>(rhs));
};

// TODO this method does way too much, but where else to put all of the
// modifications of the frequency counts for the entropy?
// - problem shouldn't have to know about the intron related entropy
// modifications
// - continuous stuff interacts with the introns...
// => current tradeoff is having the problem provide info about discrete
// values that actually correspond to a continuous value...
inline Mat<CType> estimate_entropy2(const InstanceBase& problem,
                                    const SolutionSetBase& solutions,
                                    const std::span<const usize> indices,
                                    const std::span<const usize> subset,
                                    const std::string& intron_strategy,
                                    bool merge_continuous,
                                    std::optional<usize> num_continuous_bins) {
  __goblin_runtime_assert(subset.size() > 0);

  auto& domain_sizes = problem.discrete_domain_sizes();

  usize max_value_count = 0;  // number of different discrete values to consider at most

  bool intron_aware = intron_strategy != "none";
  bool is_all_active = intron_strategy == "all_active";
  bool is_any_active = intron_strategy == "any_active";
  // bool is_mark_only = intron_strategy == "mark_only";

  // add one more discrete value for introns
  const usize intron_idx = max_value_count;
  if (intron_aware) {
    max_value_count++;
  }

  // add more discrete values for continous values considered
  const usize first_continuous_idx = max_value_count;
  std::vector<CType> continuous_bin_values_sorted;
  auto insertion_idx = [&](CType v) -> usize {
    auto lower = std::lower_bound(continuous_bin_values_sorted.begin(), continuous_bin_values_sorted.end(), v);
    return lower != continuous_bin_values_sorted.end() ? std::distance(continuous_bin_values_sorted.begin(), lower)
                                                       : continuous_bin_values_sorted.size();
  };
  if (merge_continuous) {
    __goblin_runtime_assert(!num_continuous_bins.has_value());  // cannot merge and differentiate
                                                                // continuous values...
    max_value_count++;
    // would be `is-const` in the terms of
    // https://arxiv.org/pdf/1904.02050#section.6
    // - all constants are considered to be the same value
    // => ask `should this be continuous?` instead of
    //    `what should the value be if it is continuous?`
  } else if (problem.num_continuous() > 0 && num_continuous_bins.has_value() && num_continuous_bins.value() > 0) {
    continuous_bin_values_sorted.reserve(num_continuous_bins.value());
    // https://arxiv.org/pdf/1904.02050#section.6

    usize i = 0, j = 0, _i, _j;
    while (i < indices.size() && continuous_bin_values_sorted.size() < num_continuous_bins.value()) {
      _i = i;
      _j = j;
      j++;
      if (j >= subset.size()) {
        j = 0;
        i++;
      }

      // not actively used
      if (intron_aware && !solutions[indices[_i]].discrete_active()(subset[_j])) {
        continue;
      }
      auto v = problem.as_continuous(solutions[indices[_i]], subset[_j]);
      // not a continuous value
      if (!v.has_value()) {
        continue;
      }

      CType new_active_value = v.value();

      usize idx = insertion_idx(new_active_value);

      // already seen before
      if (idx < continuous_bin_values_sorted.size() && continuous_bin_values_sorted[idx] == new_active_value) {
        continue;
      }

      // TODO insertion sort probably is faster?
      // extend, shift values + insert at sorted position
      continuous_bin_values_sorted.push_back(new_active_value);
      std::shift_right(continuous_bin_values_sorted.begin() + idx, continuous_bin_values_sorted.end(), 1);
      continuous_bin_values_sorted[idx] = new_active_value;
    }

    // IDEA: merge values that are "very close" here - no need to have multiple
    // bins for the same semantic value

    max_value_count += continuous_bin_values_sorted.size();
  }

  usize offset = max_value_count;
  max_value_count += domain_sizes(subset).maxCoeff();

  Mat<usize> counts(max_value_count, max_value_count);

  auto entropy = [&](usize lhs, usize rhs) -> CType {
    bool is_univariate = lhs == rhs;

    // TODO update illustration, the order is reversed now
    // counts is used as illustrated here:
    // https://excalidraw.com/#json=-hFhq_364YL6anYCqyyAk,c5_JKnQ7yLwB4oTf3SZUtA
    if (is_univariate) {
      counts(Eigen::seqN(0, offset + domain_sizes(lhs)), 0).setZero();
    } else {
      counts(Eigen::seqN(0, offset + domain_sizes(lhs)), Eigen::seqN(0, offset + domain_sizes(rhs))).setZero();
    }
    // 0
    for (usize i : indices) {
      usize lhs_idx = offset + solutions[i].discrete_values()(lhs);

      if (intron_aware && !solutions[i].discrete_active()(lhs)) {
        lhs_idx = intron_idx;
      } else if (merge_continuous || num_continuous_bins.has_value()) {
        auto v = problem.as_continuous(solutions[i], lhs);
        if (v.has_value()) {
          if (merge_continuous) {
            lhs_idx = first_continuous_idx;
          } else {
            // index of closest bin
            lhs_idx =
                first_continuous_idx + std::min(insertion_idx(v.value()), continuous_bin_values_sorted.size() - 1);
          }
        }
      }

      usize rhs_idx = offset + solutions[i].discrete_values()(rhs);
      if (is_univariate) {
        rhs_idx = 0;  // use only the first column if univariate
      } else if (intron_aware && !solutions[i].discrete_active()(rhs)) {
        rhs_idx = intron_idx;
      } else if (merge_continuous || num_continuous_bins.has_value()) {
        auto v = problem.as_continuous(solutions[i], rhs);
        if (v.has_value()) {
          if (merge_continuous) {
            rhs_idx = first_continuous_idx;
          } else {
            // index of closest bin
            rhs_idx =
                first_continuous_idx + std::min(insertion_idx(v.value()), continuous_bin_values_sorted.size() - 1);
          }
        }
      }

      // 1

      counts(lhs_idx, rhs_idx) += 1;
    }

    // TODO for both the following total count + entropy accumulation, there is
    // no need anymore to have the if branches nested in the loops instead 1.
    // skip increasing the count for introns not considered (// 1) and 2. keep
    // track of the total (//0) Then the final loop can just first elementwise
    // compute the contributions and then sum them up... -> less branching +
    // Eigen vectorization

    /// - none: Ignore introns
    /// - any_active: Introns are marked as such to remove random noise and only
    /// variable pairs with at least one active variable are considered
    /// - all_active: Only variable pairs where both variables are active are
    /// considered
    /// - mark_only: Introns are marked as such to reduce noise, but pairs
    /// consisting of only introns are still considered
    CType total_count = static_cast<CType>(indices.size());
    if (is_any_active) {
      total_count -= static_cast<CType>(is_univariate ? counts(intron_idx, 0) : counts(intron_idx, intron_idx));
    } else if (is_all_active) {
      if (!is_univariate) {
        total_count -= counts(intron_idx, Eigen::placeholders::all).sum();
        total_count -= counts(Eigen::placeholders::all, intron_idx).sum();
        // The "both are introns" count was removed twice, so we need to add it back once
        total_count += counts(intron_idx, intron_idx);
      } else {
        total_count -= counts(intron_idx, 0);
      }
    }

    // at least two different values are needed or entropy will be 0.0 anyway
    if (total_count < CType(2.0)) {
      return 0.0;
    }

    CType e = CType(0.0), p;
    usize r_max = is_univariate ? 1 : (offset + domain_sizes(rhs));
    for (usize r = 0; r < r_max; r++) {
      for (usize l = 0; l < offset + domain_sizes(lhs); l++) {
        if (counts(l, r) > 0) {
          if (intron_aware &&
              (l == intron_idx || (r == intron_idx && !is_univariate))  // we have an intron to account for
              && (is_all_active || (is_any_active && l == r))) {
            continue;
          }

          p = counts(l, r) / total_count;
          e += -p * std::log2(p);
        }
      }
    }

    return std::max(e, CType(0.0));
  };

  Mat<CType> H(subset.size(), subset.size());
  for (usize i = 0; i < subset.size(); i++) {
    H(i, i) = entropy(subset[i], subset[i]);
    for (usize j = 0; j < i; j++) {
      H(i, j) = entropy(subset[i], subset[j]);
      H(j, i) = H(i, j);
    }
  }

  return H;
};

// TODO since the enum and implementation are only used in the wrapped function, move all of this code into a .cpp file
// (fine, since the template is only used here)
enum class DiscreteIntronStrategy : u8 { None, AnyActive, AllActive, MarkOnly, WeightedAnyActive };
// TODO potentially try first allocating a matrix with the actual values and then doing a branch free reduction based on
// that... O(indices * subset.size())
template <DiscreteIntronStrategy intron_strategy>
inline Mat<CType> estimate_entropy_impl(const InstanceBase& problem,
                                        const SolutionSetBase& solutions,
                                        const std::span<const usize> indices,
                                        const std::span<const usize> subset,
                                        bool merge_continuous,
                                        std::optional<usize> num_continuous_bins) {
  __goblin_runtime_assert(subset.size() > 0);

  auto& domain_sizes = problem.discrete_domain_sizes();

  usize max_value_count = 0;  // number of different discrete values to consider at most

  // add one more discrete value for introns
  const usize intron_idx = max_value_count;
  if constexpr (intron_strategy != DiscreteIntronStrategy::None) {
    max_value_count++;
  }

  // add more discrete values for continous values considered
  const usize first_continuous_idx = max_value_count;
  std::vector<CType> continuous_bin_values_sorted;
  auto insertion_idx = [&](CType v) -> usize {
    auto lower = std::lower_bound(continuous_bin_values_sorted.begin(), continuous_bin_values_sorted.end(), v);
    return lower != continuous_bin_values_sorted.end() ? std::distance(continuous_bin_values_sorted.begin(), lower)
                                                       : continuous_bin_values_sorted.size();
  };

  bool specialize_continuous = false;
  if (problem.num_continuous() > 0) {
    if (merge_continuous) {
      __goblin_runtime_assert(!num_continuous_bins.has_value());  // cannot merge and differentiate
                                                                  // continuous values...
      max_value_count++;
      specialize_continuous = true;
      // would be `is-const` in the terms of
      // https://arxiv.org/pdf/1904.02050#section.6
      // - all constants are considered to be the same value
      // => ask `should this be continuous?` instead of
      //    `what should the value be if it is continuous?`
    } else if (num_continuous_bins.has_value() && num_continuous_bins.value() > 0) {
      specialize_continuous = true;
      continuous_bin_values_sorted.reserve(num_continuous_bins.value());
      // https://arxiv.org/pdf/1904.02050#section.6

      usize i = 0, j = 0, _i, _j;
      while (i < indices.size() && continuous_bin_values_sorted.size() < num_continuous_bins.value()) {
        _i = i;
        _j = j;
        j++;
        if (j >= subset.size()) {
          j = 0;
          i++;
        }

        // not actively used
        if constexpr (intron_strategy != DiscreteIntronStrategy::None) {
          if (!solutions[indices[_i]].discrete_active()(subset[_j])) {
            continue;
          }
        }

        auto v = problem.as_continuous(solutions[indices[_i]], subset[_j]);
        // not a continuous value
        if (!v.has_value()) {
          continue;
        }

        CType new_active_value = v.value();

        usize idx = insertion_idx(new_active_value);

        // already seen before
        if (idx < continuous_bin_values_sorted.size() && continuous_bin_values_sorted[idx] == new_active_value) {
          continue;
        }

        // TODO insertion sort probably is faster?
        // extend, shift values + insert at sorted position
        continuous_bin_values_sorted.push_back(new_active_value);
        std::shift_right(continuous_bin_values_sorted.begin() + idx, continuous_bin_values_sorted.end(), 1);
        continuous_bin_values_sorted[idx] = new_active_value;
      }

      // IDEA: merge values that are "very close" here - no need to have multiple
      // bins for the same semantic value

      max_value_count += continuous_bin_values_sorted.size();
    }
  }

  usize offset = max_value_count;
  max_value_count += domain_sizes(subset).maxCoeff();

  // loop over all solutions and indices and prepare the value (is read multiple times, and lots of expensive branching
  // inside of the loop...)
  std::optional<CType> v;
  Eigen::Matrix<usize, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mapped_values(indices.size(), subset.size());
  for (usize i = 0; i < indices.size(); i++) {
    for (usize j = 0, s_j; j < subset.size(); j++) {
      s_j = subset[j];
      if constexpr (intron_strategy != DiscreteIntronStrategy::None) {
        if (!solutions[indices[i]].discrete_active()(s_j)) {
          mapped_values(i, j) = intron_idx;
          continue;
        }
      }

      v = specialize_continuous ? problem.as_continuous(solutions[indices[i]], s_j) : std::nullopt;
      if (v.has_value()) {
        if (merge_continuous) {
          mapped_values(i, j) = first_continuous_idx;
        } else {
          // index of closest bin
          mapped_values(i, j) =
              first_continuous_idx + std::min(insertion_idx(v.value()), continuous_bin_values_sorted.size() - 1);
        }
      } else {
        mapped_values(i, j) = offset + solutions[indices[i]].discrete_values()(s_j);
      }
    }
  }

  Mat<usize> counts(max_value_count, max_value_count);
  Mat<CType> H(subset.size(), subset.size());
  for (usize lhs = 0; lhs < subset.size(); lhs++) {
    usize total = 0;
    counts.col(0).setZero();
    for (usize i = 0; i < indices.size(); i++) {
      if constexpr (intron_strategy == DiscreteIntronStrategy::AnyActive ||
                    intron_strategy == DiscreteIntronStrategy::WeightedAnyActive ||
                    intron_strategy == DiscreteIntronStrategy::AllActive) {
        if (mapped_values(i, lhs) == intron_idx) {
          continue;
        }
      }
      total++;
      counts(mapped_values(i, lhs), 0)++;
    }

    // turn (adjusted) frequency counts into entropy
    if (total > 1) {
      CType e = 0.0, p;
      for (usize i = 0; i < max_value_count; i++) {
        if (counts(i, 0) > 0) {
          p = static_cast<CType>(counts(i, 0)) / static_cast<CType>(total);
          e += -p * std::log2(p);
        }
      }
      if constexpr (intron_strategy == DiscreteIntronStrategy::WeightedAnyActive) {
        e *= static_cast<CType>(total) / static_cast<CType>(indices.size());
      }
      H(lhs, lhs) = e > 0.0 ? e : 0.0;
    } else {
      H(lhs, lhs) = 0.0;
    }

    for (usize rhs = 0; rhs < lhs; rhs++) {
      total = 0;
      counts.setZero();

      for (usize i = 0; i < indices.size(); i++) {
        if constexpr (intron_strategy == DiscreteIntronStrategy::AnyActive ||
                      intron_strategy == DiscreteIntronStrategy::WeightedAnyActive) {
          if (mapped_values(i, lhs) == intron_idx && mapped_values(i, rhs) == intron_idx) {
            continue;
          }
        }
        if constexpr (intron_strategy == DiscreteIntronStrategy::AllActive) {
          if (mapped_values(i, lhs) == intron_idx || mapped_values(i, rhs) == intron_idx) {
            continue;
          }
        }
        total++;
        counts(mapped_values(i, lhs), mapped_values(i, rhs))++;
      }

      // turn (adjusted) frequency counts into entropy
      if (total > 1) {
        CType e = 0.0, p;
        for (isize i = 0; i < counts.size(); i++) {
          if (counts(i) > 0) {
            p = static_cast<CType>(counts(i)) / static_cast<CType>(total);
            e += -p * std::log2(p);
          }
        }
        if constexpr (intron_strategy == DiscreteIntronStrategy::WeightedAnyActive) {
          e *= static_cast<CType>(total) / static_cast<CType>(indices.size());
        }
        H(lhs, rhs) = e > 0.0 ? e : 0.0;
      } else {
        H(lhs, rhs) = 0.0;
      }
      H(rhs, lhs) = H(lhs, rhs);
    }
  }

  return H;
};
inline Mat<CType> estimate_entropy(const InstanceBase& problem,
                                   const SolutionSetBase& solutions,
                                   const std::span<const usize> indices,
                                   const std::span<const usize> subset,
                                   const std::string& intron_strategy,
                                   bool merge_continuous,
                                   std::optional<usize> num_continuous_bins) {
  if (intron_strategy == "any_active") {
    return estimate_entropy_impl<DiscreteIntronStrategy::AnyActive>(problem, solutions, indices, subset,
                                                                    merge_continuous, num_continuous_bins);
  } else if (intron_strategy == "weighted_any_active") {
    return estimate_entropy_impl<DiscreteIntronStrategy::WeightedAnyActive>(problem, solutions, indices, subset,
                                                                            merge_continuous, num_continuous_bins);
  } else if (intron_strategy == "none") {
    return estimate_entropy_impl<DiscreteIntronStrategy::None>(problem, solutions, indices, subset, merge_continuous,
                                                               num_continuous_bins);
  } else if (intron_strategy == "mark_only") {
    return estimate_entropy_impl<DiscreteIntronStrategy::MarkOnly>(problem, solutions, indices, subset,
                                                                   merge_continuous, num_continuous_bins);
  } else if (intron_strategy == "all_active") {
    return estimate_entropy_impl<DiscreteIntronStrategy::AllActive>(problem, solutions, indices, subset,
                                                                    merge_continuous, num_continuous_bins);
  } else {
    throw std::runtime_error("Unknown intron strategy.");
  }
}
};  // namespace goblin

#endif /* _GOBLIN_LIB_LINKAGE_H */
