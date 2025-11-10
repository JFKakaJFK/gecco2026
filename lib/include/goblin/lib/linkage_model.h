#ifndef _GOBLIN_LIB_LINKAGE_MODEL_H
#define _GOBLIN_LIB_LINKAGE_MODEL_H

#include <functional>
#pragma once

#include <cstdint>
#include <print>
#include <span>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include "goblin/lib/algorithms/upgma.h"
#include "goblin/lib/assert.h"
#include "goblin/lib/instance.h"
#include "goblin/lib/linkage.h"
#include "goblin/lib/rng.h"
#include "goblin/lib/solution.h"
#include "goblin/lib/types.h"

namespace goblin {
class LinkageModelBase {
 public:
  virtual void init(Rng& rng, InstanceBase& problem, SolutionSetBase& solutions, VariableSet variables) {};
  virtual FOS subsets(Rng& rng,
                      InstanceBase& problem,
                      SolutionSetBase& solutions,
                      const std::span<const usize> indices,
                      VariableSet variables,
                      std::optional<std::reference_wrapper<const Mat<CType>>> covariance) const = 0;

  virtual bool is_static() const { return false; };

  virtual ~LinkageModelBase() {};
};

class UnivariateFOS final : public LinkageModelBase {
 public:
  FOS subsets(Rng& rng,
              InstanceBase& problem,
              SolutionSetBase& solutions,
              const std::span<const usize> indices,
              VariableSet variables,
              std::optional<std::reference_wrapper<const Mat<CType>>> covariance = std::nullopt) const override final {
    FOS fos;
    bool is_discrete = variables & VariableSet::Discrete;
    bool is_continuous = variables & VariableSet::Continuous;
    fos.resize((is_discrete ? problem.num_discrete() : 0) + (is_continuous ? problem.num_continuous() : 0));
    {
      usize i = 0;
      if (is_discrete) {
        for (usize _i = 0; _i < problem.num_discrete(); _i++) {
          fos[i++].discrete.push_back(_i);
        }
      }
      if (is_continuous) {
        for (usize _i = 0; _i < problem.num_continuous(); _i++) {
          fos[i++].continuous.push_back(_i);
        }
      }
    }
    return fos;
  };

  bool is_static() const override final { return true; };
};

class FullFOS final : public LinkageModelBase {
 public:
  FOS subsets(Rng& rng,
              InstanceBase& problem,
              SolutionSetBase& solutions,
              const std::span<const usize> indices,
              VariableSet variables,
              std::optional<std::reference_wrapper<const Mat<CType>>> covariance = std::nullopt) const override final {
    FOS fos;
    bool is_discrete = variables & VariableSet::Discrete;
    bool is_continuous = variables & VariableSet::Continuous;
    __goblin_runtime_assert(!is_discrete || !is_continuous);
    // Full mixed subset is just a copy, not search

    fos.resize(1);
    if (is_discrete) {
      fos[0].discrete.resize(problem.num_discrete());
      std::iota(fos[0].discrete.begin(), fos[0].discrete.end(), 0);
    }
    if (is_continuous) {
      fos[0].continuous.resize(problem.num_continuous());
      std::iota(fos[0].continuous.begin(), fos[0].continuous.end(), 0);
    }
    return fos;
  };

  bool is_static() const override final { return true; };
};

class LinkageTreeFOS final : public LinkageModelBase {
 public:
  LinkageTreeFOS(std::string metric = "mi",  // nmi, pearson_r2

                 /// The intron strategy determines how knowledge about inactive variables
                 /// is used to modify the estimation of linkage when learning the linkage
                 /// tree
                 /// - none: Ignore introns
                 /// - any_active: Introns are marked as such to remove random noise and
                 /// only variable pairs with at least one active variable are considered
                 /// - all_active: Only variable pairs where both variables are active are
                 /// considered
                 /// - mark_only: Introns are marked as such to reduce noise, but pairs
                 /// consisting of only introns are still considered
                 std::string intron_strategy = "none",
                 bool merge_continuous = true,
                 std::optional<usize> num_continuous_bins = std::nullopt,
                 std::optional<CType> filter_parent_threshold = std::nullopt,    // 1e-6
                 std::optional<CType> filter_children_threshold = std::nullopt,  // 1 - 1e-6
                 std::optional<bool> filter_root = std::nullopt,  // default is true if discrete, false if
                                                                  // continuous/mixed
                 std::optional<usize> max_subset_size = std::nullopt,
                 bool normalize_initial_linkage_bias = false)
      : metric(metric),
        intron_strategy(intron_strategy),
        merge_continuous(merge_continuous),
        num_continuous_bins(num_continuous_bins),
        filter_parent_threshold(filter_parent_threshold),
        filter_children_threshold(filter_children_threshold),
        filter_root(filter_root),
        max_subset_size(max_subset_size),
        normalize_initial_linkage_bias(normalize_initial_linkage_bias) {};

  void init(Rng& rng, InstanceBase& problem, SolutionSetBase& solutions, VariableSet variables) override final {
    if (normalize_initial_linkage_bias && variables == VariableSet::Discrete) {
      // __goblin_runtime_assert(intron_strategy == "none");
      // Intron awareness and Marco's linkage normalization are incompatible
      // - conditioning the unbiasing matrix to only consider active variables
      // would lead to potential division by 0 issues (you can't normalize no
      // active values) and learning linkage in an intron aware manner likely
      // reduces the bias already
      // TODO - instead of asserting non-zero entropy, I now silently set the
      // multiplier to 1 for such variables/variable pairs -> could work again

      // Assumption is that the initial values are non-uniformally distributed,
      // suggesting linkage where there likely is none. To fix this, the linkage
      // can be normalized to the identity matrix as per
      // https://arxiv.org/pdf/1904.02050v3#section.5
      std::vector<usize> indices(solutions.size());
      std::iota(indices.begin(), indices.end(), 0);
      auto H = estimate_entropy(problem, solutions, indices, intron_strategy, merge_continuous, num_continuous_bins);

      usize l = problem.num_discrete();
      initial_bias_adjustments.resize(l, l);
      for (usize i = 0; i < l; i++) {
        initial_bias_adjustments(i, i) = H(i, i) > CType(0.0) ? CType(1.0) / H(i, i) : CType(1.0);

        for (usize j = 0; j < i; j++) {
          initial_bias_adjustments(i, j) = H(i, j) > CType(0.0) ? CType(2.0) / H(i, j) : CType(1.0);
          initial_bias_adjustments(j, i) = initial_bias_adjustments(i, j);
        }
      }
    }
  };

  FOS subsets(Rng& rng,
              InstanceBase& problem,
              SolutionSetBase& solutions,
              const std::span<const usize> indices,
              VariableSet variables,
              std::optional<std::reference_wrapper<const Mat<CType>>> covariance = std::nullopt) const override final {
    bool is_discrete = variables & VariableSet::Discrete;
    bool is_continuous = variables & VariableSet::Continuous;

    // 1. get similarity matrix based on the measure...
    usize l;
    Mat<CType> similarity;
    if (variables == VariableSet::Mixed) {
      l = problem.num_discrete() + problem.num_continuous();
      similarity.resize(l, l);
      // No mixed subset learning (yet)
      // https://homepages.cwi.nl/~bosman/publications/2016_learningandexploiting.pdf
      __goblin_runtime_assert(variables != VariableSet::Mixed);
    } else if (is_discrete) {
      l = problem.num_discrete();
      similarity =
          estimate_entropy(problem, solutions, indices, intron_strategy, merge_continuous, num_continuous_bins);
      if (normalize_initial_linkage_bias && initial_bias_adjustments.size() > 0) {
        similarity.array() *= initial_bias_adjustments.array();
      }
      entropy2similarity(similarity);
    } else if (is_continuous) {
      l = problem.num_continuous();
      similarity.resize(l, l);
      __goblin_runtime_assert(covariance.has_value());

      Mat<CType> cov = covariance.value();
      __goblin_runtime_assert(cov.rows() == l);
      __goblin_runtime_assert(cov.cols() == l);
      for (usize i = 0; i < l; i++) {
        for (usize j = 0; j < i; j++) {
          // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
          double r = cov(i, j) / (std::sqrt(cov(i, i)) * std::sqrt(cov(j, j)));
          similarity(i, j) = std::log(std::sqrt(1.0 / (1.0 - r * r)));
          similarity(j, i) = similarity(i, j);
        }
      }
    } else {
      __goblin_runtime_assert(false);  // unknown subset of variables to learn LT for
    }

    // 2. clustering
    auto merges = UPGMA::cluster(rng, similarity);

    // 3. merge subsets...
    FOS fos(2 * l - 1);
    {  // init with the univariate FOS
      usize i = 0;
      if (is_discrete) {
        for (usize _i = 0; _i < problem.num_discrete(); _i++) {
          fos[i++].discrete.push_back(_i);
        }
      }
      if (is_continuous) {
        for (usize _i = 0; _i < problem.num_continuous(); _i++) {
          fos[i++].continuous.push_back(_i);
        }
      }
    }

    // and add all of the merges performed during clustering as subsets
    usize num_merges = filter_root.value_or(variables == VariableSet::Discrete) ? merges.size() - 1 : merges.size();
    for (usize i = 0; i < num_merges; i++) {
      if (max_subset_size.has_value() && merges[i].size > max_subset_size.value()) {
        continue;
      }

      // Clustering happens based on similarity, not distance!
      auto merge_similarity = merges[i].distance;

      // https://homepages.cwi.nl/~bosman/publications/2013_moreconciseand.pdf
      // The linkage is effectively 0, so we can assume that the subsets are
      // independent and thus the merged subset is probably not beneficial
      if (filter_parent_threshold.has_value() &&
          merge_similarity <= filter_parent_threshold.value()) {  // no added benefit is assumed
        continue;
      }

      fos[i + l] = fos[merges[i].left].merge(fos[merges[i].right]);

      // https://homepages.cwi.nl/~bosman/publications/2013_moreconciseand.pdf
      // https://arxiv.org/pdf/2109.05259#subsubsection.3.1.3
      // The larger building block is linked stronger (or at least as strong),
      // so we don't break up building blocks by excluding the smaller subsets
      if (filter_children_threshold.has_value() && merge_similarity >= filter_children_threshold.value()) {
        fos[merges[i].left].clear();
        fos[merges[i].right].clear();
      }
    }

    // remove the empty subsets
    std::erase_if(fos, [](const auto& s) { return s.size() == 0; });

    // std::println("Population:");
    // for (auto i : indices) {
    //   std::println("  {}", problem.format_solution(solutions[i]));
    // }

    // std::println("Similarity:");
    // for (usize i = 0; i < similarity.rows(); i++) {
    //   for (usize j = 0; j < similarity.cols(); j++) {
    //     std::print("{:>4.2f},", similarity(i, j));
    //   }
    //   std::println("");
    // }

    // std::println("FOS:");
    // usize i = 0;
    // for (auto &s : fos) {
    //   std::println("{:>3d}: {{d: {}, c: {}}}", i++, s.discrete,
    //   s.continuous);
    // }

    return fos;
  };

 private:
  void entropy2similarity(Mat<CType>& H) const {
    // entropy -> MI/NMI
    if (metric == "mi") {
      auto& MI = H;
      for (usize i = 0; i < H.rows(); i++) {
        for (usize j = 0; j < i; j++) {
          MI(i, j) = H(i, i) + H(j, j) - H(i, j);
          MI(j, i) = MI(i, j);
        }
      }
    } else if (metric == "nmi") {
      auto& NMI = H;
      for (usize i = 0; i < H.rows(); i++) {
        for (usize j = 0; j < i; j++) {
          NMI(i, j) = H(i, j) > 0 ? (((H(i, i) + H(j, j)) / H(i, j)) - CType(1.0)) : CType(0.0);
          NMI(j, i) = NMI(i, j);
        }
      }
    } else {
      // TODO raise unknown or unsupported for discrete domain
      __goblin_runtime_assert(metric != metric);
    }
  };

  std::string metric;
  std::string intron_strategy;
  bool merge_continuous;
  std::optional<usize> num_continuous_bins;
  std::optional<CType> filter_parent_threshold;
  std::optional<CType> filter_children_threshold;
  std::optional<bool> filter_root;
  std::optional<usize> max_subset_size;
  bool normalize_initial_linkage_bias;

  Mat<CType> initial_bias_adjustments;
};
};  // namespace goblin

#endif /* _GOBLIN_LIB_LINKAGE_MODEL_H */
