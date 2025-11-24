#pragma once
#ifndef _GOBLIN_LIB_LINKAGE_MODEL_H
#define _GOBLIN_LIB_LINKAGE_MODEL_H

#include <functional>
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
  virtual void init(Rng& rng, InstanceBase& problem, SolutionSetBase& solutions, VariableSet variables) = 0;
  virtual FOS subsets(Rng& rng,
                      InstanceBase& problem,
                      SolutionSetBase& solutions,
                      const std::span<const usize> indices,
                      std::optional<std::reference_wrapper<const Mat<CType>>> covariance) const = 0;

  virtual bool is_static() const { return false; };

  virtual std::unique_ptr<LinkageModelBase> clone() const = 0;

  virtual ~LinkageModelBase() {};
};

class UnivariateFOS final : public LinkageModelBase {
 public:
  UnivariateFOS(std::optional<Subset> subset = std::nullopt) : subset(subset.value_or(Subset{})) {}

  std::unique_ptr<LinkageModelBase> clone() const override final { return std::make_unique<UnivariateFOS>(*this); }

  void init(Rng& rng, InstanceBase& problem, SolutionSetBase& solutions, VariableSet variables) override final {
    if (subset.empty()) {
      if (variables & VariableSet::Discrete) {
        subset.discrete.reserve(problem.num_discrete());
        for (usize i = 0; i < problem.num_discrete(); i++) {
          subset.discrete.push_back(i);
        }
      }
      if (variables & VariableSet::Continuous) {
        subset.continuous.reserve(problem.num_continuous());
        for (usize i = 0; i < problem.num_continuous(); i++) {
          subset.continuous.push_back(i);
        }
      }
    }
  }

  FOS subsets(Rng& rng,
              InstanceBase& problem,
              SolutionSetBase& solutions,
              const std::span<const usize> indices,
              std::optional<std::reference_wrapper<const Mat<CType>>> covariance = std::nullopt) const override final {
    FOS fos;
    fos.resize(subset.size());
    {
      usize i = 0;
      for (usize j : subset.discrete) {
        fos[i++].discrete.push_back(j);
      }
      for (usize j : subset.continuous) {
        fos[i++].continuous.push_back(j);
      }
    }
    return fos;
  };

  bool is_static() const override final { return true; };

 private:
  Subset subset;
};

class FullFOS final : public LinkageModelBase {
 public:
  FullFOS(std::optional<Subset> subset = std::nullopt) : subset(subset.value_or(Subset{})) {}

  std::unique_ptr<LinkageModelBase> clone() const override final { return std::make_unique<FullFOS>(*this); }

  void init(Rng& rng, InstanceBase& problem, SolutionSetBase& solutions, VariableSet variables) override final {
    if (subset.empty()) {
      if (variables & VariableSet::Discrete) {
        subset.discrete.reserve(problem.num_discrete());
        for (usize i = 0; i < problem.num_discrete(); i++) {
          subset.discrete.push_back(i);
        }
      }
      if (variables & VariableSet::Continuous) {
        subset.continuous.reserve(problem.num_continuous());
        for (usize i = 0; i < problem.num_continuous(); i++) {
          subset.continuous.push_back(i);
        }
      }
    }
  }

  FOS subsets(Rng& rng,
              InstanceBase& problem,
              SolutionSetBase& solutions,
              const std::span<const usize> indices,
              std::optional<std::reference_wrapper<const Mat<CType>>> covariance = std::nullopt) const override final {
    __goblin_runtime_assert(subset.continuous.size() > 0 ||
                            subset.discrete.size() <
                                problem.num_discrete());  // Full discrete subset probably is a mistake, that just
                                                          // copies solutions instead of performing search...
    return {subset};
  };

  bool is_static() const override final { return true; };

 private:
  Subset subset;
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
                 bool normalize_initial_linkage_bias = false,
                 std::optional<Subset> subset = std::nullopt)
      : metric(metric),
        intron_strategy(intron_strategy),
        merge_continuous(merge_continuous),
        num_continuous_bins(num_continuous_bins),
        filter_parent_threshold(filter_parent_threshold),
        filter_children_threshold(filter_children_threshold),
        filter_root(filter_root),
        max_subset_size(max_subset_size),
        normalize_initial_linkage_bias(normalize_initial_linkage_bias),
        subset(subset.value_or(Subset{})) {};

  std::unique_ptr<LinkageModelBase> clone() const override final { return std::make_unique<LinkageTreeFOS>(*this); }

  void init(Rng& rng, InstanceBase& problem, SolutionSetBase& solutions, VariableSet variables) override final {
    if (subset.empty()) {
      if (variables & VariableSet::Discrete) {
        subset.discrete.reserve(problem.num_discrete());
        for (usize i = 0; i < problem.num_discrete(); i++) {
          subset.discrete.push_back(i);
        }
      }
      if (variables & VariableSet::Continuous) {
        subset.continuous.reserve(problem.num_continuous());
        for (usize i = 0; i < problem.num_continuous(); i++) {
          subset.continuous.push_back(i);
        }
      }
    }

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
      auto H = estimate_entropy(problem, solutions, indices, subset.discrete, intron_strategy, merge_continuous,
                                num_continuous_bins);

      usize l = subset.discrete.size();
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
              std::optional<std::reference_wrapper<const Mat<CType>>> covariance = std::nullopt) const override final {
    bool is_discrete = subset.discrete.size() > 0;
    bool is_continuous = subset.continuous.size() > 0;

    // 1. get similarity matrix based on the measure...
    isize l = subset.size();
    Mat<CType> similarity;
    if (is_discrete && is_continuous) {
      similarity.resize(l, l);
      // No mixed subset learning (yet?)
      // https://homepages.cwi.nl/~bosman/publications/2016_learningandexploiting.pdf
      __goblin_runtime_assert(false);
    } else if (is_discrete) {
      similarity = estimate_entropy(problem, solutions, indices, subset.discrete, intron_strategy, merge_continuous,
                                    num_continuous_bins);
      if (normalize_initial_linkage_bias && initial_bias_adjustments.size() > 0) {
        similarity.array() *= initial_bias_adjustments.array();
      }
      entropy2similarity(similarity);
    } else if (is_continuous) {
      similarity.resize(l, l);
      __goblin_runtime_assert(covariance.has_value());

      Mat<CType> cov = covariance.value();
      __goblin_runtime_assert(cov.rows() == l);
      __goblin_runtime_assert(cov.cols() == l);
      for (isize i = 0; i < l; i++) {
        for (isize j = 0; j < i; j++) {
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
      for (usize j : subset.discrete) {
        fos[i++].discrete.push_back(j);
      }
      for (usize j : subset.continuous) {
        fos[i++].continuous.push_back(j);
      }
    }

    // and add all of the merges performed during clustering as subsets
    usize num_merges = merges.size();
    if (num_merges > 0 && filter_root.value_or(is_discrete && !is_continuous)) {
      num_merges--;
    }
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
      for (isize i = 0; i < H.rows(); i++) {
        for (isize j = 0; j < i; j++) {
          MI(i, j) = H(i, i) + H(j, j) - H(i, j);
          MI(j, i) = MI(i, j);
        }
      }
    } else if (metric == "nmi") {
      auto& NMI = H;
      for (isize i = 0; i < H.rows(); i++) {
        for (isize j = 0; j < i; j++) {
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
  Subset subset;

  Mat<CType> initial_bias_adjustments;
};

class CombinedFOS final : public LinkageModelBase {
 public:
  CombinedFOS(const std::vector<std::unique_ptr<LinkageModelBase>>& linkage_models) {
    models.reserve(linkage_models.size());
    for (usize i = 0; i < linkage_models.size(); i++) {
      models.push_back(linkage_models[i]->clone());
    }
  }

  void add_model(const LinkageModelBase& model) { models.push_back(model.clone()); }

  // Explicitly disallow copies to tell the Python binding generation that
  // a vector of unique pointers cannot be copied
  // (the other option would be to explicitly define a version that clones the data)
  CombinedFOS(const CombinedFOS&) = delete;
  CombinedFOS& operator=(const CombinedFOS&) = delete;

  // But moving is allowed
  CombinedFOS(CombinedFOS&&) = default;
  CombinedFOS& operator=(CombinedFOS&&) = default;

  std::unique_ptr<LinkageModelBase> clone() const override final { return std::make_unique<CombinedFOS>(models); }

  void init(Rng& rng, InstanceBase& problem, SolutionSetBase& solutions, VariableSet variables) override final {
    for (usize i = 0; i < models.size(); i++) {
      models[i]->init(rng, problem, solutions, variables);
    }
  }

  FOS subsets(Rng& rng,
              InstanceBase& problem,
              SolutionSetBase& solutions,
              const std::span<const usize> indices,
              std::optional<std::reference_wrapper<const Mat<CType>>> covariance = std::nullopt) const override final {
    FOS combined;
    for (usize i = 0; i < models.size(); i++) {
      FOS fos = models[i]->subsets(rng, problem, solutions, indices, covariance);

      // remove any duplicates
      usize j = 0;
      while (j < fos.size()) {
        bool is_new = true;
        for (auto& s : combined) {
          if (fos[j] == s) {
            is_new = false;
            std::swap(fos[j], fos.back());
            fos.pop_back();
            break;
          }
        }

        if (is_new) {
          j++;
        }
      }

      combined.insert(combined.end(), fos.begin(), fos.end());
    }
    return combined;
  };

  bool is_static() const override final {
    for (usize i = 0; i < models.size(); i++) {
      if (!models[i]->is_static()) {
        return false;
      }
    }
    return true;
  };

 private:
  std::vector<std::unique_ptr<LinkageModelBase>> models;
};
};  // namespace goblin

#endif /* _GOBLIN_LIB_LINKAGE_MODEL_H */
