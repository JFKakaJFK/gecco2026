#pragma once
#ifndef _GOBLIN_LIB_LINKAGE_MODEL_H
#define _GOBLIN_LIB_LINKAGE_MODEL_H

#include <functional>
#include <cstdint>
#include <print>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <optional>
#include <random>

#include <Eigen/Dense>

#include "goblin/lib/algorithms/upgma.h"
#include "goblin/lib/assert.h"
#include "goblin/lib/instance.h"
#include "goblin/lib/rng.h"
#include "goblin/lib/solution.h"
#include "goblin/lib/types.h"

namespace goblin {
enum class VariableSet : u8 { Discrete = 0b01, Continuous = 0b10, Mixed = 0b11 };

inline constexpr bool operator&(VariableSet lhs, VariableSet rhs) noexcept {
  return static_cast<bool>(static_cast<u8>(lhs) & static_cast<u8>(rhs));
};

Mat<CType> estimate_entropy(const InstanceBase& problem,
                            const SolutionSetBase& solutions,
                            const std::span<const usize> indices,
                            const std::span<const usize> subset,
                            const std::string& intron_strategy,
                            bool merge_continuous,
                            std::optional<usize> num_continuous_bins);

class LinkageModelBase {
 public:
  // LinkageModelBase() = default;
  // LinkageModelBase(const LinkageModelBase&) = delete;
  // LinkageModelBase(LinkageModelBase&&) = delete;

  // LinkageModelBase& operator=(const LinkageModelBase&) = delete;
  // LinkageModelBase& operator=(LinkageModelBase&&) = delete;

  virtual void init(Rng& rng, InstanceBase& problem, SolutionSetBase& solutions, VariableSet variables) = 0;
  virtual FOS subsets(Rng& rng,
                      InstanceBase& problem,
                      SolutionSetBase& solutions,
                      const std::span<const usize> indices,
                      std::optional<std::reference_wrapper<const Mat<CType>>> covariance) const = 0;

  virtual bool is_static() const { return false; };

  virtual std::unique_ptr<LinkageModelBase> clone() const = 0;

  virtual ~LinkageModelBase() = default;  //{};
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
  LinkageTreeFOS(std::string metric = "mi",  // nmi, pearson_r2, random

                 /// The intron strategy determines how knowledge about inactive variables
                 /// is used to modify the estimation of linkage when learning the linkage
                 /// tree
                 /// - none: Ignore introns
                 /// - any_active: introns are marked as such to remove random noise and
                 /// only variable pairs with at least one active variable are considered
                 /// - all_active: Only variable pairs where both variables are active are
                 /// considered
                 /// - mark_only: introns are marked as such to reduce noise, but pairs
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
                 std::optional<Subset> subset = std::nullopt,
                 std::optional<Mat<CType>> custom_similarity = std::nullopt,
                 std::optional<CType> eta_custom_similarity = std::nullopt,
                 std::optional<std::string> custom_similarity_agg = std::nullopt,
                 std::optional<std::function<void(CRef<Mat<CType>>)>> similarity_callback = std::nullopt,
                 bool freeze = false)
      : subset(subset.value_or(Subset{})),
        custom_similarity(custom_similarity),
        similarity_callback(similarity_callback),
        metric(metric),
        intron_strategy(intron_strategy),
        num_continuous_bins(num_continuous_bins),
        filter_parent_threshold(filter_parent_threshold),
        filter_children_threshold(filter_children_threshold),
        filter_root(filter_root),
        max_subset_size(max_subset_size),
        eta_custom_similarity(eta_custom_similarity),
        custom_similarity_agg(custom_similarity_agg),
        merge_continuous(merge_continuous),
        normalize_initial_linkage_bias(normalize_initial_linkage_bias),
        freeze(freeze) {
    if (this->eta_custom_similarity.has_value()) {
      if (!(0.0 < this->eta_custom_similarity.value() && this->eta_custom_similarity.value() < 1.0)) {
        throw std::runtime_error("Invalid eta_custom_similarity, must be in (0, 1) if provided.");
      }
    }
  };

  std::unique_ptr<LinkageModelBase> clone() const override final { return std::make_unique<LinkageTreeFOS>(*this); }

  bool is_static() const override final { return freeze; };

  void register_similarity_callback(std::function<void(CRef<Mat<CType>>)>&& similarity_callback) {
    this->similarity_callback = similarity_callback;
  };

  void unregister_similarity_callback() { this->similarity_callback = std::nullopt; };

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

    if (custom_similarity.has_value()) {
      if (static_cast<usize>(custom_similarity.value().rows()) != subset.size() ||
          static_cast<usize>(custom_similarity.value().cols()) != subset.size()) {
        throw std::runtime_error(
            "Custom similarity matrix has invalid dimensions, expected a square matrix corresponding to the subset "
            "size.");
      }
    }

    if (normalize_initial_linkage_bias && variables == VariableSet::Discrete) {
      // __goblin_runtime_assert(intron_strategy == "none");
      // intron awareness and Marco's linkage normalization are incompatible
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

  Mat<CType> compute_similarity(
      Rng& rng,
      InstanceBase& problem,
      SolutionSetBase& solutions,
      const std::span<const usize> indices,
      std::optional<std::reference_wrapper<const Mat<CType>>> covariance = std::nullopt) const {
    const isize l = subset.size();
    const bool is_discrete = subset.discrete.size() > 0;
    const bool is_continuous = subset.continuous.size() > 0;

    Mat<CType> similarity;
    if (custom_similarity.has_value() && !eta_custom_similarity.has_value()) {
      similarity = custom_similarity.value();
    } else if (metric == "random") {
      similarity.resize(l, l);
      std::uniform_real_distribution<CType> U(0.0, 1.0);
      for (isize i = 0; i < l; i++) {
        for (isize j = 0; j <= i; j++) {
          similarity(i, j) = U(rng);
          similarity(j, i) = similarity(i, j);
        }
      }
    } else if (is_discrete && is_continuous) {
      similarity.resize(l, l);
      // No mixed subset learning (yet?)
      // https://homepages.cwi.nl/~bosman/publications/2016_learningandexploiting.pdf
      throw std::runtime_error("Mixed linkage learning isn't supported.");
    } else if (is_discrete) {
      similarity = estimate_entropy(problem, solutions, indices, subset.discrete, intron_strategy, merge_continuous,
                                    num_continuous_bins);
      if (normalize_initial_linkage_bias && initial_bias_adjustments.size() > 0) {
        similarity.array() *= initial_bias_adjustments.array();
      }
      entropy2similarity(similarity);
    } else if (is_continuous) {
      similarity.resize(l, l);
      if (!covariance.has_value()) {
        throw std::runtime_error("Continuous linkage learning requires a covariance matrix.");
      } else if (covariance.value().get().rows() != l || covariance.value().get().cols() != l) {
        throw std::runtime_error(
            "Covariance matrix has invalid dimensions, expected a square matrix corresponding to the subset size.");
      }

      Mat<CType> cov = covariance.value();
      for (isize i = 0; i < l; i++) {
        for (isize j = 0; j < i; j++) {
          // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
          double r = cov(i, j) / (std::sqrt(cov(i, i)) * std::sqrt(cov(j, j)));
          similarity(i, j) = std::log(std::sqrt(1.0 / (1.0 - r * r)));
          similarity(j, i) = similarity(i, j);
        }
      }
    } else {
      throw std::runtime_error("Unknown subset of variables to learn a LT for.");
    }

    if (custom_similarity.has_value() && custom_similarity_agg.has_value()) {
      const auto& agg = custom_similarity_agg.value();
      if (agg == "add") {
        similarity = similarity + custom_similarity.value();
      } else if (agg == "mul") {
        similarity = similarity.array() * custom_similarity.value().array();
      } else if (agg == "max") {
        similarity = similarity.array().max(custom_similarity.value().array());
      } else {
        throw std::runtime_error("Unknown or unsupported similarity aggregation method.");
      }
    } else if (custom_similarity.has_value() && eta_custom_similarity.has_value()) {
      similarity = (1.0 - eta_custom_similarity.value()) * similarity +
                   eta_custom_similarity.value() * custom_similarity.value();
    }

    if (similarity_callback.has_value()) {
      similarity_callback.value()(similarity);
    }
    return similarity;
  };

  FOS learn_lt(Rng& rng, Mat<CType> similarity, bool filter_root_default) const {
    auto merges = UPGMA::cluster(rng, similarity);

    // 3. merge subsets...
    isize l = similarity.rows();
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
    if (num_merges > 0 && filter_root.value_or(filter_root_default)) {
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
    return fos;
  };

  FOS learn_lt_original(Rng& rng, Mat<CType> similarity, bool filter_root_default) const {
    const isize numberOfVariables = subset.size();
    const isize maximumSetSize = max_subset_size.value_or(numberOfVariables);

    Mat<CType> S = similarity;

    auto determineNearestNeighbour = [&](usize index, const std::vector<std::vector<isize>>& mpm) {
      usize result = 0;

      if (result == index)
        result++;

      for (usize i = 1; i < mpm.size(); i++) {
        if (mpm[i].size() > static_cast<usize>(numberOfVariables)) {
          assert(false);
        }
        if (i != index) {
          if (mpm[index].size() + mpm[result].size() > static_cast<usize>(maximumSetSize)) {
            if (mpm[i].size() < mpm[result].size()) {
              result = i;
            }
          } else if (mpm[index].size() + mpm[i].size() <= static_cast<usize>(maximumSetSize)) {
            if ((S(index, i) > S(index, result)) ||
                ((S(index, i) == S(index, result)) && (mpm[i].size() < mpm[result].size()))) {
              result = i;
            }
          }
        }
      }
      return result;
    };

    std::vector<std::vector<isize>> fos_;
    fos_.reserve(2 * numberOfVariables - 1);

    std::vector<isize> mpmFOSMap;
    std::vector<isize> mpmFOSMapNew;

    /* Initialize MPM to the univariate factorization */
    std::vector<isize> order(numberOfVariables);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    std::vector<std::vector<isize>> mpm(numberOfVariables);
    std::vector<std::vector<isize>> mpmNew(numberOfVariables);

    for (isize i = 0; i < numberOfVariables; i++) {
      mpm[i].push_back(order[i]);
    }

    /* Initialize LT to the initial MPM */
    fos_.resize(numberOfVariables);

    isize FOSsIndex = 0;
    for (isize i = 0; i < numberOfVariables; i++) {
      fos_[i] = mpm[i];
      mpmFOSMap.push_back(i);
      FOSsIndex++;
    }

    for (isize i = 0; i < numberOfVariables; ++i) {
      for (isize j = 0; j < numberOfVariables; j++)
        S(i, j) = similarity(mpm[i][0], mpm[j][0]);

      S(i, i) = 0;
    }
    // printf("Initialized similarity matrix. (%.3fs)\n",getTime(t)/1000.0);

    std::vector<usize> NN_chain;
    NN_chain.resize(numberOfVariables + 2);
    isize NN_chain_length = 0;
    bool done = false;
    while (!done) {
      if (NN_chain_length == 0) {
        NN_chain[NN_chain_length] =
            std::uniform_int_distribution<usize>(0, mpm.size() - 1)(rng);  //  utils::rng() % mpm.size();

        NN_chain_length++;
      }

      while (NN_chain_length < 3) {
        NN_chain[NN_chain_length] = determineNearestNeighbour(NN_chain[NN_chain_length - 1], mpm);
        NN_chain_length++;
      }

      while (NN_chain[NN_chain_length - 3] != NN_chain[NN_chain_length - 1]) {
        NN_chain[NN_chain_length] = determineNearestNeighbour(NN_chain[NN_chain_length - 1], mpm);
        if (((S(NN_chain[NN_chain_length - 1], NN_chain[NN_chain_length]) ==
              S(NN_chain[NN_chain_length - 1], NN_chain[NN_chain_length - 2]))) &&
            (NN_chain[NN_chain_length] != NN_chain[NN_chain_length - 2]))
          NN_chain[NN_chain_length] = NN_chain[NN_chain_length - 2];

        NN_chain_length++;
        if (NN_chain_length > numberOfVariables)
          break;
      }

      usize r0 = NN_chain[NN_chain_length - 2];
      usize r1 = NN_chain[NN_chain_length - 1];
      bool skipFOSElement = false;
      // TODO
      // if( filtered )
      // {
      // 	if ( (similarityMeasure == linkage::MI || similarityMeasure == linkage::NMI ) )
      // 	{
      // 		if( S(r1, r0) >= 1-(1e-6))
      // 			skipFOSElement = true;
      // 	}
      // 	else
      // 	{
      // 		if( S(r1, r0) == 0 )
      // 			skipFOSElement = true;
      // 	}
      // }

      if (r1 >= mpm.size() || r0 >= mpm.size() ||
          mpm[r0].size() + mpm[r1].size() > static_cast<usize>(maximumSetSize)) {
        NN_chain_length = 1;
        NN_chain[0] = 0;
        if (maximumSetSize < numberOfVariables) {
          done = true;
          for (usize i = 1; i < mpm.size(); i++) {
            if (mpm[i].size() + mpm[NN_chain[0]].size() <= static_cast<usize>(maximumSetSize)) {
              done = false;
              // printf("%d %d [%d + %d]\n",i,NN_chain[0],mpm[i].size(),mpm[NN_chain[0]].size());
            }
            if (mpm[i].size() < mpm[NN_chain[0]].size())
              NN_chain[0] = i;
          }
          if (done)
            break;
        }
        continue;
      }

      if (r0 > r1) {
        isize rswap = r0;
        r0 = r1;
        r1 = rswap;
      }
      NN_chain_length -= 3;

      if (r1 < mpm.size()) {
        if (mpm[r0].size() + mpm[r1].size() > static_cast<usize>(maximumSetSize)) {
          done = true;
          break;
        }
        std::vector<isize> indices(mpm[r0].size() + mpm[r1].size());

        usize i = 0;
        for (usize j = 0; j < mpm[r0].size(); j++) {
          indices[i] = mpm[r0][j];
          i++;
        }

        for (usize j = 0; j < mpm[r1].size(); j++) {
          indices[i] = mpm[r1][j];
          i++;
        }

        if (!skipFOSElement) {
          fos_.push_back(indices);
          FOSsIndex++;
          assert(fos_.size() == static_cast<usize>(FOSsIndex));
        }

        double mul0 = static_cast<double>(mpm[r0].size()) / static_cast<double>(mpm[r0].size() + mpm[r1].size());
        double mul1 = static_cast<double>(mpm[r1].size()) / static_cast<double>(mpm[r0].size() + mpm[r1].size());
        for (usize i = 0; i < mpm.size(); i++) {
          if ((i != r0) && (i != r1)) {
            S(i, r0) = mul0 * S(i, r0) + mul1 * S(i, r1);
            S(r0, i) = S(i, r0);
          }
        }

        mpmNew.resize(mpm.size() - 1);
        mpmFOSMapNew.resize(mpmFOSMap.size() - 1);
        for (usize i = 0; i < mpmNew.size(); i++) {
          mpmNew[i] = mpm[i];
          mpmFOSMapNew[i] = mpmFOSMap[i];
        }

        mpmNew[r0] = indices;
        mpmFOSMapNew[r0] = FOSsIndex - 1;

        if (r1 < mpm.size() - 1) {
          mpmNew[r1] = mpm[mpm.size() - 1];
          mpmFOSMapNew[r1] = mpmFOSMap[mpm.size() - 1];

          for (usize i = 0; i < r1; i++) {
            S(i, r1) = S(i, mpm.size() - 1);
            S(r1, i) = S(i, r1);
          }

          for (usize j = r1 + 1; j < mpmNew.size(); j++) {
            S(r1, j) = S(j, mpm.size() - 1);
            S(j, r1) = S(r1, j);
          }
        }

        for (i = 0; i < static_cast<usize>(NN_chain_length); i++) {
          if (NN_chain[i] == mpm.size() - 1) {
            NN_chain[i] = r1;
            break;
          }
        }

        mpm = mpmNew;
        mpmFOSMap = mpmFOSMapNew;

        if (!filter_root.value_or(filter_root_default)) {
          if (mpm.size() == 1)
            done = true;
        } else {
          if (mpm.size() == 2)
            done = true;
        }
      }
    }

    // Map subset indices to structs
    FOS fos;
    fos.resize(fos_.size());
    for (usize i = 0; i < fos_.size(); i++) {
      for (usize j : fos_[i]) {
        if (j < subset.discrete.size()) {
          fos[i].discrete.push_back(subset.discrete[j]);
        } else {
          fos[i].continuous.push_back(subset.continuous[j - subset.discrete.size()]);
        }
      }
    }

    return fos;
  };

  FOS subsets(Rng& rng,
              InstanceBase& problem,
              SolutionSetBase& solutions,
              const std::span<const usize> indices,
              std::optional<std::reference_wrapper<const Mat<CType>>> covariance = std::nullopt) const override final {
    // 1. get similarity matrix based on the measure...
    Mat<CType> similarity = compute_similarity(rng, problem, solutions, indices, covariance);
    assert(static_cast<usize>(similarity.rows()) == subset.size() &&
           static_cast<usize>(similarity.cols()) == subset.size());

    // 2. turn that into a FOS by clustering + merging subsets
    bool is_discrete = subset.discrete.size() > 0;
    bool is_continuous = subset.continuous.size() > 0;
    bool filter_root_default = is_discrete && !is_continuous;
    FOS fos = learn_lt(rng, similarity, filter_root_default);
    // FOS fos = learn_lt_original(rng, similarity, filter_root_default);

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
    CType tmp;
    if (metric == "mi") {
      auto& MI = H;
      for (isize i = 0; i < H.rows(); i++) {
        for (isize j = 0; j < i; j++) {
          tmp = H(i, i) + H(j, j) - H(i, j);
          MI(i, j) = tmp;
          MI(j, i) = tmp;
        }
      }
    } else if (metric == "nmi") {
      auto& NMI = H;
      for (isize i = 0; i < H.rows(); i++) {
        for (isize j = 0; j < i; j++) {
          tmp = H(i, j) > 0 ? (((H(i, i) + H(j, j)) / H(i, j)) - CType(1.0)) : CType(0.0);
          NMI(i, j) = tmp;
          NMI(j, i) = tmp;
        }
      }
    } else {
      // TODO raise unknown or unsupported for discrete domain
      __goblin_runtime_assert(metric != metric);
    }
  };

  Subset subset;
  std::optional<Mat<CType>> custom_similarity;
  std::optional<std::function<void(CRef<Mat<CType>>)>> similarity_callback;
  std::string metric;
  std::string intron_strategy;
  std::optional<usize> num_continuous_bins;
  std::optional<CType> filter_parent_threshold;
  std::optional<CType> filter_children_threshold;
  std::optional<bool> filter_root;
  std::optional<usize> max_subset_size;
  std::optional<CType> eta_custom_similarity;
  std::optional<std::string> custom_similarity_agg;
  bool merge_continuous;
  bool normalize_initial_linkage_bias;
  bool freeze;

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
