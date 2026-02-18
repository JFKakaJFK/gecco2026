#pragma once
#ifndef _GOBLIN_LIB_INSTANCE_H
#define _GOBLIN_LIB_INSTANCE_H

#include <format>
#include <iostream>
#include <sstream>
#include <tuple>
#include <optional>
#include <cstddef>
#include <memory>

#include "goblin/lib/types.h"
#include "goblin/lib/fitness.h"
#include "goblin/lib/solution.h"
#include "goblin/lib/archive.h"

namespace goblin {

class CacheKey {
 public:
  explicit CacheKey(const std::string& key);
  CacheKey(std::vector<std::byte>&& key);

  bool operator==(const CacheKey& other) const { return hash_value_ == other.hash_value_ && key_ == other.key_; };

 private:
  friend struct std::hash<CacheKey>;

  std::vector<std::byte> key_;
  usize hash_value_;
};

class InstanceBase {
 public:
  virtual usize num_objectives() const { return fitness().num_objectives(); };

  virtual usize num_discrete() const = 0;
  virtual CRef<Vec<DType>> discrete_domain_sizes() const = 0;

  virtual usize num_continuous() const = 0;
  virtual CRef<Vec<CType>> continuous_lower_bounds() const = 0;
  virtual CRef<Vec<CType>> continuous_upper_bounds() const = 0;

  virtual CRef<Vec<CType>> continuous_init_lower_bounds() const = 0;
  virtual CRef<Vec<CType>> continuous_init_upper_bounds() const = 0;

  virtual void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) = 0;
  virtual void evaluate_partial(Rng& rng,
                                SolutionSetBase& solutions,
                                SolutionSetBase& parents,
                                const std::vector<const Subset*>& subsets,
                                const std::span<const usize>& indices) {
    evaluate(rng, solutions, indices);
  };

  void evaluate(SolutionSetBase& solutions, std::optional<u64> seed = std::nullopt) {
    Rng rng = seeded_rng(seed);
    std::vector<usize> indices(solutions.size());
    std::iota(indices.begin(), indices.end(), 0);
    evaluate(rng, solutions, indices);
  };

  /// Possibly adapts the problem in some way that may require re-evaluating any elites stored thus far (indicated by
  /// the return value)
  virtual bool adapt(Rng& rng) { return false; };

  /// Returns the gradient for each index of indices (row) and continuous variable (column) with respect to the
  /// optimization goal. The number of evaluations performed to calculate the gradients are added to `evaluations`;
  ///
  /// Note: The default implementation assumes that the passes subsets contain all active indices and the finite
  /// difference approach used requires up to `2 * indices.size() * num_continuous()` evaluations.
  virtual Mat<CType> gradients(Rng& rng,
                               SolutionSetBase& solutions,
                               SolutionSetBase& parents,
                               const std::vector<const Subset*>& subsets,
                               const std::span<const usize>& indices,
                               u64& evaluations);

  /// Performs gradient optimization steps of the continuous variables and returns the
  /// subset of `indices` that changed together with the number of evaluations performed.
  /// Guarantees that the quality of all solutions is up-to-date.
  virtual std::tuple<std::vector<usize>, u64> gradient_steps(Rng& rng,
                                                             SolutionSetBase& solutions,
                                                             SolutionSetBase& parents,
                                                             const std::span<const usize>& indices,
                                                             usize num_steps);

  virtual void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const = 0;

  virtual const FitnessBase& fitness() const = 0;

  /// The fitness the archive should be use, in case the archive should record the front in more objectives than what
  /// should be optimized (as per https://arxiv.org/abs/2507.03777v1)
  ///
  /// Note: What `fitness()` optimizes must always be a compatible subset of what `archive_fitness()` optimizes.
  virtual const ArchiveFitnessBase& archive_fitness() const = 0;

  // // corresponds to e.g. ERCs / one constant per edge in GP
  // virtual bool always_inherit_continuous() const { return false; };

  /// The offspring inherits a subset of the decision variables from the donor, returning true if there was a change to
  /// the active variables and an evaluation is needed, possibly with problem specific modifications.
  ///
  /// Returns a `(any_active_changed, anything_changed)` tuple
  virtual std::tuple<bool, bool> inherit_discrete(SolutionBase& offspring,
                                                  const SolutionBase& donor,
                                                  const Subset& subset) const {
    bool any_active_changed = false, anything_changed = false;

    for (usize i : subset.discrete) {
      if (offspring.discrete_values()(i) != donor.discrete_values()(i)) {
        any_active_changed |= offspring.discrete_active()(i);
        anything_changed = true;
        offspring.discrete_values()(i) = donor.discrete_values()(i);
      }
    }
    return std::make_tuple(any_active_changed, anything_changed);
  };

  // useful for discrete linkage learning in GP
  // - needed to be able to perform constant binning
  // - needed to erase constant pool references
  virtual std::optional<CType> as_continuous(const SolutionBase& solution, usize discrete_index) const {
    return std::nullopt;
  };

  virtual bool target_reached(const ArchiveBase& archive) const { return false; };

  virtual void log_header(std::ostream& os) const {
    os << "values,";
    fitness().log_header(os);
  };

  virtual void log_solution(std::ostream& os, const SolutionBase& solution) const {
    os << "([";
    for (usize i = 0; i < solution.num_discrete(); i++) {
      if (i > 0) {
        os << ",";
      }
      if (solution.discrete_active()(i)) {
// fmt to alwyas use the decimal instead of the ascii byte value for (unsigned)
// chars
#ifdef __cpp_lib_print
        std::print(os,
#else
        os << std::format(
#endif
                   "{:d}", solution.discrete_values()(i));
      } else {
        os << "█";
      }
    }
    os << "],[";
    for (usize i = 0; i < solution.num_continuous(); i++) {
      if (i > 0) {
        os << ",";
      }
      if (solution.continuous_active()(i)) {
        os << solution.continuous_values()(i);
      } else {
        os << "█";
      }
    }
    os << "])";
  };

  virtual void log(std::ostream& os, const SolutionBase& solution) const {
    os << '"';
    log_solution(os, solution);
    os << "\",";
    fitness().log(os, solution.quality());
  };

  std::string format_solution(const SolutionBase& solution) const {
    std::stringstream ss;
    log_solution(ss, solution);
    return ss.str();
  };

  /// By returning std::nullopt, caching can be disabled on a per solution basis
  virtual std::optional<CacheKey> solution_cache_key(const SolutionBase& solution) const {
    return CacheKey(format_solution(solution));
  };

  u64 reevaluate_and_rebuild_archive(Rng& rng, ArchiveBase& archive) {
    // 1. put all solutions into a solutionset
    AoSSet solutions;
    std::vector<usize> indices;
    indices.reserve(archive.size());
    for (usize i = 0; i < archive.size(); i++) {
      solutions.add(archive[i]);
      indices.push_back(i);
    }
    // 2. evaluate them
    evaluate(rng, solutions, indices);
    // 3. re-build the archive
    archive.clear();
    for (usize i = 0; i < solutions.size(); i++) {
      archive.update(solutions[i], true);
    }

    return indices.size();
  };

  virtual ~InstanceBase() {};
};

/// Intermediate class for wrapping instances that by default forwards everything to the actual inner method. Still
/// allows accidentally not overwriting some methods, but at least defaults to the behaviour of the wrapped instance
/// instead of the default implementations of virtual methods.
class WrappedInstance : public InstanceBase {
 public:
  WrappedInstance(InstanceBase& instance) : inner(instance) {};

  usize num_objectives() const override { return inner.num_objectives(); };

  usize num_discrete() const override { return inner.num_discrete(); }
  CRef<Vec<DType>> discrete_domain_sizes() const override { return inner.discrete_domain_sizes(); }

  usize num_continuous() const override { return inner.num_continuous(); }
  CRef<Vec<CType>> continuous_lower_bounds() const override { return inner.continuous_lower_bounds(); }
  CRef<Vec<CType>> continuous_upper_bounds() const override { return inner.continuous_upper_bounds(); }

  CRef<Vec<CType>> continuous_init_lower_bounds() const override { return inner.continuous_init_lower_bounds(); }
  CRef<Vec<CType>> continuous_init_upper_bounds() const override { return inner.continuous_init_upper_bounds(); }

  void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override {
    return inner.evaluate(rng, solutions, indices);
  };
  void evaluate_partial(Rng& rng,
                        SolutionSetBase& solutions,
                        SolutionSetBase& parents,
                        const std::vector<const Subset*>& subsets,
                        const std::span<const usize>& indices) override {
    return inner.evaluate_partial(rng, solutions, parents, subsets, indices);
  };

  bool adapt(Rng& rng) override { return inner.adapt(rng); };

  Mat<CType> gradients(Rng& rng,
                       SolutionSetBase& solutions,
                       SolutionSetBase& parents,
                       const std::vector<const Subset*>& subsets,
                       const std::span<const usize>& indices,
                       u64& evaluations) override {
    return inner.gradients(rng, solutions, parents, subsets, indices, evaluations);
  }

  std::tuple<std::vector<usize>, u64> gradient_steps(Rng& rng,
                                                     SolutionSetBase& solutions,
                                                     SolutionSetBase& parents,
                                                     const std::span<const usize>& indices,
                                                     usize num_steps) override {
    return inner.gradient_steps(rng, solutions, parents, indices, num_steps);
  };

  void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override {
    return inner.add_random(rng, solutions, count);
  };

  const FitnessBase& fitness() const override { return inner.fitness(); };
  const ArchiveFitnessBase& archive_fitness() const override { return inner.archive_fitness(); };

  std::tuple<bool, bool> inherit_discrete(SolutionBase& offspring,
                                          const SolutionBase& donor,
                                          const Subset& subset) const override {
    return inner.inherit_discrete(offspring, donor, subset);
  };

  std::optional<CType> as_continuous(const SolutionBase& solution, usize discrete_index) const override {
    return inner.as_continuous(solution, discrete_index);
  };

  bool target_reached(const ArchiveBase& archive) const override { return inner.target_reached(archive); };

  void log_header(std::ostream& os) const override { return inner.log_header(os); };

  void log_solution(std::ostream& os, const SolutionBase& solution) const override {
    return inner.log_solution(os, solution);
  };

  void log(std::ostream& os, const SolutionBase& solution) const override { return inner.log(os, solution); };

  std::optional<CacheKey> solution_cache_key(const SolutionBase& solution) const override {
    return inner.solution_cache_key(solution);
  };

  virtual ~WrappedInstance() = default;

 protected:
  InstanceBase& inner;
};

class CachedInstanceBase : public WrappedInstance {
 public:
  CachedInstanceBase(InstanceBase& instance) : WrappedInstance(instance) {};

  virtual usize hit_count() const = 0;
  virtual usize miss_count() const = 0;
  virtual usize access_count() const = 0;
  virtual usize entry_invalidation_count() const = 0;
  virtual usize cache_invalidation_count() const = 0;
  virtual usize evicted_count() const = 0;

  virtual CType hit_ratio() const = 0;
  virtual CType miss_ratio() const = 0;
  /// Proportion of cache entries used w.r.t. maximum size
  virtual CType utilization() const = 0;
  // TODO expose cache api & stats...
  virtual ~CachedInstanceBase() = default;
};

std::shared_ptr<CachedInstanceBase> Cached(std::shared_ptr<InstanceBase> problem,
                                           usize cache_size = 10000,
                                           std::string cache_policy = "lru");

};  // namespace goblin

#endif /* _GOBLIN_LIB_INSTANCE_H */
