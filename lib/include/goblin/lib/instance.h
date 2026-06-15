#pragma once
#ifndef _GOBLIN_LIB_INSTANCE_H
#define _GOBLIN_LIB_INSTANCE_H

#include <vector>
#include <cstddef>
#include <format>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <tuple>
#include <variant>
#include <stdexcept>
#include <functional>
#include <type_traits>

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

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

class InstanceBase {
  std::function<bool(const ArchiveBase&)> _target_reached;

 public:
  usize num_objectives() const { return fitness().num_objectives(); };

  usize num_discrete() const { return discrete_domain_sizes().size(); };
  virtual CRef<Vec<DType>> discrete_domain_sizes() const = 0;

  usize num_continuous() const { return continuous_lower_bounds().size(); };
  virtual CRef<Vec<CType>> continuous_lower_bounds() const = 0;
  virtual CRef<Vec<CType>> continuous_upper_bounds() const = 0;

  virtual CRef<Vec<CType>> continuous_init_lower_bounds() const = 0;
  virtual CRef<Vec<CType>> continuous_init_upper_bounds() const = 0;

  // TODO support ordinal discrete spaces (e.g. bool per discrete to indicate categorical/ordinal)
  // TODO support permutation spaces (e.g. bool per continuous variable to indicate continuous/random keys)

  virtual void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) = 0;
  virtual void evaluate_partial(Rng& rng,
                                SolutionSetBase& solutions,
                                SolutionSetBase& parents,
                                const std::vector<const Subset*>& subsets,
                                const std::span<const usize>& indices) {
    evaluate(rng, solutions, indices);
  };

  void evaluate_solutions(SolutionSetBase& solutions, std::optional<u64> seed = std::nullopt) {
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

  bool target_reached(const ArchiveBase& archive) const {
    if (_target_reached) {
      return _target_reached(archive);
    }
    return false;
  };

  InstanceBase& add_target(std::function<bool(const ArchiveBase&)> target_reached_check) {
    if (_target_reached) {
      auto existing = std::move(_target_reached);
      _target_reached = [a = std::move(existing), b = std::move(target_reached_check)](const ArchiveBase& archive) {
        return a(archive) || b(archive);
      };
    } else {
      _target_reached = std::move(target_reached_check);
    }
    return *this;
  }
  InstanceBase& add_target(const ArchiveBase& target_front) {
    std::shared_ptr<ArchiveBase> front = target_front.clone();
    std::function<bool(const ArchiveBase&)> incoming = [front = std::move(front)](const ArchiveBase& archive) {
      return archive.covers(*front);
    };

    return add_target(std::move(incoming));
  }
  InstanceBase& add_target(const SolutionBase& target_solution) {
    UnboundedArchive target_front(archive_fitness());
    target_front.update(target_solution);
    return add_target(target_front);
  }
  InstanceBase& add_target(const QualityBase& target_quality) {
    Solution s(
        target_quality.clone(),
        num_discrete() > 0 ? std::make_optional<Vec<DType>>(Vec<DType>::Zero(num_discrete())) : std::nullopt,
        num_continuous() > 0 ? std::make_optional<Vec<CType>>(Vec<CType>::Zero(num_continuous())) : std::nullopt);
    return add_target(s);
  }
  InstanceBase& add_target(std::variant<Vec<CType>, std::vector<CType>> target_objectives) {
    auto vtr = archive_fitness().worst();
    if (auto p = dynamic_cast<const MOQuality*>(vtr.get()); p == nullptr) {
      throw std::runtime_error(
          "A vector of objective values is only a valid target value in combination with `MOQuality` or a subclass "
          "as quality.");
    }

    auto& q = vtr->as<MOQuality>();
    q.constraint_value = 0.0;
    q.objectives.fill(0.0);

    Vec<CType> to;
    if (std::holds_alternative<Vec<CType>>(target_objectives)) {
      to = std::get<Vec<CType>>(target_objectives);
    } else {
      auto t = std::get<std::vector<CType>>(target_objectives);
      to = Eigen::Map<Vec<CType>>(t.data(), t.size());
    }

    usize no = to.size();
    if (fitness().num_objectives() > no || no > archive_fitness().num_objectives()) {
      throw std::runtime_error("Mismatch in number of target objectives provided.");
    }
    for (usize i = 0; i < no; i++) {
      q.objectives(i) = to(i);
    }

    return add_target(*vtr);
  };
  InstanceBase& add_target_front_size(usize target_front_size) {
    return add_target([target_front_size](const ArchiveBase& archive) { return archive.size() >= target_front_size; });
  }
  InstanceBase& add_target_front(Mat<CType> target_objectives) {
    auto vtr = archive_fitness().worst();
    if (auto p = dynamic_cast<const MOQuality*>(vtr.get()); p == nullptr) {
      throw std::runtime_error(
          "A matrix of objective values is only a valid target value in combination with `MOQuality` or a subclass "
          "as quality.");
    }

    auto& q = vtr->as<MOQuality>();
    q.constraint_value = 0.0;
    q.objectives.fill(0.0);

    usize no = target_objectives.cols();
    if (fitness().num_objectives() > no || no > archive_fitness().num_objectives()) {
      throw std::runtime_error("Mismatch in number of target objectives provided.");
    }

    UnboundedArchive target_front(archive_fitness());
    for (isize i = 0; i < target_objectives.rows(); i++) {
      for (isize j = 0; j < target_objectives.cols(); j++) {
        q.objectives(j) = target_objectives(i, j);
      }
      target_front.update(Solution(vtr->clone(), Vec<DType>::Zero(num_discrete()), Vec<CType>::Zero(num_continuous())));
    }

    return add_target(target_front);
  };
  InstanceBase& add_target_front(std::optional<Mat<DType>> discrete,
                                 std::optional<Mat<CType>> continuous,
                                 std::optional<usize> evaluation_seed = std::nullopt) {
    if (num_discrete() > 0) {
      if (!discrete.has_value()) {
        throw std::runtime_error("Problem has discrete variables but none were provided.");
      }
      if (static_cast<usize>(discrete.value().cols()) != num_discrete()) {
        throw std::runtime_error("Mismatch in number of discrete variables provided");
      }
    }
    if (num_continuous() > 0) {
      if (!continuous.has_value()) {
        throw std::runtime_error("Problem has continuous variables but none were provided.");
      }
      if (static_cast<usize>(discrete.value().cols()) != num_discrete()) {
        throw std::runtime_error("Mismatch in number of discrete variables provided");
      }
    }
    if (num_discrete() > 0 && num_continuous() > 0 && discrete.value().rows() != continuous.value().rows()) {
      throw std::runtime_error("Mismatch in number of discrete/continuous solutions provided");
    }

    usize n = std::max(num_discrete(), num_continuous());

    AoSSet s;
    std::vector<usize> indices(n);
    for (usize i = 0; i < n; i++) {
      s.add(Solution(archive_fitness().worst(),
                     num_discrete() > 0 ? std::make_optional<Vec<DType>>(discrete.value().row(i)) : std::nullopt,
                     num_continuous() > 0 ? std::make_optional<Vec<CType>>(continuous.value().row(i)) : std::nullopt));
    }

    Rng rng = seeded_rng(evaluation_seed);
    evaluate(rng, s, indices);

    UnboundedArchive target_front(archive_fitness());
    for (usize i = 0; i < n; i++) {
      target_front.update(s[i]);
    }

    return add_target(target_front);
  }
  InstanceBase& add_target(CType target_fitness) {
    if (num_objectives() != 1) {
      throw std::runtime_error("A single objective value is only a valid target value for single-objective instances.");
    }
    auto vtr = archive_fitness().worst();
    if (auto p = dynamic_cast<const MOQuality*>(vtr.get()); p == nullptr) {
      throw std::runtime_error(
          "A single objective value is only a valid target value in combination with `MOQuality` or a subclass as "
          "quality.");
    }

    auto& q = vtr->as<MOQuality>();
    q.constraint_value = 0.0;
    q.objectives.fill(0.0);
    q.objectives(0) = target_fitness;

    return add_target(*vtr);
  };

  virtual void log_header(std::ostream& os) const {
    os << "values,";
    archive_fitness().log_header(os);
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
    archive_fitness().log(os, solution.quality());
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

  virtual const InstanceBase& unwrap() const { return *this; }

  virtual ~InstanceBase() = default;
};

/// Intermediate class for wrapping instances that by default forwards everything to the actual inner method. Still
/// allows accidentally not overwriting some methods, but at least defaults to the behaviour of the wrapped instance
/// instead of the default implementations of virtual methods.
class WrappedInstance : public InstanceBase {
 public:
  WrappedInstance(InstanceBase& instance) : inner(instance) {};

  CRef<Vec<DType>> discrete_domain_sizes() const override { return inner.discrete_domain_sizes(); }

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

  bool target_reached(const ArchiveBase& archive) const { return inner.target_reached(archive); };

  void log_header(std::ostream& os) const override { return inner.log_header(os); };

  void log_solution(std::ostream& os, const SolutionBase& solution) const override {
    return inner.log_solution(os, solution);
  };

  void log(std::ostream& os, const SolutionBase& solution) const override { return inner.log(os, solution); };

  std::optional<CacheKey> solution_cache_key(const SolutionBase& solution) const override {
    return inner.solution_cache_key(solution);
  };

  const InstanceBase& unwrap() const override { return inner.unwrap(); }

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
