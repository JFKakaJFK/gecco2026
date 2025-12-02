#pragma once
#ifndef _GOBLIN_LIB_INSTANCE_H
#define _GOBLIN_LIB_INSTANCE_H

#include <format>
#include <iostream>
#include <sstream>
#include <tuple>
#include <optional>

#include "goblin/lib/types.h"
#include "goblin/lib/fitness.h"
#include "goblin/lib/solution.h"
#include "goblin/lib/archive.h"

namespace goblin {
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

  // corresponds to e.g. ERCs / one constant per edge in GP
  virtual bool always_inherit_continuous() const { return false; };

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

  virtual void log(std::ostream& os, SolutionBase& solution) {
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

  virtual ~InstanceBase() {};
};
};  // namespace goblin

#endif /* _GOBLIN_LIB_INSTANCE_H */
