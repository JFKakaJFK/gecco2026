#pragma once
#ifndef _GOBLIN_LIB_ARCHIVE_H
#define _GOBLIN_LIB_ARCHIVE_H

#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <print>
#include <type_traits>
#include <tuple>

#include <Eigen/Dense>

#include "goblin/lib/assert.h"
#include "goblin/lib/fitness.h"
#include "goblin/lib/ordering.h"
#include "goblin/lib/rng.h"
#include "goblin/lib/solution.h"

namespace goblin {

class ArchiveBase {
 public:
  virtual usize size() const = 0;
  bool empty() const { return size() == 0; };
  virtual const SolutionBase& operator[](usize idx) const = 0;

  // Unsafe because by being in the archive we assume the invariant that the
  // solution is non-dominated. Allowing potential changes to the solution
  // without re-checking non-dominance hence is not "safe" from the logic
  // perspective.
  virtual SolutionBase& unsafe_at(usize idx) = 0;

  usize change_count() const { return _change_count; };
  void reset_change_count() { _change_count = 0; };

  virtual const SolutionBase& so_solution(usize objective) const = 0;

  virtual void adapt() {};

  virtual void clear() = 0;

  virtual const ArchiveFitnessBase& fitness() const = 0;

  virtual std::unique_ptr<ArchiveBase> clone() const = 0;

  /// Updates the archive with the solution and returns whether the solution was
  /// accepted into the archive.
  bool update(const SolutionBase& solution,
              bool strict,               // If false, returns true for solutions that are
                                         // non-dominated by the archive but not accepted
              bool check_synched = true  // Only consider a solution accepted if it
                                         // also is accepted by all synched archives
  );

  void sync_with(ArchiveBase& other) { synched_archives.push_back(other); };
  void unsync_all() { synched_archives.clear(); };

  bool dominates(const SolutionBase& solution, bool strict) const;
  bool covers(const ArchiveBase& other) const;

  inline const SolutionBase& random_solution(Rng& rng) const  {
    __goblin_runtime_assert(!empty());
    return operator[](std::uniform_int_distribution<usize>(0, size() - 1)(rng));
  };

  virtual ~ArchiveBase() {};

 protected:
  virtual std::tuple<bool, bool> update_archive(const SolutionBase& solution, bool strict) = 0;

  usize _change_count = 0;
  std::vector<std::reference_wrapper<ArchiveBase>> synched_archives;
};

class UnboundedArchive : public ArchiveBase {
 public:
  UnboundedArchive(const ArchiveFitnessBase& fitness) : _fitness(fitness) {};

  std::unique_ptr<ArchiveBase> clone() const override final { return std::make_unique<UnboundedArchive>(*this); };

  usize size() const override final { return _solutions.size(); }
  const SolutionBase& operator[](usize idx) const override final { return _solutions[idx]; }
  SolutionBase& unsafe_at(usize idx) override final { return _solutions[idx]; }

  const SolutionBase& so_solution(usize objective) const override final {
    __goblin_runtime_assert(objective < _so_solutions.size());
    return _so_solutions[objective];
  }

  void clear() override final {
    _so_solutions.clear();
    _solutions.clear();
  };

  std::tuple<bool, bool> update_archive(const SolutionBase& solution, bool strict) override final;
  const ArchiveFitnessBase& fitness() const override final { return _fitness; };

 private:
  // is_so_elite, is_dominated
  std::tuple<bool, bool> update_so_solutions(const SolutionBase& solution);

  DefaultSolutionSet _so_solutions;
  DefaultSolutionSet _solutions;
  const ArchiveFitnessBase& _fitness;
};

class AdaptiveGridArchive : public ArchiveBase {
 public:
  AdaptiveGridArchive(const ArchiveFitnessBase& fitness,
                      usize capacity,
                      float max_deviation = 0.25,
                      usize max_iterations = 25,
                      std::optional<CType> initial_discretization = std::nullopt)
      : _fitness(fitness),
        _capacity(capacity),
        _max_deviation(max_deviation),
        _c_min(static_cast<usize>(capacity * (1.0 - max_deviation))),
        _c_max(static_cast<usize>(capacity * (1.0 + max_deviation))),
        _max_iterations(max_iterations),
        // this is the maximum resolution of binary search with 25 steps...
        _max_resolution(std::pow(2.0, max_iterations)),
        _initial_discretization(initial_discretization) {
    __goblin_runtime_assert(capacity > 0);
    __goblin_runtime_assert(max_deviation > 0.0);
    __goblin_runtime_assert(!initial_discretization.has_value() || initial_discretization.value() > CType(0.0));
  };

  std::unique_ptr<ArchiveBase> clone() const override final { return std::make_unique<AdaptiveGridArchive>(*this); };

  usize size() const override final { return _solutions.size(); };
  const SolutionBase& operator[](usize idx) const override final { return _solutions[idx]; };
  SolutionBase& unsafe_at(usize idx) override final { return _solutions[idx]; };

  void clear() override final {
    _so_solutions.clear();
    _solutions.clear();
  };

  const SolutionBase& so_solution(usize objective) const override final { return _so_solutions[objective]; };

  std::tuple<bool, bool> update_archive(const SolutionBase& solution, bool strict) override final;

  void adapt() override final;

  const ArchiveFitnessBase& fitness() const override final { return _fitness; };

 private:
  bool same_box(const SolutionBase& lhs, const SolutionBase& rhs);
  bool update_so_solutions(const SolutionBase& solution);

  const ArchiveFitnessBase& _fitness;
  [[maybe_unused]] usize _capacity;
  [[maybe_unused]] float _max_deviation;
  usize _c_min;
  usize _c_max;
  usize _max_iterations;
  CType _max_resolution;

  std::optional<CType> _initial_discretization;
  std::vector<CType> _discretization;

  DefaultSolutionSet _so_solutions;
  DefaultSolutionSet _solutions;
};

};  // namespace goblin

#endif /* _GOBLIN_LIB_ARCHIVE_H */
