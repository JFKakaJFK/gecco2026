#ifndef _GOBLIN_LIB_ARCHIVE_H
#define _GOBLIN_LIB_ARCHIVE_H

#pragma once

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

  virtual const Fitness& fitness() const = 0;

  virtual std::unique_ptr<ArchiveBase> clone() const = 0;

  /// Updates the archive with the solution and returns whether the solution was
  /// accepted into the archive.
  bool update(const SolutionBase& solution,
              bool strict,               // If false, returns true for solutions that are
                                         // non-dominated by the archive but not accepted
              bool check_synched = true  // Only consider a solution accepted if it
                                         // also is accepted by all synched archives
  ) {
    auto [accepted, accepted_strict] = update_archive(solution, strict);
    // the assumption is that the synched archives always are at least as //
    // "good" as this archive (after all, we update them with each improvement
    // on this archive!), so if the solution is not improving // this archive
    // there is no reason to update the synched ones
    if (accepted_strict) {
      _change_count++;
      for (auto& a : synched_archives) {
        bool a_accepted = a.get().update(solution, strict);
        if (check_synched) {
          // The synching is only used to keep a local archive to detect
          // stagnation GOMEA typically uses the global archive for testing if
          // something is an improvement and for FI
          accepted &= a_accepted;
        }
      }
    }
    return accepted;
  };

  void sync_with(ArchiveBase& other) { synched_archives.push_back(other); };

  void unsync_all() { synched_archives.clear(); };

  bool dominates(const SolutionBase& solution, bool strict) const {
    if (!empty()) {
      for (usize i = 0; i < fitness().num_objectives(); i++) {
        auto o = fitness().cmp(so_solution(i).quality(), solution.quality());
        if (o == Ordering::Better || (!strict && o == Ordering::Equal)) {
          return true;
        }
      }

      for (usize i = 0; i < size(); i++) {
        auto o = fitness().cmp(operator[](i).quality(), solution.quality());
        if (o == Ordering::Better || (!strict && o == Ordering::Equal)) {
          return true;
        }
      }
    }

    return false;
  }

  bool covers(const ArchiveBase& other) const {
    if (other.empty())
      return true;  // any archive covers the empty archive
    if (empty()) {
      return false;  // an empty archive covers nothing
    } else {
      // at this point we return false if other contains a solution not matched
      // by this, otherwise the archive other is indeed covered by this

      // neither archive is empty, so the so solutions must be set...
      for (usize obj = 0; obj < fitness().num_objectives(); obj++) {
        auto o = fitness().cmp(so_solution(obj).quality(), other.so_solution(obj).quality(), obj);
        if (o == Ordering::NonDominated || o == Ordering::Worse) {
          return false;
        }
      }

      for (usize other_idx = 0; other_idx < other.size(); other_idx++) {
        bool covered = false;
        for (usize idx = 0; idx < size(); idx++) {
          auto o = fitness().cmp(operator[](idx).quality(), other[other_idx].quality());
          if (o == Ordering::Better || o == Ordering::Equal) {
            covered = true;
            break;
          } else if (o == Ordering::Worse) {
            // Both archives only contain non-dominated solutions,
            // so if a solution s in this archive is dominated by other AND
            // s is non-dominated in this archive, at least one solution in
            // other cannot be matched by this archive
            return false;
          }
        }
        if (!covered)
          return false;
      }

      return true;
    }
  };

  const SolutionBase& random_solution(Rng& rng) const {
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
  UnboundedArchive(const Fitness& fitness) : _fitness(fitness) {};

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

  std::tuple<bool, bool> update_archive(const SolutionBase& solution, bool strict) override final {
    std::vector<usize> sorted_dominations;
    auto [is_so_elite, is_dominated] = update_so_solutions(solution);
    if (is_dominated) {
      return std::make_tuple(false, false);
    }
    for (usize i = 0; i < _solutions.size(); i++) {
      auto o = fitness().cmp(solution.quality(), _solutions[i].quality(), std::nullopt);
      if (o == Ordering::Better) {
        sorted_dominations.push_back(i);
      } else if (o == Ordering::Worse) {
        return std::make_tuple(false, false);
      } else if (o == Ordering::Equal) {
        // TODO only accept if not in the set already
        return std::make_tuple(!strict, false);
      }
    }

    if (!sorted_dominations.empty()) {
      //   std::span<usize> si(sorted_dominations.data(),
      //   sorted_dominations.size());
      // _solutions.remove_indices_sorted(si); //sorted_dominations);
      for (usize i = sorted_dominations.size(); i > 0; i--) {
        _solutions.remove_at(sorted_dominations[i - 1]);
      }
    }

    _solutions.add(solution);

    return std::make_tuple(true, true);
  };

  const Fitness& fitness() const override final { return _fitness; };

 private:
  // is_so_elite, is_dominated
  std::tuple<bool, bool> update_so_solutions(const SolutionBase& solution) {
    if (_so_solutions.empty()) {
      _so_solutions.reserve(fitness().num_objectives());
      for (usize obj = 0; obj < fitness().num_objectives(); obj++) {
        _so_solutions.add(solution);
      }
      return std::make_tuple(true, false);
    } else {
      for (usize obj = 0; obj < fitness().num_objectives(); obj++) {
        if (fitness().cmp(solution.quality(), _so_solutions[obj].quality(), obj) == Ordering::Better) {
          _so_solutions[obj] = solution;
          return std::make_tuple(true, false);
        }
        if (fitness().cmp(solution.quality(), _so_solutions[obj].quality(), std::nullopt) == Ordering::Worse) {
          return std::make_tuple(false, true);
        }
      }
      return std::make_tuple(false, false);
    }
  };

  DefaultSolutionSet _so_solutions;
  DefaultSolutionSet _solutions;
  const Fitness& _fitness;
};

class AdaptiveGridArchive : public ArchiveBase {
 public:
  AdaptiveGridArchive(const Fitness& fitness,
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

  std::tuple<bool, bool> update_archive(const SolutionBase& solution, bool strict) override final {
    std::vector<usize> sorted_dominations;
    bool so_elite = update_so_solutions(solution);
    for (usize i = 0; i < _solutions.size(); i++) {
      auto o = _fitness.cmp(solution.quality(), _solutions[i].quality(), std::nullopt);
      if (o == Ordering::Better) {
        sorted_dominations.push_back(i);
      } else if (o == Ordering::Worse) {
        return std::make_tuple(false, false);
      } else if (o == Ordering::Equal || same_box(solution, _solutions[i])) {
        // same or non-dominated and same box

        if (so_elite || !sorted_dominations.empty()) {
          // both cases are non-standard
          // - we always keep so elite improvements, even if in the same box as
          // some other solution
          // - if the box is occupied but the new solution already dominates
          // another we keep the new one
          sorted_dominations.push_back(i);
        } else {
          bool replace = false;

          // if discrete: check if diversity in parameter space is added
          if (solution.num_discrete() > 0) {
            usize max_dist = std::numeric_limits<usize>::max();
            usize max_other_dist = std::numeric_limits<usize>::max();
            usize dist, other_dist;
            for (usize j = 0; j < _solutions.size(); j++) {
              if (j == i)
                continue;
              dist = 0;
              other_dist = 0;
              for (usize k = 0; k < solution.num_discrete(); k++) {
                if (solution.discrete_active()[k] &&
                    solution.discrete_values()[k] != _solutions[j].discrete_values()[k]) {
                  dist++;
                }
                if (_solutions[i].discrete_active()[k] &&
                    _solutions[i].discrete_values()[k] != _solutions[j].discrete_values()[k]) {
                  other_dist++;
                }
              }
              if (dist < max_dist) {
                max_dist = dist;
              }
              if (other_dist < max_other_dist) {
                max_other_dist = dist;
              }
            }
            if (max_dist > max_other_dist) {
              replace = true;
            }
          }

          if (replace) {
            sorted_dominations.push_back(i);
          } else {
            return std::make_tuple(!strict, false);
          }
        }
      }
    }

    if (!sorted_dominations.empty()) {
      // _solutions.remove_sorted_indices(sorted_dominations);
      for (usize i = sorted_dominations.size(); i > 0; i--) {
        _solutions.remove_at(sorted_dominations[i - 1]);
      }
    }

    _solutions.add(solution);

    return std::make_tuple(true, true);
  };

  void adapt() override final {
    if (size() > _c_max && _max_iterations > 0) {
      // get the maximum finite distance to the best so solution for each
      // objective
      std::vector<CType> max_objective_dist(_fitness.num_objectives(), 0.0);
      for (usize i = 0; i < _solutions.size(); i++) {
        for (usize obj = 0; obj < _fitness.num_objectives(); obj++) {
          auto d = _fitness.distance(_so_solutions[obj].quality(), _solutions[i].quality(), obj);
          if (!isna(d) && d > max_objective_dist[obj]) {
            max_objective_dist[obj] = d;
          }
        }
      }

      DefaultSolutionSet backup = _solutions;
      usize cc = _change_count;

      CType low = 1.0;
      CType high = _max_resolution;
      CType prev = 0.0;

      if (_discretization.empty()) {
        _discretization.resize(_fitness.num_objectives(), 0.0);
      }

      for (usize i = 0; i < _max_iterations; i++) {
        CType mid = 0.5 * (low + high);

        if (prev > 0 && prev == mid)
          break;
        prev = mid;

        for (usize obj = 0; obj < _fitness.num_objectives(); obj++) {
          _discretization[obj] = max_objective_dist[obj] / mid;
          // shouldn't matter - we don't divide by it and semantically it just
          // means "not identical" is a different box assert(discretization[obj]
          // > 0.0);
        }

        // clear the archive and add everything back with the new discretization
        _solutions.clear();
        for (usize obj = 0; obj < _fitness.num_objectives(); obj++) {
          update(_so_solutions[obj], true);
        }
        for (usize j = 0; j < backup.size(); j++) {
          update(backup[j], true);
        }

        if (_solutions.size() < _c_min) {
          low = mid;
        } else if (_solutions.size() == _c_min) {
          break;
        } else {
          high = mid;
        }
      }

      // change count should not be affected by re-discretizing
      _change_count = cc;
    }
  };

  const Fitness& fitness() const override final { return _fitness; };

 private:
  bool same_box(const SolutionBase& lhs, const SolutionBase& rhs) {
    if (_discretization.empty() && _initial_discretization.has_value()) {
      _discretization.resize(_fitness.num_objectives(), _initial_discretization.value());
    }
    if (!_discretization.empty()) {
      __goblin_runtime_assert(_discretization.size() == _fitness.num_objectives());
      for (usize obj = 0; obj < _fitness.num_objectives(); obj++) {
        // The original code uses floor(f1 / discretization) == floor(f2 /
        // discretization), but since I only have relative comparisons, this is
        // as close as it gets and semantically almost the same. Conceptually,
        // the original code uses an infinite grid and here the grid cells are
        // centered around each solution
        if (_fitness.distance(lhs.quality(), rhs.quality(), obj) > _discretization[obj]) {
          return false;
        }
      }
      return true;
    }
    return false;
  };

  bool update_so_solutions(const SolutionBase& solution) {
    if (_so_solutions.empty()) {
      _so_solutions.reserve(fitness().num_objectives());
      for (usize obj = 0; obj < fitness().num_objectives(); obj++) {
        _so_solutions.add(solution);
      }
      return true;
    } else {
      for (usize obj = 0; obj < fitness().num_objectives(); obj++) {
        if (fitness().cmp(solution.quality(), _so_solutions[obj].quality(), obj) == Ordering::Better) {
          _so_solutions[obj] = solution;
          return true;
        }
      }
    }
    return false;
  };

  const Fitness& _fitness;
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
