#pragma once
#ifndef _GOBLIN_LIB_FITNESS_H
#define _GOBLIN_LIB_FITNESS_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>

#include <Eigen/Dense>

#include "goblin/lib/ordering.h"
#include "goblin/lib/types.h"

namespace goblin {

class MOFitness {
 public:
  // TODO remove fitness type and add the comparison, distance and
  // num_objectives to the quality
  // -> Can do runtime polymorphism again
  // !PROBLEM: how to have different fitneses for the use in the archive, and
  // how to init?
  // -> quality could expose multiple comparisons
  // -> init to empty, and put initializing on the function?
  class Quality {
   public:
    Quality() = delete;

    Vec<CType> objectives;
    CType constraint_value;

   private:
    friend class MOFitness;  // Allow constructing qualities
    Quality(Vec<CType> objectives, CType constraint_value = CType(0.0))
        : objectives(std::move(objectives)), constraint_value(std::max(CType(0.0), constraint_value)) {};
  };

  MOFitness() = delete;
  MOFitness(usize num_objectives, bool minimize = true, CType epsilon = 0.0)
      : _num_objectives(num_objectives), _epsilon(epsilon), _minimize(minimize) {};

  void log_header(std::ostream& os) const { os << "objectives,constraint_value"; };

  void log(std::ostream& os, const Quality& quality) const {
    os << "\"[";
    for (usize i = 0; i < _num_objectives; i++) {
      if (i > 0) {
        os << ',';
      }
      os << quality.objectives(i);
    }
    os << "]\"," << quality.constraint_value;
  };

  std::string format(const Quality& quality) const {
    std::stringstream ss;
    log(ss, quality);
    return ss.str();
  };

  inline usize num_objectives() const { return _num_objectives; };

  [[nodiscard]]
  inline Ordering cmp(const Quality& lhs, const Quality& rhs, std::optional<usize> objective = std::nullopt) const {
    // Constraints are always minimized
    Ordering o = cmp(lhs.constraint_value, rhs.constraint_value, _epsilon, true);

    if (o == Ordering::Equal || o == Ordering::NonDominated) {
      if (objective.has_value()) {
        o = cmp(lhs.objectives(objective.value()), rhs.objectives(objective.value()), _epsilon, _minimize);
      } else {
        for (usize i = 0; i < _num_objectives && o != Ordering::NonDominated; i++) {
          o = o | cmp(lhs.objectives(i), rhs.objectives(i), _epsilon, _minimize);
        }
      }
    }
    return o;
  };

  [[nodiscard]]
  inline CType distance(const Quality& lhs, const Quality& rhs, std::optional<usize> objective = std::nullopt) const {
    if (objective.has_value()) {
      return distance(lhs.objectives(objective.value()), rhs.objectives(objective.value()));
    } else {
      return (lhs.objectives - rhs.objectives).norm();
    }
  };

  Quality worst() const {
    return Quality(Vec<CType>::Constant(_num_objectives, (_minimize ? CType(1.0) : CType(-1.0)) *
                                                             std::numeric_limits<CType>().infinity()),
                   std::numeric_limits<CType>().infinity());
  };

 private:
  inline Ordering cmp(const CType& lhs, const CType& rhs, double epsilon, bool minimize) const noexcept {
    bool lhs_finite = !isna(lhs);
    bool rhs_finite = !isna(rhs);
    if (!lhs_finite && !rhs_finite)
      return Ordering::NonDominated;
    if (!lhs_finite)
      return Ordering::Worse;
    if (!rhs_finite)
      return Ordering::Better;
    if (minimize) {
      if (lhs + epsilon < rhs)
        return Ordering::Better;
      if (lhs > rhs + epsilon)
        return Ordering::Worse;
    } else {
      if (lhs > rhs + epsilon)
        return Ordering::Better;
      if (lhs + epsilon < rhs)
        return Ordering::Worse;
    }
    return Ordering::Equal;
  };

  inline CType distance(const CType& lhs, const CType& rhs) const noexcept { return std::abs(lhs - rhs); };

  usize _num_objectives;
  CType _epsilon;
  bool _minimize;
};

// Shoddy compile time polymorphism - replace these with compatible classes to
// use different fitnesses
using Fitness = MOFitness;
using Quality = Fitness::Quality;

};  // namespace goblin

#endif /* _GOBLIN_LIB_FITNESS_H */
