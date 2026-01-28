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

// TODO make quality virtual to allow arbitrary quality types?
// -> but then pointers are needed for everything and fitness types should downcast to their fitness type...
// -> performance issue?

class QualityBase {
 public:

 virtual std::unique_ptr<QualityBase> clone() const = 0;

 virtual ~QualityBase() = default;
};

/// Something that describes how good a solution is
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

class FitnessBase {
 public:
  virtual usize num_objectives() const = 0;

  virtual Ordering cmp(const Quality& lhs, const Quality& rhs, std::optional<usize> objective) const = 0;

  virtual CType distance(const Quality& lhs, const Quality& rhs, std::optional<usize> objective) const = 0;

  virtual void log_header(std::ostream& os) const = 0;

  virtual void log(std::ostream& os, const Quality& quality) const = 0;

  virtual std::string format(const Quality& quality) const {
    std::stringstream ss;
    log(ss, quality);
    return ss.str();
  };

  virtual ~FitnessBase() = default;
};

class ArchiveFitnessBase : public FitnessBase {
 public:
  virtual Quality worst() const = 0;

  virtual ~ArchiveFitnessBase() = default;
};

class MOFitness final : public ArchiveFitnessBase {
 public:
  MOFitness() = delete;
  MOFitness(usize num_objectives, bool minimize = true, CType epsilon = 0.0)
      : _num_objectives(num_objectives), _epsilon(epsilon), _minimize(minimize) {};

  void log_header(std::ostream& os) const override final { os << "objectives,constraint_value"; };

  void log(std::ostream& os, const Quality& quality) const override final {
    os << "\"[";
    for (usize i = 0; i < _num_objectives; i++) {
      if (i > 0) {
        os << ',';
      }
      os << quality.objectives(i);
    }
    os << "]\"," << quality.constraint_value;
  };

  std::string format(const Quality& quality) const override final {
    std::stringstream ss;
    log(ss, quality);
    return ss.str();
  };

  usize num_objectives() const override final { return _num_objectives; };

  Ordering cmp(const Quality& lhs,
               const Quality& rhs,
               std::optional<usize> objective = std::nullopt) const override final {
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

  CType distance(const Quality& lhs,
                 const Quality& rhs,
                 std::optional<usize> objective = std::nullopt) const override final {
    CType dist;
    if (objective.has_value()) {
      dist = distance(lhs.objectives(objective.value()), rhs.objectives(objective.value()));
    } else {
      dist = (lhs.objectives - rhs.objectives).norm();
    }
    return isna(dist) ? std::numeric_limits<CType>::infinity() : dist;
  };

  Quality worst() const override final {
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

};  // namespace goblin

#endif /* _GOBLIN_LIB_FITNESS_H */
