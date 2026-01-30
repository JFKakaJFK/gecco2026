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

/// Something that describes how good a solution is
class QualityBase {
 public:
  virtual std::unique_ptr<QualityBase> clone() const = 0;

  virtual ~QualityBase() = default;

  template <typename T>
  const T& as() const {
#ifndef NDEBUG
    auto p = dynamic_cast<const T*>(this);
    assert(p != nullptr && "Quality type mismatch");
#endif
    return static_cast<const T&>(*this);
  }
  template <typename T>
  T& as() {
#ifndef NDEBUG
    auto p = dynamic_cast<T*>(this);
    assert(p != nullptr && "Quality type mismatch");
#endif
    return static_cast<T&>(*this);
  }
};

class FitnessBase {
 public:
  virtual usize num_objectives() const = 0;

  virtual Ordering cmp(const QualityBase& lhs, const QualityBase& rhs, std::optional<usize> objective) const = 0;

  virtual CType distance(const QualityBase& lhs, const QualityBase& rhs, std::optional<usize> objective) const = 0;

  virtual void log_header(std::ostream& os) const = 0;

  virtual void log(std::ostream& os, const QualityBase& quality) const = 0;

  virtual std::string format(const QualityBase& quality) const {
    std::stringstream ss;
    log(ss, quality);
    return ss.str();
  };

  virtual ~FitnessBase() = default;
};

class ArchiveFitnessBase : public FitnessBase {
 public:
  virtual std::unique_ptr<QualityBase> worst() const = 0;

  virtual ~ArchiveFitnessBase() = default;
};

/// Something that describes how good a solution is
class MOQuality : public QualityBase {
 public:
  std::unique_ptr<QualityBase> clone() const override { return std::make_unique<MOQuality>(*this); };

  Vec<CType> objectives;
  CType constraint_value;
};

class MOFitness : public ArchiveFitnessBase {
 public:
  MOFitness() = delete;
  MOFitness(usize num_objectives, bool minimize = true, CType epsilon = 0.0)
      : _num_objectives(num_objectives), _epsilon(epsilon), _minimize(minimize) {};

  void log_header(std::ostream& os) const override final { os << "objectives,constraint_value"; };

  void log(std::ostream& os, const QualityBase& quality) const override final {
    const auto& q = quality.as<MOQuality>();
    os << "\"[";
    for (usize i = 0; i < _num_objectives; i++) {
      if (i > 0) {
        os << ',';
      }
      os << q.objectives(i);
    }
    os << "]\"," << q.constraint_value;
  };

  std::string format(const QualityBase& quality) const override final {
    std::stringstream ss;
    log(ss, quality);
    return ss.str();
  };

  usize num_objectives() const override final { return _num_objectives; };

  Ordering cmp(const QualityBase& lhs,
               const QualityBase& rhs,
               std::optional<usize> objective = std::nullopt) const override final {
    const auto& ql = lhs.as<MOQuality>();
    const auto& qr = rhs.as<MOQuality>();
    // Constraints are always minimized
    Ordering o = cmp(ql.constraint_value, qr.constraint_value, _epsilon, true);

    if (o == Ordering::Equal || o == Ordering::NonDominated) {
      if (objective.has_value()) {
        o = cmp(ql.objectives(objective.value()), qr.objectives(objective.value()), _epsilon, _minimize);
      } else {
        for (usize i = 0; i < _num_objectives && o != Ordering::NonDominated; i++) {
          o = o | cmp(ql.objectives(i), qr.objectives(i), _epsilon, _minimize);
        }
      }
    }
    return o;
  };

  CType distance(const QualityBase& lhs,
                 const QualityBase& rhs,
                 std::optional<usize> objective = std::nullopt) const override final {
    const auto& ql = lhs.as<MOQuality>();
    const auto& qr = rhs.as<MOQuality>();
    CType dist;
    if (objective.has_value()) {
      dist = distance(ql.objectives(objective.value()), qr.objectives(objective.value()));
    } else {
      dist = (ql.objectives - qr.objectives).norm();
    }
    return isna(dist) ? std::numeric_limits<CType>::infinity() : dist;
  };

  virtual std::unique_ptr<QualityBase> worst() const override {
    const CType inf = std::numeric_limits<CType>().infinity();
    auto q = std::make_unique<MOQuality>();
    q->objectives = Vec<CType>::Constant(_num_objectives, (_minimize ? inf : -inf));
    q->constraint_value = inf;
    return q;
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
