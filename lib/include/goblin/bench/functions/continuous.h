#pragma once
#ifndef _GOBLIN_BENCH_FUNCTIONS_CONTINUOUS_H
#define _GOBLIN_BENCH_FUNCTIONS_CONTINUOUS_H

#include <cassert>
#include <numbers>
#include <bit>
#include <stdexcept>
#include <algorithm>

#include "goblin/bench/functions.h"
#include "goblin/lib/assert.h"
#include "goblin/lib/fitness.h"
#include "goblin/lib/archive.h"

namespace goblin {

class Sphere final : public ObjectiveBase {
 public:
  Sphere(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return 0; };
  usize num_continuous() const override final { return dims; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    continuous_active.fill(true);
    return std::make_tuple(continuous_values.array().pow(2).sum(), 0.0);
  };

  std::tuple<CType, CType> evaluate_partial(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Array<BType>> discrete_active,
                                            RefS<Array<BType>> continuous_active,
                                            CRefS<Vec<DType>> parent_discrete_values,
                                            CRefS<Vec<CType>> parent_continuous_values,
                                            CRefS<Array<BType>> parent_discrete_active,
                                            CRefS<Array<BType>> parent_continuous_active,
                                            const CType parent_objective_value,
                                            const CType parent_constraint_value,
                                            const std::span<const usize>& discrete_indices,
                                            const std::span<const usize>& continuous_indices) override final {
    continuous_active.fill(true);
    return std::make_tuple(parent_objective_value + continuous_values(continuous_indices).array().pow(2).sum() -
                               parent_continuous_values(continuous_indices).array().pow(2).sum(),
                           0.0);
  };

 private:
  usize dims;
};

class Rosenbrock final : public ObjectiveBase {
 public:
  Rosenbrock(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return 0; };
  usize num_continuous() const override final { return dims; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    continuous_active.fill(true);
    CType ov = 0.0;
    for (usize i = 0; i < dims - 1; i++) {
      auto x = continuous_values(i);
      auto y = continuous_values(i + 1);
      ov += 100 * (y - x * x) * (y - x * x) + (1.0 - x) * (1.0 - x);
    }

    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
};

class Rastrigin final : public ObjectiveBase {
 public:
  Rastrigin(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return 0; };
  usize num_continuous() const override final { return dims; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    continuous_active.fill(true);
    CType ov = CType(10.0) * static_cast<CType>(dims)

               + continuous_values.norm() -
               CType(10.0) * (continuous_values.array() * CType(2.0 * std::numbers::pi)).cos().sum();

    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
};

class Griewank final : public ObjectiveBase {
 public:
  Griewank(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return 0; };
  usize num_continuous() const override final { return dims; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    continuous_active.fill(true);

    CType prod = 1.0;
    for (usize i = 0; i < dims; i++) {
      prod *= std::cos(continuous_values(i) / std::sqrt(static_cast<CType>(i + 1)));
    }
    CType ov = continuous_values.norm() / CType(4000.0) - prod + CType(1.0);
    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
};

class Ellipsoid final : public ObjectiveBase {
 public:
  Ellipsoid(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return 0; };
  usize num_continuous() const override final { return dims; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    continuous_active.fill(true);

    CType ov = 0.0;
    for (usize i = 0; i < dims; i++) {
      ov += std::pow(10.0, 6.0 * static_cast<CType>(i) / static_cast<CType>(dims - 1)) * continuous_values(i) *
            continuous_values(i);
    }
    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
};

class CirclesInASquare final : public ObjectiveBase {
 public:
  CirclesInASquare(usize ndims) : dims(ndims) { __goblin_runtime_assert(ndims > 2 && ndims % 2 == 0); };

  usize num_discrete() const override final { return 0; };
  usize num_continuous() const override final { return dims; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    continuous_active.fill(true);
    CType min_dist = 1e308, cv = 0;
    for (usize i = 0; i < dims; i += 2) {
      auto circle_i = continuous_values(Eigen::seqN(i, 2));
      cv += (circle_i.array().max(1.0) - 1).sum() - circle_i.array().min(0.0).sum();
      for (usize j = i + 2; j < dims; j += 2) {
        auto circle_j = continuous_values(Eigen::seqN(j, 2));
        CType dist = (circle_i - circle_j).norm();
        min_dist = std::min(min_dist, dist);
      }
    }

    return std::make_tuple(-min_dist, cv);
  };

 private:
  usize dims;
};

// TODO for ZDT problems: add domains & fronts/success criterion?

/// ZDT1 from https://doi.org/10.1162/106365600568202
class ZDT1 : public MOFunctionBase {
  usize dims;
  usize num_samples;

 public:
  ZDT1(usize dims = 30, usize pareto_front_samples = 100) : dims(dims), num_samples(pareto_front_samples) {
    if (dims < 2) {
      throw std::runtime_error("ZDT1 requires at least 2 parameters.");
    }
  };

  usize num_objectives() const override final { return 2; }
  usize num_discrete() const override final { return 0; }
  usize num_continuous() const override final { return dims; }

  void evaluate(SolutionBase& solution) override final {
    solution.discrete_active().fill(false);
    solution.continuous_active().fill(true);
    auto& q = solution.quality_as<MOQuality>();

    q.constraint_value = 0.0;
    q.objectives(0) = solution.continuous_values()(0);

    CType g = 0.0;
    for (usize i = 1; i < dims; i++) {
      g += solution.continuous_values()(i) / static_cast<CType>(dims - 1);
    }
    g = 1.0 + 9.0 * g;

    CType h = 1.0 - std::sqrt(solution.continuous_values()(0) / g);

    q.objectives(1) = g * h;
  }

  std::optional<CRef<Vec<CType>>> continuous_lower_bounds() const override final { return Vec<CType>::Zero(dims); };
  std::optional<CRef<Vec<CType>>> continuous_upper_bounds() const override final { return Vec<CType>::Ones(dims); };

  std::optional<std::shared_ptr<ArchiveBase>> pareto_front() const override final {
    CType x0 = 0.0;
    CType step = 1.0 / static_cast<CType>(num_samples - 1);

    MOFitness f(2);
    auto pf = std::make_shared<UnboundedArchive>(f);
    for (usize i = 0; i < num_samples; i++) {
      auto q = f.worst();
      auto& o = q->as<MOQuality>();
      o.objectives(0) = x0;
      o.objectives(1) = 1.0 - std::sqrt(x0);
      o.constraint_value = 0.0;
      pf->update(Solution(std::move(q), std::nullopt, Vec<CType>::Zero(dims)));

      x0 += step;
    }
    return pf;
  }
};

/// ZDT2 from https://doi.org/10.1162/106365600568202
class ZDT2 : public MOFunctionBase {
  usize dims;
  usize num_samples;

 public:
  ZDT2(usize dims = 30, usize pareto_front_samples = 100) : dims(dims), num_samples(pareto_front_samples) {
    if (dims < 2) {
      throw std::runtime_error("ZDT2 requires at least 2 parameters.");
    }
  };

  usize num_objectives() const override final { return 2; }
  usize num_discrete() const override final { return 0; }
  usize num_continuous() const override final { return dims; }

  void evaluate(SolutionBase& solution) override final {
    solution.discrete_active().fill(false);
    solution.continuous_active().fill(true);
    auto& q = solution.quality_as<MOQuality>();

    q.constraint_value = 0.0;
    q.objectives(0) = solution.continuous_values()(0);

    CType g = 0.0;
    for (usize i = 1; i < dims; i++) {
      g += solution.continuous_values()(i) / static_cast<CType>(dims - 1);
    }
    g = 1.0 + 9.0 * g;

    CType h = solution.continuous_values()(0) / g;
    h = 1.0 - h * h;

    q.objectives(1) = g * h;
  }

  std::optional<CRef<Vec<CType>>> continuous_lower_bounds() const override final { return Vec<CType>::Zero(dims); };
  std::optional<CRef<Vec<CType>>> continuous_upper_bounds() const override final { return Vec<CType>::Ones(dims); };

  std::optional<std::shared_ptr<ArchiveBase>> pareto_front() const override final {
    CType x0 = 0.0;
    CType step = 1.0 / static_cast<CType>(num_samples - 1);

    MOFitness f(2);
    auto pf = std::make_shared<UnboundedArchive>(f);
    for (usize i = 0; i < num_samples; i++) {
      auto q = f.worst();
      auto& o = q->as<MOQuality>();
      o.objectives(0) = x0;
      o.objectives(1) = 1.0 - x0 * x0;
      o.constraint_value = 0.0;
      pf->update(Solution(std::move(q), std::nullopt, Vec<CType>::Zero(dims)));

      x0 += step;
    }
    return pf;
  }
};

/// ZDT3 from https://doi.org/10.1162/106365600568202
class ZDT3 : public MOFunctionBase {
  usize dims;
  usize num_samples;

 public:
  ZDT3(usize dims = 30, usize pareto_front_samples = 100) : dims(dims), num_samples(pareto_front_samples) {
    if (dims < 2) {
      throw std::runtime_error("ZDT3 requires at least 2 parameters.");
    }
  };

  usize num_objectives() const override final { return 2; }
  usize num_discrete() const override final { return 0; }
  usize num_continuous() const override final { return dims; }

  void evaluate(SolutionBase& solution) override final {
    solution.discrete_active().fill(false);
    solution.continuous_active().fill(true);
    auto& q = solution.quality_as<MOQuality>();

    q.constraint_value = 0.0;
    q.objectives(0) = solution.continuous_values()(0);

    CType g = 0.0;
    for (usize i = 1; i < dims; i++) {
      g += solution.continuous_values()(i) / static_cast<CType>(dims - 1);
    }
    g = 1.0 + 9.0 * g;

    CType h = solution.continuous_values()(0) / g;
    h = 1.0 - std::sqrt(h) - h * std::sin(10.0 * std::numbers::pi * solution.continuous_values()(0));

    q.objectives(1) = g * h;
  }

  std::optional<CRef<Vec<CType>>> continuous_lower_bounds() const override final { return Vec<CType>::Zero(dims); };
  std::optional<CRef<Vec<CType>>> continuous_upper_bounds() const override final { return Vec<CType>::Ones(dims); };

  std::optional<std::shared_ptr<ArchiveBase>> pareto_front() const override final {
    CType x0 = 0.0;
    CType step = 1.0 / static_cast<CType>(num_samples - 1);

    MOFitness f(2);
    auto pf = std::make_shared<UnboundedArchive>(f);
    for (usize i = 0; i < num_samples; i++) {
      auto q = f.worst();
      auto& o = q->as<MOQuality>();
      o.objectives(0) = x0;
      o.objectives(1) = 1.0 - std::sqrt(x0) - x0 * std::sin(10.0 * std::numbers::pi * x0);
      o.constraint_value = 0.0;
      pf->update(Solution(std::move(q), std::nullopt, Vec<CType>::Zero(dims)));

      x0 += step;
    }
    return pf;
  }
};

// TODO fix bug (crashes sometimes...)
/// ZDT4 from https://doi.org/10.1162/106365600568202
class ZDT4 : public MOFunctionBase {
  usize dims;
  usize num_samples;

 public:
  ZDT4(usize dims = 10, usize pareto_front_samples = 100) : dims(dims), num_samples(pareto_front_samples) {
    if (dims < 2) {
      throw std::runtime_error("ZDT4 requires at least 2 parameters.");
    }
  };

  usize num_objectives() const override final { return 2; }
  usize num_discrete() const override final { return 0; }
  usize num_continuous() const override final { return dims; }

  void evaluate(SolutionBase& solution) override final {
    solution.discrete_active().fill(false);
    solution.continuous_active().fill(true);
    auto& q = solution.quality_as<MOQuality>();

    q.constraint_value = 0.0;
    q.objectives(0) = solution.continuous_values()(0);

    CType g = 1.0 + 10.0 * static_cast<CType>(dims - 1);
    for (usize i = 1; i < dims; i++) {
      g += solution.continuous_values()(i) * solution.continuous_values()(i) -
           10.0 * std::cos(4.0 * std::numbers::pi * solution.continuous_values()(i));
    }

    CType h = 1.0 - std::sqrt(solution.continuous_values()(0) / g);

    q.objectives(1) = g * h;
  }

  std::optional<CRef<Vec<CType>>> continuous_lower_bounds() const override final {
    Vec<CType> lb = Vec<CType>::Constant(dims, -5.0);
    lb(0) = 0.0;
    return lb;
  };
  std::optional<CRef<Vec<CType>>> continuous_upper_bounds() const override final {
    Vec<CType> ub = Vec<CType>::Constant(dims, 5.0);
    ub(0) = 1.0;
    return ub;
  };

  std::optional<std::shared_ptr<ArchiveBase>> pareto_front() const override final {
    CType x0 = 0.0;
    CType step = 1.0 / static_cast<CType>(num_samples - 1);

    MOFitness f(2);
    auto pf = std::make_shared<UnboundedArchive>(f);
    for (usize i = 0; i < num_samples; i++) {
      auto q = f.worst();
      auto& o = q->as<MOQuality>();
      o.objectives(0) = x0;
      o.objectives(1) = 1.0 - std::sqrt(x0);
      o.constraint_value = 0.0;
      pf->update(Solution(std::move(q), std::nullopt, Vec<CType>::Zero(dims)));

      x0 += step;
    }
    return pf;
  }
};

/// ZDT5 from https://doi.org/10.1162/106365600568202
class ZDT5 : public MOFunctionBase {
  usize dims;
  usize num_samples;

  u32 u(CType x, u32 num_bits) const {
    u64 bits;
    std::memcpy(&bits, &x, sizeof(bits));  // interpret the memory as bytes

    bits &= ((1ULL << 52) - 1);  // extract 52 fraction bits
    bits >>= 52 - num_bits;      // truncate to target number of bits

    return std::popcount(bits);  // count number of ones in the remaining bitstring
  };

  CType v(u32 num_ones) const { return num_ones < 5 ? 2.0 + static_cast<CType>(num_ones) : 1.0; };

 public:
  ZDT5(usize dims = 11, usize pareto_front_samples = 100) : dims(dims), num_samples(pareto_front_samples) {
    if (dims < 2) {
      throw std::runtime_error("ZDT5 requires at least 2 parameters.");
    }
  };

  usize num_objectives() const override final { return 2; }
  usize num_discrete() const override final { return 0; }
  usize num_continuous() const override final { return dims; }

  void evaluate(SolutionBase& solution) override final {
    solution.discrete_active().fill(false);
    solution.continuous_active().fill(true);
    auto& q = solution.quality_as<MOQuality>();

    q.constraint_value = 0.0;
    q.objectives(0) = 1.0 + static_cast<CType>(u(solution.continuous_values()(0), 30));

    CType g = 0.0;
    for (usize i = 1; i < dims; i++) {
      g += v(u(solution.continuous_values()(i), 5));
    }

    CType h = 1.0 / q.objectives(0);

    q.objectives(1) = g * h;
  }

  std::optional<CRef<Vec<CType>>> continuous_lower_bounds() const override final { return Vec<CType>::Zero(dims); };
  std::optional<CRef<Vec<CType>>> continuous_upper_bounds() const override final {
    Vec<CType> ub = Vec<CType>::Zero(dims);
    u64 bits;
    for (usize i = 0; i < dims; i++) {
      usize nbits = dims > 0 ? 5 : 30;

      std::memcpy(&bits, &ub(i), sizeof(bits));  // interpret the memory as bytes
      for (usize j = 0; j < nbits; j++) {
        bits |= 1ULL << (52 - j);  // set fraction bits to 1
      }
      std::memcpy(&ub(i), &bits, sizeof(CType));  // interpret the memory as CType
    }
    return ub;
  };

  std::optional<std::shared_ptr<ArchiveBase>> pareto_front() const override final {
    CType x0 = 0.0;
    CType step = 1.0 / static_cast<CType>(num_samples - 1);

    MOFitness f(2);
    auto pf = std::make_shared<UnboundedArchive>(f);
    for (usize i = 0; i < num_samples; i++) {
      auto q = f.worst();
      auto& o = q->as<MOQuality>();
      o.objectives(0) = 1.0 + static_cast<CType>(u(x0, 30));
      o.objectives(1) = 10.0 * (1.0 / o.objectives(0));
      o.constraint_value = 0.0;
      pf->update(Solution(std::move(q), std::nullopt, Vec<CType>::Zero(dims)));

      x0 += step;
    }
    return pf;
  }
};

/// ZDT6 from https://doi.org/10.1162/106365600568202
class ZDT6 : public MOFunctionBase {
  usize dims;
  usize num_samples;

 public:
  ZDT6(usize dims = 10, usize pareto_front_samples = 100) : dims(dims), num_samples(pareto_front_samples) {
    if (dims < 2) {
      throw std::runtime_error("ZDT6 requires at least 2 parameters.");
    }
  };

  usize num_objectives() const override final { return 2; }
  usize num_discrete() const override final { return 0; }
  usize num_continuous() const override final { return dims; }

  void evaluate(SolutionBase& solution) override final {
    solution.discrete_active().fill(false);
    solution.continuous_active().fill(true);
    auto& q = solution.quality_as<MOQuality>();

    q.constraint_value = 0.0;
    q.objectives(0) = 1.0 - std::exp(-4.0 * solution.continuous_values()(0)) *
                                std::pow(std::sin(6.0 * std::numbers::pi * solution.continuous_values()(0)), 6);

    CType g = 0.0;
    for (usize i = 1; i < dims; i++) {
      g += solution.continuous_values()(i);
    }
    g = 1.0 + 9.0 * std::pow(g / static_cast<double>(dims - 1), 0.25);

    CType h = q.objectives(0) / g;
    h = 1.0 - h * h;

    q.objectives(1) = g * h;
  }

  std::optional<CRef<Vec<CType>>> continuous_lower_bounds() const override final { return Vec<CType>::Zero(dims); };
  std::optional<CRef<Vec<CType>>> continuous_upper_bounds() const override final { return Vec<CType>::Ones(dims); };

  std::optional<std::shared_ptr<ArchiveBase>> pareto_front() const override final {
    CType x0 = 0.0;
    CType step = 1.0 / static_cast<CType>(num_samples - 1);

    MOFitness f(2);
    auto pf = std::make_shared<UnboundedArchive>(f);
    for (usize i = 0; i < num_samples; i++) {
      auto q = f.worst();
      auto& o = q->as<MOQuality>();
      o.objectives(0) = 1.0 - std::exp(-4.0 * x0) * std::pow(std::sin(6.0 * std::numbers::pi * x0), 6);
      o.objectives(1) = 1.0 - o.objectives(0) * o.objectives(0);
      o.constraint_value = 0.0;
      pf->update(Solution(std::move(q), std::nullopt, Vec<CType>::Zero(dims)));

      x0 += step;
    }
    return pf;
  }
};

};  // namespace goblin

#endif /* _GOBLIN_BENCH_FUNCTIONS_CONTINUOUS_H */
