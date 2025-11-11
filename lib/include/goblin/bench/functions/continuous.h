#pragma once
#ifndef _GOBLIN_BENCH_FUNCTIONS_CONTINUOUS_H
#define _GOBLIN_BENCH_FUNCTIONS_CONTINUOUS_H

#include <cassert>
#include <numbers>

#include "goblin/bench/functions.h"
#include "goblin/lib/assert.h"

namespace goblin {

class Sphere final : public ObjectiveBase {
 public:
  Sphere(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return 0; };
  usize num_continuous() const override final { return dims; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
    continuous_active.fill(true);
    return std::make_tuple(continuous_values.array().pow(2).sum(), 0.0);
  };

  std::tuple<CType, CType> evaluate_partial(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Active> discrete_active,
                                            RefS<Active> continuous_active,
                                            CRefS<Vec<DType>> parent_discrete_values,
                                            CRefS<Vec<CType>> parent_continuous_values,
                                            CRefS<Active> parent_discrete_active,
                                            CRefS<Active> parent_continuous_active,
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
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
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
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
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
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
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
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
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
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
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

};  // namespace goblin

#endif /* _GOBLIN_BENCH_FUNCTIONS_CONTINUOUS_H */
