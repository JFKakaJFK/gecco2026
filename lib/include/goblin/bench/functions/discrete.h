#ifndef _GOBLIN_BENCH_FUNCTIONS_DISCRETE_H
#define _GOBLIN_BENCH_FUNCTIONS_DISCRETE_H

#pragma once

#include "goblin/bench/functions.h"

// TODO
// - [x] OneMax
// - [x] ZeroMax
// - [ ] DeceptiveTrap
// - [ ] BimodalDTrap
// - [ ] Leading Ones
// - [ ] Trailing Zeroes

namespace goblin {

class OneMax final : public ObjectiveBase {
 public:
  OneMax(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return dims; };
  usize num_continuous() const override final { return 0; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
    discrete_active.fill(true);
    return std::make_tuple(discrete_values.array().cast<double>().sum(), 0.0);
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
    discrete_active.fill(true);
    return std::make_tuple(parent_objective_value + discrete_values(discrete_indices).array().cast<double>().sum() -
                               parent_discrete_values(discrete_indices).array().cast<double>().sum(),
                           0.0);
  };

 private:
  usize dims;
};

class ZeroMax final : public ObjectiveBase {
 public:
  ZeroMax(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return dims; };
  usize num_continuous() const override final { return 0; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
    discrete_active.fill(true);
    return std::make_tuple(discrete_values.size() - discrete_values.array().cast<double>().sum(), 0.0);
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
    discrete_active.fill(true);
    return std::make_tuple(parent_objective_value - discrete_values(discrete_indices).array().cast<double>().sum() +
                               parent_discrete_values(discrete_indices).array().cast<double>().sum(),
                           0.0);
  };

 private:
  usize dims;
};

class LeadingOnes final : public ObjectiveBase {
 public:
  LeadingOnes(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return dims; };
  usize num_continuous() const override final { return 0; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
    CType ov = CType(0.0);
    for (usize i = 0; i < dims; i++) {
      discrete_active(i) = true;
      if (discrete_values(i) != DType(1)) {
        break;
      }
      ov += CType(1.0);
    }
    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
};

class TrailingZeros final : public ObjectiveBase {
 public:
  TrailingZeros(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return dims; };
  usize num_continuous() const override final { return 0; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
    CType ov = CType(0.0);
    for (usize i = dims; i > 0;) {
      i--;
      discrete_active(i) = true;
      if (discrete_values(i) != DType(0)) {
        break;
      }
      ov += CType(1.0);
    }
    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
};

class DeceptiveTrap final : public ObjectiveBase {
 public:
  DeceptiveTrap(usize ndims) : DeceptiveTrap(ndims, ndims) {};
  DeceptiveTrap(usize ndims, usize block_size) : dims(ndims), block_size(block_size) {};

  usize num_discrete() const override final { return dims; };
  usize num_continuous() const override final { return 0; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
    discrete_active.fill(true);

    CType ov = CType(0.0);
    for (usize i = 0; i < dims; i += block_size) {
      int unitation = discrete_values(Eigen::seqN(i, std::min(block_size, dims - i))).cast<int>().sum();
      ov += unitation == static_cast<isize>(block_size) ? block_size : static_cast<isize>(block_size) - unitation - 1;
    }
    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
  usize block_size;
};

class BimodalTrap final : public ObjectiveBase {
 public:
  BimodalTrap(usize ndims) : BimodalTrap(ndims, ndims) {};
  BimodalTrap(usize ndims, usize block_size) : dims(ndims), block_size(block_size) {};

  usize num_discrete() const override final { return dims; };
  usize num_continuous() const override final { return 0; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
    discrete_active.fill(true);

    CType ov = CType(0.0);
    for (usize i = 0; i < dims; i += block_size) {
      isize unitation = discrete_values(Eigen::seqN(i, std::min(block_size, dims - i))).cast<isize>().sum();
      ov += unitation == 0 || unitation == static_cast<isize>(block_size) ? static_cast<isize>(block_size) : std::abs<isize>(2 * unitation - block_size - 2);
    }
    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
  usize block_size;
};

};  // namespace goblin

#endif /* _GOBLIN_BENCH_FUNCTIONS_DISCRETE_H */
