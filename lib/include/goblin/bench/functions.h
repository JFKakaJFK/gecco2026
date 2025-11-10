#ifndef _GOBLIN_BENCH_FUNCTIONS_H
#define _GOBLIN_BENCH_FUNCTIONS_H

#pragma once
#include "goblin/lib/types.h"
#include <tuple>

namespace goblin {

class ObjectiveBase {
 public:
  virtual usize num_discrete() const = 0;
  virtual usize num_continuous() const = 0;

  virtual std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Active> discrete_active,
                                            RefS<Active> continuous_active) = 0;

  virtual std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
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
                                            const std::span<const usize>& continuous_indices) {
    return evaluate(discrete_values, continuous_values, discrete_active, continuous_active);
  };

  virtual ~ObjectiveBase() {};
};

};  // namespace goblin

#endif /* _GOBLIN_BENCH_FUNCTIONS_H */
