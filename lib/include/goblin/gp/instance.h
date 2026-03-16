#pragma once
#ifndef _GOBLIN_GP_INSTANCE_H
#define _GOBLIN_GP_INSTANCE_H

#include <limits>
#include <functional>
#include <tuple>

#include "goblin/lib/instance.h"
#include "goblin/gp/context.h"

namespace goblin {

class GPInstanceBase : public InstanceBase {
 public:
  virtual const GPContext& context() const = 0;

  virtual ~GPInstanceBase() = default;
};

class PyGPInstance : public GPInstanceBase {
  GPContext ctx;
  std::function<std::tuple<Vec<CType>, CType>(std::function<std::optional<Arr2D<CType>>(Arr2D<CType>)>,
                                              std::vector<std::string>,
                                              usize)>
      evalfn;
  MOFitness _fitness;
  std::shared_ptr<InitBase> init;

  Vec<CType> init_lb;
  Vec<CType> init_ub;

  Vec<CType> lb;
  Vec<CType> ub;

  Arr2D<CType> eval_buffer;

 public:
  PyGPInstance(GPContext ctx,
               std::function<std::tuple<Vec<CType>, CType>(std::function<std::optional<Arr2D<CType>>(Arr2D<CType>)>,
                                                           std::vector<std::string>,
                                                           usize)> evaluate,
               usize num_objectives,
               AnyInit init,
               bool minimize = true,
               CType erc_init_lb = -10.0,
               CType erc_init_ub = 10.0)
      : ctx(ctx),
        evalfn(evaluate),
        _fitness(MOFitness(num_objectives, minimize)),
        init(from_any_init(init)),
        init_lb(Vec<CType>::Constant(ctx.num_continuous, erc_init_lb)),
        init_ub(Vec<CType>::Constant(ctx.num_continuous, erc_init_ub)) {
    lb = Vec<CType>::Constant(ctx.num_continuous, -std::numeric_limits<CType>::infinity());
    ub = Vec<CType>::Constant(ctx.num_continuous, std::numeric_limits<CType>::infinity());
  };

  CRef<Vec<DType>> discrete_domain_sizes() const override final { return ctx.domain_sizes; };

  CRef<Vec<CType>> continuous_lower_bounds() const override final { return lb; }
  CRef<Vec<CType>> continuous_upper_bounds() const override final { return ub; }

  CRef<Vec<CType>> continuous_init_lower_bounds() const override final { return init_lb; }
  CRef<Vec<CType>> continuous_init_upper_bounds() const override final { return init_ub; }

  void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
    Array<CType> params;
    for (auto i : indices) {
      std::function<std::optional<Arr2D<CType>>(Arr2D<CType>)> predict = [&](Arr2D<CType> inputs) {
        usize size;
        return ctx.compute_outputs(eval_buffer, solutions[i], inputs, params, size);
      };

      solutions[i].discrete_active().fill(false);
      solutions[i].continuous_active().fill(false);

      std::vector<std::string> exprs = ctx.to_sympy(solutions[i]);
      usize size;
      ctx.nodes_post_order(solutions[i], false, size);
      auto [objectives, cv] = evalfn(predict, exprs, size);

      MOQuality q;
      q.objectives = objectives;
      q.constraint_value = cv;
      solutions[i].assign_quality(q);
    }
  };

  void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override final {
    return init->add_random(rng, *this, solutions, count);
  };

  const FitnessBase& fitness() const override final { return _fitness; };

  const ArchiveFitnessBase& archive_fitness() const override final { return _fitness; };

  virtual std::tuple<bool, bool> inherit_discrete(SolutionBase& offspring,
                                                  const SolutionBase& donor,
                                                  const Subset& subset) const override {
    const bool inherit_continuous = ctx.const_repr == ConstantRepr::ERCs || ctx.const_repr == ConstantRepr::Edges;

    // the pool size is not tied to the number of discrete variables, so the full pool instead of the paired values is
    // inherited...
    const bool inherit_by_index = ctx.const_repr != ConstantRepr::Pool;

    bool any_active_changed = false, anything_changed = false;
    for (usize i : subset.discrete) {
      if (offspring.discrete_values()(i) != donor.discrete_values()(i)) {
        any_active_changed |= offspring.discrete_active()(i);
        anything_changed = true;
        offspring.discrete_values()(i) = donor.discrete_values()(i);
      }

      // TODO for GCS: inherit child arities + permutations

      if (inherit_continuous && inherit_by_index) {
        // TODO sufficiently relatively + absolutely different or no check, but floating point equality is not really
        // useful...
        //
        // yes, the indices here should be from the discrete subset!
        if (offspring.continuous_values()(i) != donor.continuous_values()(i)) {
          any_active_changed |= offspring.continuous_active()(i);
          anything_changed = true;
          offspring.continuous_values()(i) = donor.continuous_values()(i);
        }
      }
    }

    if (inherit_continuous && !inherit_by_index) {
      // note: arguably just inheriting all continuous variables even if the inherited discrete values might not even be
      // constants is not the best idea - but earlier experiments on another codebase suggested that more
      // appropriate/interpolating continuous mixing doesn't really work and here it also is more for completeness and
      // not used by default...
      for (usize i = 0; i < num_continuous(); i++) {
        // TODO sufficiently relatively + absolutely different or no check, but floating point equality is not really
        // useful...
        if (offspring.continuous_values()(i) != donor.continuous_values()(i)) {
          any_active_changed |= offspring.continuous_active()(i);
          anything_changed = true;
          offspring.continuous_values()(i) = donor.continuous_values()(i);
        }
      }
    }

    return std::make_tuple(any_active_changed, anything_changed);
  }

  std::optional<CType> as_continuous(const SolutionBase& solution, usize discrete_index) const override final {
    auto value = ctx.domain2value(discrete_index, solution.discrete_values()(discrete_index));
    if (ctx.value_kind[value] == ValueKind::Constant) {
      return solution.continuous_values()(ctx.const_repr == ConstantRepr::Pool ? ctx.value_idx[value] : discrete_index);
    }
    return std::nullopt;
  };

  void log_solution(std::ostream& os, const SolutionBase& solution) const override final {
    auto exprs = ctx.to_sympy(solution);
    for (usize i = 0; i < exprs.size(); i++) {
      if (i > 0) {
        os << " , ";
      }
      os << exprs[i];
    }
  };

  const GPContext& context() const override final { return ctx; };
};

};  // namespace goblin

#endif /* _GOBLIN_GP_INSTANCE_H */
