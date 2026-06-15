#pragma once
#ifndef _GOBLIN_BENCH_PROBLEM_H
#define _GOBLIN_BENCH_PROBLEM_H

#include <memory>
#include <type_traits>
#include <variant>
#include <stdexcept>

#include "goblin/bench/functions.h"
#include "goblin/lib/archive.h"
#include "goblin/lib/instance.h"
#include "goblin/lib/solution.h"
#include "goblin/lib/types.h"
#include "goblin/lib/init.h"

namespace goblin {

class PyFunctionBase : public MOFunctionBase {
 public:
  virtual usize num_objectives() const override = 0;
  virtual usize num_discrete() const override = 0;
  virtual usize num_continuous() const override = 0;

  virtual std::tuple<Vec<CType>, CType> eval(SolutionBase& solution) = 0;

  void evaluate(SolutionBase& solution) override {
    solution.discrete_active().fill(true);
    solution.continuous_active().fill(true);
    auto [objectives, cv] = eval(solution);
    solution.quality_as<MOQuality>().objectives = objectives;
    solution.quality_as<MOQuality>().constraint_value = cv;
  };
  void evaluate_partial(SolutionBase& solution, const SolutionBase& parent, const Subset& subset) override {
    evaluate(solution);
  };

  virtual ~PyFunctionBase() {};
};

class Objectives final : public MOFunctionBase {
 public:
  Objectives(std::vector<std::shared_ptr<ObjectiveBase>> objectives) : objectives(std::move(objectives)) {
    _num_discrete = 0;
    _num_continuous = 0;
    for (auto& o : this->objectives) {
      _num_discrete = std::max(_num_discrete, o->num_discrete());
      _num_continuous = std::max(_num_continuous, o->num_continuous());
    }
  };

  usize num_objectives() const override final { return objectives.size(); }
  usize num_discrete() const override final { return _num_discrete; }
  usize num_continuous() const override final { return _num_continuous; }

  void evaluate(SolutionBase& solution) override final {
    solution.discrete_active().fill(false);
    solution.continuous_active().fill(false);
    solution.quality_as<MOQuality>().constraint_value = 0.0;
    for (usize i = 0; i < num_objectives(); i++) {
      auto [ov, cv] = objectives[i]->evaluate(solution.discrete_values(), solution.continuous_values(),
                                              solution.discrete_active(), solution.continuous_active());
      solution.quality_as<MOQuality>().objectives(i) = ov;
      solution.quality_as<MOQuality>().constraint_value += std::max(CType(0.0), cv);
    }
  };

  void evaluate_partial(SolutionBase& solution, const SolutionBase& parent, const Subset& subset) override final {
    solution.discrete_active().fill(false);
    solution.continuous_active().fill(false);
    const auto& pq = parent.quality_as<MOQuality>();
    solution.quality_as<MOQuality>().constraint_value = 0.0;
    for (usize i = 0; i < num_objectives(); i++) {
      auto [ov, cv] = objectives[i]->evaluate_partial(
          solution.discrete_values(), solution.continuous_values(), solution.discrete_active(),
          solution.continuous_active(), parent.discrete_values(), parent.continuous_values(), parent.discrete_active(),
          parent.continuous_active(), pq.objectives(i), pq.constraint_value, subset.discrete, subset.continuous);
      solution.quality_as<MOQuality>().objectives(i) = ov;
      solution.quality_as<MOQuality>().constraint_value += std::max(CType(0.0), cv);
    }
  };

 private:
  usize _num_discrete = 0;
  usize _num_continuous = 0;
  std::vector<std::shared_ptr<ObjectiveBase>> objectives;
};

class BenchmarkInstance final : public InstanceBase {
 public:
  BenchmarkInstance(std::variant<std::vector<std::shared_ptr<ObjectiveBase>>,
                                 std::shared_ptr<MOFunctionBase>,
                                 std::shared_ptr<ObjectiveBase>> objectives,
                    std::variant<DType, Vec<DType>> discrete_domain = DType(2),
                    std::variant<CType, Vec<CType>> continuous_lower_bound = -std::numeric_limits<CType>::infinity(),
                    std::variant<CType, Vec<CType>> continuous_upper_bound = std::numeric_limits<CType>::infinity(),
                    std::variant<CType, Vec<CType>> continuous_init_lower_bound = CType(0.0),
                    std::variant<CType, Vec<CType>> continuous_init_upper_bound = CType(1.0),
                    std::optional<AnyInit> init = std::nullopt,
                    std::optional<std::vector<CType>> target_objectives = std::nullopt,
                    std::optional<usize> target_archive_size = std::nullopt,
                    std::optional<CType> target_tolerance = std::nullopt)
      : _objectives(std::holds_alternative<std::shared_ptr<MOFunctionBase>>(objectives)
                        ? std::get<std::shared_ptr<MOFunctionBase>>(objectives)
                        : (std::holds_alternative<std::shared_ptr<ObjectiveBase>>(objectives)
                               ? std::static_pointer_cast<MOFunctionBase>(
                                     std::make_shared<Objectives>(std::vector<std::shared_ptr<ObjectiveBase>>{
                                         std::get<std::shared_ptr<ObjectiveBase>>(objectives)}))
                               : std::static_pointer_cast<MOFunctionBase>(std::make_shared<Objectives>(
                                     std::move(std::get<std::vector<std::shared_ptr<ObjectiveBase>>>(objectives)))))),
        _fitness(MOFitness(_objectives->num_objectives(),
                           /* minimize */ _objectives->num_continuous() > 0)),
        _init(from_any_init(init.value_or(std::make_shared<CompleteInit>()))) {
    if (std::holds_alternative<DType>(discrete_domain)) {
      _discrete_domain_sizes.resize(_objectives->num_discrete());
      _discrete_domain_sizes.fill(std::get<DType>(discrete_domain));
    } else {
      _discrete_domain_sizes = std::get<Vec<DType>>(discrete_domain);
    }
    __goblin_runtime_assert(static_cast<usize>(_discrete_domain_sizes.size()) == _objectives->num_discrete());

    if (std::holds_alternative<CType>(continuous_lower_bound)) {
      _continuous_lower_bounds.resize(_objectives->num_continuous());
      _continuous_lower_bounds.fill(std::get<CType>(continuous_lower_bound));
    } else {
      _continuous_lower_bounds = std::get<Vec<CType>>(continuous_lower_bound);
    }
    __goblin_runtime_assert(static_cast<usize>(_continuous_lower_bounds.size()) == _objectives->num_continuous());

    if (std::holds_alternative<CType>(continuous_upper_bound)) {
      _continuous_upper_bounds.resize(_objectives->num_continuous());
      _continuous_upper_bounds.fill(std::get<CType>(continuous_upper_bound));
    } else {
      _continuous_upper_bounds = std::get<Vec<CType>>(continuous_upper_bound);
    }
    __goblin_runtime_assert(static_cast<usize>(_continuous_upper_bounds.size()) == _objectives->num_continuous());

    set_initial_bounds(continuous_init_lower_bound, continuous_init_upper_bound);

    for (usize i = 0; i < _objectives->num_continuous(); i++) {
      __goblin_runtime_assert(!isna(_continuous_init_lower_bounds(i)));
      __goblin_runtime_assert(!isna(_continuous_init_upper_bounds(i)));
      __goblin_runtime_assert(_continuous_init_lower_bounds(i) < _continuous_init_upper_bounds(i));
    }

    if (target_objectives.has_value()) {
      add_target(target_objectives.value());
    }
    if (target_archive_size.has_value()) {
      add_target_front_size(target_archive_size.value());
    }

    // TODO for problem specified bounds/targets: either merge with user-provided information or only use one instead of
    // silently overwriting user provided parameters...

    auto fn_dbounds = _objectives->discrete_domain_sizes();
    if (fn_dbounds.has_value()) {
      // TODO check if current bounds are non-inf and outside -> conflict...
      _discrete_domain_sizes = fn_dbounds.value();
    }

    auto fn_lb = _objectives->continuous_lower_bounds();
    if (fn_lb.has_value()) {
      // TODO check if current bounds are non-inf and outside -> conflict...
      _continuous_lower_bounds = fn_lb.value();
      _continuous_init_lower_bounds = fn_lb.value();
    }

    auto fn_ub = _objectives->continuous_upper_bounds();
    if (fn_ub.has_value()) {
      // TODO check if current bounds are non-inf and outside -> conflict...
      _continuous_upper_bounds = fn_ub.value();
      _continuous_init_upper_bounds = fn_ub.value();
    }

    // if the problem provides a pareto front of optimal solution objectives, typically the success condition is to
    // approximate the target with some tolerance instead of perfect recovery -> only register a target condition if a
    // tolerance is provided
    auto fn_front = _objectives->pareto_front();
    if (fn_front.has_value() && target_tolerance.has_value()) {
      add_target([pareto_front = std::move(fn_front.value()), tolerance = target_tolerance.value()](
                     const ArchiveBase& archive) { return archive.approximately_covers(*pareto_front, tolerance); });
    }
  };

  void set_init(AnyInit init) { _init = from_any_init(init); }

  void set_initial_bounds(std::variant<CType, Vec<CType>> continuous_init_lower_bound = CType(0.0),
                          std::variant<CType, Vec<CType>> continuous_init_upper_bound = CType(1.0)) {
    if (std::holds_alternative<CType>(continuous_init_lower_bound)) {
      _continuous_init_lower_bounds.resize(_objectives->num_continuous());
      _continuous_init_lower_bounds.fill(std::get<CType>(continuous_init_lower_bound));
    } else {
      _continuous_init_lower_bounds = std::get<Vec<CType>>(continuous_init_lower_bound);
    }
    __goblin_runtime_assert(static_cast<usize>(_continuous_init_lower_bounds.size()) == _objectives->num_continuous());

    if (std::holds_alternative<CType>(continuous_init_upper_bound)) {
      _continuous_init_upper_bounds.resize(_objectives->num_continuous());
      _continuous_init_upper_bounds.fill(std::get<CType>(continuous_init_upper_bound));
    } else {
      _continuous_init_upper_bounds = std::get<Vec<CType>>(continuous_init_upper_bound);
    }
    __goblin_runtime_assert(static_cast<usize>(_continuous_init_upper_bounds.size()) == _objectives->num_continuous());
  };

  CRef<Vec<DType>> discrete_domain_sizes() const override final { return _discrete_domain_sizes; };

  CRef<Vec<CType>> continuous_lower_bounds() const override final { return _continuous_lower_bounds; };
  CRef<Vec<CType>> continuous_upper_bounds() const override final { return _continuous_upper_bounds; };

  void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override final {
    _init->add_random(rng, *this, solutions, count);
  };

  CRef<Vec<CType>> continuous_init_lower_bounds() const override final { return _continuous_init_lower_bounds; }
  CRef<Vec<CType>> continuous_init_upper_bounds() const override final { return _continuous_init_upper_bounds; }

  void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
    for (auto i : indices) {
      _objectives->evaluate(solutions[i]);
    }
  };

  void evaluate_partial(Rng& rng,
                        SolutionSetBase& solutions,
                        SolutionSetBase& parents,
                        const std::vector<const Subset*>& subsets,
                        const std::span<const usize>& indices) override final {
    for (auto i : indices) {
      _objectives->evaluate_partial(solutions[i], parents[i], *(subsets[i]));
    }
  };

  const FitnessBase& fitness() const override final { return _fitness; }

  const ArchiveFitnessBase& archive_fitness() const override final { return _fitness; }

 private:
  std::shared_ptr<MOFunctionBase> _objectives;
  MOFitness _fitness;
  std::shared_ptr<InitBase> _init;
  Vec<DType> _discrete_domain_sizes;
  Vec<CType> _continuous_lower_bounds;
  Vec<CType> _continuous_upper_bounds;

  Vec<CType> _continuous_init_lower_bounds;
  Vec<CType> _continuous_init_upper_bounds;
};

};  // namespace goblin

#endif /* _GOBLIN_BENCH_PROBLEM_H */
