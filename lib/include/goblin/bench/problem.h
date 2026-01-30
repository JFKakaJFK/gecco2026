#pragma once
#ifndef _GOBLIN_BENCH_PROBLEM_H
#define _GOBLIN_BENCH_PROBLEM_H

#include <memory>
#include <type_traits>
#include <variant>

#include "goblin/bench/functions.h"
#include "goblin/lib/archive.h"
#include "goblin/lib/instance.h"
#include "goblin/lib/solution.h"
#include "goblin/lib/types.h"
#include "goblin/lib/init.h"

namespace goblin {

class MOFunctionBase {
 public:
  virtual usize num_objectives() const = 0;
  virtual usize num_discrete() const = 0;
  virtual usize num_continuous() const = 0;

  virtual void evaluate(SolutionBase& solution) = 0;

  virtual void evaluate_partial(SolutionBase& solution, const SolutionBase& parent, const Subset& subset) {
    evaluate(solution);
  };

  virtual ~MOFunctionBase() {};
};

class PyFunctionBase : MOFunctionBase {
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
    auto& q = solution.quality_as<MOQuality>();
    q.constraint_value = 0.0;
    for (usize i = 0; i < num_objectives(); i++) {
      auto [ov, cv] = objectives[i]->evaluate(solution.discrete_values(), solution.continuous_values(),
                                              solution.discrete_active(), solution.continuous_active());
      q.objectives(i) = ov;
      q.constraint_value += std::max(CType(0.0), cv);
    }
  };

  void evaluate_partial(SolutionBase& solution, const SolutionBase& parent, const Subset& subset) override final {
    solution.discrete_active().fill(false);
    solution.continuous_active().fill(false);
    auto& q = solution.quality_as<MOQuality>();
    const auto& pq = parent.quality_as<MOQuality>();
    q.constraint_value = 0.0;
    for (usize i = 0; i < num_objectives(); i++) {
      auto [ov, cv] = objectives[i]->evaluate_partial(
          solution.discrete_values(), solution.continuous_values(), solution.discrete_active(),
          solution.continuous_active(), parent.discrete_values(), parent.continuous_values(), parent.discrete_active(),
          parent.continuous_active(), pq.objectives(i), pq.constraint_value, subset.discrete, subset.continuous);
      q.objectives(i) = ov;
      q.constraint_value += std::max(CType(0.0), cv);
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
                    std::optional<std::variant<Vec<CType>, std::tuple<Mat<DType>, Mat<CType>>, std::vector<CType>>>
                        target = std::nullopt,
                    std::optional<usize> target_archive_size = std::nullopt)
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
        _init(from_any_init(init.value_or(std::make_shared<CompleteInit>()))),
        _target(_fitness),
        _target_archive_size(target_archive_size) {
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

    if (target.has_value()) {
      if (std::holds_alternative<Vec<CType>>(target.value())) {
        register_target(std::get<Vec<CType>>(target.value()));
      } else if (std::holds_alternative<std::vector<CType>>(target.value())) {
        register_target(std::get<std::vector<CType>>(target.value()));
      } else {
        // this assumes that the evaluation function is reasonably cheap
        // (lazy evaluation would be an option, but more complex and not really
        // that beneficial)
        auto [dvals, cvals] = std::get<1>(target.value());
        register_target_front(dvals, cvals);
      }
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

  void register_target(CRefS<Vec<CType>> target_objectives) {
    _target.clear();
    Solution s(
        archive_fitness().worst(),
        num_discrete() > 0 ? std::make_optional<Vec<DType>>(Vec<DType>::Zero(num_discrete())) : std::nullopt,
        num_continuous() > 0 ? std::make_optional<Vec<CType>>(Vec<CType>::Zero(num_continuous())) : std::nullopt);
    auto& q = s.quality_as<MOQuality>();
    q.objectives = target_objectives;
    __goblin_runtime_assert(static_cast<usize>(q.objectives.size()) >= fitness().num_objectives());
    q.constraint_value = 0.0;
    _target.update(s, false);
  };

  void register_target(std::vector<CType> target_objectives) {
    register_target(Eigen::Map<Vec<CType>>(target_objectives.data(), target_objectives.size()));
  };

  void register_target_front(const ArchiveBase& other) {
    _target.clear();
    for (usize i = 0; i < other.size(); i++) {
      _target.update(other[i], false);
    }
  };

  void register_target_front(Mat<DType> discrete, Mat<CType> continuous) {
    _target.clear();
    __goblin_runtime_assert(discrete.rows() == continuous.rows());
    __goblin_runtime_assert(static_cast<usize>(discrete.cols()) == num_discrete());
    __goblin_runtime_assert(static_cast<usize>(continuous.cols()) == num_continuous());
    for (isize i = 0; i < discrete.rows(); i++) {
      Solution s(archive_fitness().worst(),
                 num_discrete() > 0 ? std::make_optional<Vec<DType>>(discrete.row(i)) : std::nullopt,
                 num_continuous() > 0 ? std::make_optional<Vec<CType>>(continuous.row(i)) : std::nullopt);
      _objectives->evaluate(s);
      _target.update(s, false);
    }
  };

  void register_target_archive_size(usize target_archive_size) {
    __goblin_runtime_assert(target_archive_size > 1);
    _target_archive_size = target_archive_size;
  };

  usize num_objectives() const override final { return _objectives->num_objectives(); };

  usize num_discrete() const override final { return _objectives->num_discrete(); };
  CRef<Vec<DType>> discrete_domain_sizes() const override final { return _discrete_domain_sizes; };

  usize num_continuous() const override final { return _objectives->num_continuous(); };
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

  bool target_reached(const ArchiveBase& archive) const override final {
    if (!_target.empty()) {
      return archive.covers(_target);
    } else if (_target_archive_size.has_value()) {
      return archive.size() >= _target_archive_size.value();
    } else {
      return false;
    }
  };

 private:
  std::shared_ptr<MOFunctionBase> _objectives;
  MOFitness _fitness;
  std::shared_ptr<InitBase> _init;
  UnboundedArchive _target;
  std::optional<usize> _target_archive_size;
  Vec<DType> _discrete_domain_sizes;
  Vec<CType> _continuous_lower_bounds;
  Vec<CType> _continuous_upper_bounds;

  Vec<CType> _continuous_init_lower_bounds;
  Vec<CType> _continuous_init_upper_bounds;
};

};  // namespace goblin

#endif /* _GOBLIN_BENCH_PROBLEM_H */
