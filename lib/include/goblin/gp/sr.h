#pragma once
#include "context.h"
#ifndef _GOBLIN_GP_SR_H
#define _GOBLIN_GP_SR_H

#include <limits>
#include <set>
#include <cassert>
#include <memory>
#include <tuple>
#include <variant>
#include <vector>
#include <string>
#include <optional>
#include <print>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include "goblin/gp/context.h"
#include "goblin/gp/operator.h"
#include "goblin/gp/template.h"
#include "goblin/gp/init.h"
#include "goblin/gp/instance.h"
#include "goblin/lib/archive.h"
#include "goblin/lib/assert.h"
#include "goblin/lib/instance.h"
#include "goblin/lib/solution.h"
#include "goblin/lib/types.h"
#include "goblin/lib/init.h"

namespace goblin {

class SRQuality : public MOQuality {
 public:
  std::unique_ptr<QualityBase> clone() const override { return std::make_unique<SRQuality>(*this); };

  /// Linear scaling parameters
  Arr2D<CType> ls_params{};

  /*
  The test accuracy uses interior mutability (i.e. it ignores const) since it is not
  part of what defines a solution or its accuracy - as indicated by the name, it is never
  used to make any decisions and only tracked for analysis purposes. By making it mutable
  it an be lazily computed only when requested.
   */
  /// Optional test set accuracy
  mutable std::optional<MOQuality> test_quality = std::nullopt;
};

class SRFitness : public MOFitness {
 public:
  SRFitness(usize num_objectives, bool minimize = true, CType epsilon = 0.0)
      : MOFitness(num_objectives, minimize, epsilon) {}

  std::unique_ptr<QualityBase> worst() const override final {
    const CType inf = std::numeric_limits<CType>().infinity();
    auto q = std::make_unique<SRQuality>();
    q->objectives = Vec<CType>::Constant(num_objectives(), inf);
    q->constraint_value = inf;
    return q;
  };
};

class SRProblem : public GPInstanceBase {
  using ScalarType = CType;  // TODO template the implementation and add a wrapper class - by doing so the wrapper can
                             // at compile time delegate to different ScalarTypes (float, double, mpfr, autodiff
                             // versions) while still having a nice Python API...

 public:
  SRProblem(GPContext ctx,
            Arr2D<CType> X_train,
            Arr2D<CType> Y_train,
            std::optional<Arr2D<CType>> X_test = std::nullopt,
            std::optional<Arr2D<CType>> Y_test = std::nullopt,
            std::variant<std::string, std::vector<std::string>> objectives =
                "mse",  /// The objectives that should be recorded in the archive, and by default the objectives that
                        /// are optimized.

            // TODO I really don't like this API - it is the way it is because this way is convenient for
            // the Python bindings, but makes adding custom objectives hard. Dependency injection would be
            // better, but isn't perfect either - a fully decoupled design ("given the solution, give me the
            // objective value") would potentially mean recomputing the output multiple times, and passing a
            // few fixed values (e.g. the solution, the output, the size) isn't enough for some objectives
            // that could be interesting (e.g. diversity, effective information criterion,...). Until I have
            // a better API design, I will leave this hardcoded, and maybe look at how other GP/SR libraries
            // do it to see if there are better solutions.

            std::optional<usize> objectives_to_optimize =
                std::nullopt,  /// The number of objectives to optimize in case those differ from the `objectives`
                               /// parameter, corresponds to the first `objectives_to_optimize` entries in `objectives`.
            bool linear_scaling = true,
            std::optional<AnyInit> init = std::nullopt,
            CType constant_init_lower_bound = -1.0,
            CType constant_init_upper_bound = 1.0,
            std::optional<std::vector<CType>> target_objectives = std::nullopt,
            std::string gradient_mode = "forward",
            CType gradient_epsilon = 1e-5,
            CType archive_epsilon = 0.0,
            std::optional<bool> always_inherit_continuous = std::nullopt)
      : ctx(ctx),
        linear_scaling(linear_scaling),
        objectives(std::holds_alternative<std::string>(objectives)
                       ? std::vector<std::string>{std::get<std::string>(objectives)}
                       : std::get<std::vector<std::string>>(objectives)),
        X_train(X_train.cast<ScalarType>()),
        Y_train(Y_train.cast<ScalarType>()),
        _archive_fitness(SRFitness(this->objectives.size(), /* minimize = */ true, archive_epsilon)),
        _fitness(SRFitness(objectives_to_optimize.value_or(this->objectives.size()))),
        _init(from_any_init(init.value_or(std::make_shared<HalfHalfInit>()))),
        _target(_archive_fitness),
        _gradient_mode(gradient_mode),
        _gradient_epsilon(gradient_epsilon),
        _always_inherit_continuous(always_inherit_continuous) {
    __goblin_runtime_assert(this->objectives.size() > 0);
    __goblin_runtime_assert(
        !objectives_to_optimize.has_value() ||
        (objectives_to_optimize.value() > 0 && objectives_to_optimize.value() <= this->objectives.size()));

    _num_continuous = this->ctx.num_continuous;
    _continuous_upper_bounds = Vec<CType>::Constant(_num_continuous, std::numeric_limits<CType>::max());
    _continuous_lower_bounds = -_continuous_upper_bounds;

    __goblin_runtime_assert(!isna(constant_init_lower_bound));
    __goblin_runtime_assert(!isna(constant_init_upper_bound));
    __goblin_runtime_assert(constant_init_lower_bound < constant_init_upper_bound);
    _continuous_init_lower_bounds = Vec<CType>::Constant(_num_continuous, constant_init_lower_bound);
    _continuous_init_upper_bounds = Vec<CType>::Constant(_num_continuous, constant_init_upper_bound);

    __goblin_runtime_assert(this->X_train.rows() == this->Y_train.rows());
    __goblin_runtime_assert(static_cast<usize>(this->X_train.cols()) == ctx.num_inputs);
    __goblin_runtime_assert(static_cast<usize>(this->Y_train.cols()) == ctx.num_outputs);

    var_Y_train = (this->Y_train.rowwise() - this->Y_train.colwise().mean()).square().colwise().mean();
    // ~0 => 1 (R2 is not defined, so we fall back to the MSE by not
    // normalizing...)
    for (isize i = 0; i < var_Y_train.size(); i++) {
      if (std::abs(var_Y_train(i)) < CType(1e-12)) {
        var_Y_train(i) = 1.0;
      }
    }

    if (X_test.has_value()) {
      __goblin_runtime_assert(Y_test.has_value());
      this->X_test = X_test.value().cast<ScalarType>();
      this->Y_test = Y_test.value().cast<ScalarType>();

      __goblin_runtime_assert(this->X_train.cols() == this->X_test.cols());
      __goblin_runtime_assert(this->Y_train.cols() == this->Y_test.cols());

      __goblin_runtime_assert(this->X_test.rows() == this->Y_test.rows());

      var_Y_test = (this->Y_test.rowwise() - this->Y_test.colwise().mean()).square().colwise().mean();
      // ~0 => 1 (R2 is not defined, so we fall back to the MSE by not
      // normalizing...)
      for (isize i = 0; i < var_Y_test.size(); i++) {
        if (std::abs(var_Y_test(i)) < CType(1e-12)) {
          var_Y_test(i) = 1.0;
        }
      }
    }

    if (target_objectives.has_value()) {
      register_target(target_objectives.value());
    }
  };

  usize num_discrete() const override final { return ctx.num_discrete; };
  CRef<Vec<DType>> discrete_domain_sizes() const override final { return ctx.domain_sizes; };

  usize num_continuous() const override final { return _num_continuous; };
  CRef<Vec<CType>> continuous_lower_bounds() const override final { return _continuous_lower_bounds; };
  CRef<Vec<CType>> continuous_upper_bounds() const override final { return _continuous_upper_bounds; };

  CRef<Vec<CType>> continuous_init_lower_bounds() const override final { return _continuous_init_lower_bounds; };
  CRef<Vec<CType>> continuous_init_upper_bounds() const override final { return _continuous_init_upper_bounds; };

  void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
    Array<ScalarType> params;
    for (auto i : indices) {
      auto& q = solutions[i].quality_as<SRQuality>();
      q.test_quality = std::nullopt;  // Non test evaluations indicate that the test quality is likely out of date...
      eval_one(solutions[i], X_train, Y_train, var_Y_train, params, true, q, q.ls_params);
    }
  };

  void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override final {
    _init->add_random(rng, *this, solutions, count);
    auto p = dynamic_cast<SRQuality*>(&solutions[solutions.size() - 1].quality());
    assert(p != nullptr && "Quality mismatch");
  };

  const FitnessBase& fitness() const override final { return _fitness; };

  const ArchiveFitnessBase& archive_fitness() const override final { return _archive_fitness; };

  virtual std::tuple<bool, bool> inherit_discrete(SolutionBase& offspring,
                                                  const SolutionBase& donor,
                                                  const Subset& subset) const override {
    const bool inherit_continuous = _always_inherit_continuous.value_or(ctx.const_repr == ConstantRepr::ERCs ||
                                                                        ctx.const_repr == ConstantRepr::Edges) &&
                                    ctx.const_repr != ConstantRepr::None;

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

  // bool always_inherit_continuous() const override final {
  //   return ;
  // };

  std::optional<CType> as_continuous(const SolutionBase& solution, usize discrete_index) const override final {
    auto value = ctx.domain2value(discrete_index, solution.discrete_values()(discrete_index));
    if (ctx.value_kind[value] == ValueKind::Constant) {
      return solution.continuous_values()(ctx.const_repr == ConstantRepr::Pool ? ctx.value_idx[value] : discrete_index);
    }
    return std::nullopt;
  };

  void register_target(CRefS<Vec<CType>> target_objectives) {
    _target.clear();
    Solution s(
        archive_fitness().worst(),
        num_discrete() > 0 ? std::make_optional<Vec<DType>>(Vec<DType>::Zero(num_discrete())) : std::nullopt,
        num_continuous() > 0 ? std::make_optional<Vec<CType>>(Vec<CType>::Zero(num_continuous())) : std::nullopt);
    s.quality_as<SRQuality>().objectives = target_objectives;
    __goblin_runtime_assert(static_cast<usize>(s.quality_as<SRQuality>().objectives.size()) >=
                            fitness().num_objectives());
    s.quality_as<SRQuality>().constraint_value = 0.0;
    _target.update(s, false);
  };

  void register_target(std::vector<CType> target_objectives) {
    register_target(Eigen::Map<Vec<CType>>(target_objectives.data(), target_objectives.size()));
  };

  bool target_reached(const ArchiveBase& archive) const override final {
    if (!_target.empty()) {
      return archive.covers(_target);
    } else {
      return false;
    }
  };

  void log_header(std::ostream& os) const override final {
    os << "expressions,";
    for (auto& o : objectives) {
      os << o << "_train,";
    }
    if (Y_test.size() > 0) {
      for (auto& o : objectives) {
        os << o << "_test,";
      }
    }

    fitness().log_header(os);
  };

  void evaluate_test(const SolutionBase& solution) const {
    const auto& q = solution.quality_as<SRQuality>();
    if (Y_test.size() > 0 && !q.test_quality.has_value()) {
      Solution copy = solution;  // copy is needed since active variables are not mutable...

      Array<ScalarType> params;  // TODO fit FC params...
      auto& cq = copy.quality_as<SRQuality>();
      eval_one(copy, X_test, Y_test, var_Y_test, params, false, cq, cq.ls_params);

      q.test_quality = MOQuality();
      q.test_quality.value().objectives = copy.quality_as<SRQuality>().objectives;
      q.test_quality.value().constraint_value = copy.quality_as<SRQuality>().constraint_value;
    }
  };

  void log(std::ostream& os, const SolutionBase& solution) override final {
    os << '"';
    log_solution(os, solution);
    os << "\",";

    const auto& q = solution.quality_as<SRQuality>();
    for (usize i = 0; i < objectives.size(); i++) {
      os << q.objectives(i) << ',';
    }
    if (Y_test.size() > 0) {
      if (!q.test_quality.has_value()) {
        evaluate_test(solution);
      }

      for (usize i = 0; i < objectives.size(); i++) {
        os << q.test_quality.value().objectives(i) << ',';
      }
    }

    archive_fitness().log(os, solution.quality());
  };

  void log_solution(std::ostream& os, const SolutionBase& solution) const override final {
    auto exprs = ctx.to_sympy(solution);
    for (usize i = 0; i < exprs.size(); i++) {
      if (i > 0) {
        os << " , ";
      }
      if (linear_scaling) {
        const auto& q = solution.quality_as<SRQuality>();
        if (static_cast<usize>(q.ls_params.cols()) != ctx.num_outputs) {
          os << exprs[i];  // for the edge case where unevaluated solutions are logged...
        } else {
          os << q.ls_params(0, i) << " + (" << q.ls_params(1, i) << " * (" << exprs[i] << "))";
        }
      } else {
        os << exprs[i];
      }
    }
  };

  std::tuple<std::vector<usize>, u64> gradient_steps(Rng& rng,
                                                     SolutionSetBase& solutions,
                                                     SolutionSetBase& parents,
                                                     const std::span<const usize>& indices,
                                                     usize num_steps) override final;

  const GPContext& context() const override final { return ctx; }

  GPContext ctx;
  bool linear_scaling;
  std::vector<std::string> objectives;
  Arr2D<ScalarType> X_train;
  Arr2D<ScalarType> Y_train;
  Array<ScalarType> var_Y_train;
  Arr2D<ScalarType> X_test;
  Arr2D<ScalarType> Y_test;
  Array<ScalarType> var_Y_test;

 private:
  // solution, X, Y, train/test, quality to write to
  void eval_one(SolutionBase& solution,
                const Arr2D<ScalarType>& X,
                const Arr2D<ScalarType>& Y,
                const Array<ScalarType>& var_Y,
                const Array<ScalarType>& params,
                bool is_train,
                MOQuality& quality,
                Arr2D<CType>& ls_params) const {
    usize expression_size;
    auto out = ctx.compute_outputs(_eval_buffer, solution, X, params, expression_size);

    if (!out.has_value()) {
      quality.objectives.array() = std::numeric_limits<CType>::infinity();
      quality.constraint_value = 1.0;
      return;
    }

    Arr2D<ScalarType> Y_pred = out.value();
    if (linear_scaling && is_train) {
      Mat<ScalarType> A_ls = Mat<ScalarType>::Ones(Y_train.rows(), 2);
      ls_params.resize(2, ctx.num_outputs);
      for (usize o = 0; o < ctx.num_outputs; o++) {
        A_ls.col(1) = Y_pred.col(o);
        Vec<ScalarType> b = A_ls.colPivHouseholderQr().solve(Y_train.matrix().col(o));
        ls_params.col(o) = A_ls.colPivHouseholderQr().solve(Y_train.matrix().col(o));
      }
    }

    quality.constraint_value = 0.0;
    for (usize j = 0; j < objectives.size(); j++) {
      if (objectives[j] == "mse") {
        if (linear_scaling) {
          quality.objectives(j) = 0.0;
          for (usize o = 0; o < ctx.num_outputs; o++) {
            CType intercept = ls_params(0, o);  // solution.continuous_values()(ctx.num_continuous + 2 * o);
            CType slope = ls_params(1, o);      // solution.continuous_values()(ctx.num_continuous + 2 * o + 1);
            quality.objectives(j) += ((intercept + slope * Y_pred.col(o)) - Y.col(o)).square().mean();
          }
        } else {
          quality.objectives(j) = (Y_pred - Y).square().mean();
        }
      } else if (objectives[j] == "nmse") {
        if (linear_scaling) {
          quality.objectives(j) = 0.0;
          for (usize o = 0; o < ctx.num_outputs; o++) {
            CType intercept = ls_params(0, o);  // solution.continuous_values()(ctx.num_continuous + 2 * o);
            CType slope = ls_params(1, o);      // solution.continuous_values()(ctx.num_continuous + 2 * o + 1);
            quality.objectives(j) += ((intercept + slope * Y_pred.col(o)) - Y.col(o)).square().mean() / var_Y(o);
          }
        } else {
          quality.objectives(j) = 0.0;
          for (usize o = 0; o < ctx.num_outputs; o++) {
            quality.objectives(j) += (Y_pred.col(o) - Y.col(o)).square().mean() / var_Y(o);
          }
        }
      } else if (objectives[j] == "size") {
        quality.objectives(j) = static_cast<CType>(expression_size);
      }
    }
  };

  SRFitness _archive_fitness;
  SRFitness _fitness;
  std::shared_ptr<InitBase> _init;
  UnboundedArchive _target;
  std::string _gradient_mode{};
  CType _gradient_epsilon{};
  std::optional<bool> _always_inherit_continuous{};
  usize _num_continuous{};
  Vec<CType> _continuous_lower_bounds{};
  Vec<CType> _continuous_upper_bounds{};
  Vec<CType> _continuous_init_lower_bounds{};
  Vec<CType> _continuous_init_upper_bounds{};
  mutable Arr2D<ScalarType> _eval_buffer{};
};

};  // namespace goblin

#endif /* _GOBLIN_GP_SR_H */
