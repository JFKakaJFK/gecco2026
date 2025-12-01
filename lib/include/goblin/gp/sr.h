#pragma once
#ifndef _GOBLIN_GP_SR_H
#define _GOBLIN_GP_SR_H

#include <limits>
#include <set>
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
            std::optional<std::vector<CType>> target_objectives = std::nullopt
  )
      : ctx(ctx),
        linear_scaling(linear_scaling),
        objectives(std::holds_alternative<std::string>(objectives)
                       ? std::vector<std::string>{std::get<std::string>(objectives)}
                       : std::get<std::vector<std::string>>(objectives)),
        X_train(X_train.cast<ScalarType>()),
        Y_train(Y_train.cast<ScalarType>()),
        _archive_fitness(MOFitness(this->objectives.size())),
        _fitness(MOFitness(objectives_to_optimize.value_or(this->objectives.size()))),
        _init(from_any_init(init.value_or(std::make_shared<HalfHalfInit>()))),
        _target(_archive_fitness) {
    __goblin_runtime_assert(this->objectives.size() > 0);
    __goblin_runtime_assert(
        !objectives_to_optimize.has_value() ||
        (objectives_to_optimize.value() > 0 && objectives_to_optimize.value() <= this->objectives.size()));

    _num_continuous = this->ctx.num_continuous;
    if (linear_scaling) {
      _num_continuous +=
          2 *
          this->ctx.num_outputs;  // TODO allocate the LS coefficients separately to not mess with optimizers (not a
                                  // problem right now, especially since they are marked as introns...) - this
                                  // effectively is abusing the fact that the continuous values have the same type as
                                  // the LS coefficients combined with the LS coefficients overriding the previous value
    }

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

    if(target_objectives.has_value()){
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
      eval_one(solutions[i], X_train, Y_train, var_Y_train, params, true, solutions[i].quality());
    }
  };

  void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override final {
    _init->add_random(rng, *this, solutions, count);
  };

  const FitnessBase& fitness() const override final { return _fitness; };

  const ArchiveFitnessBase& archive_fitness() const override final { return _archive_fitness; };

  bool always_inherit_continuous() const override final {
    return ctx.const_repr == ConstantRepr::ERCs || ctx.const_repr == ConstantRepr::Edges;
  };

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
    s.quality().objectives = target_objectives;
    __goblin_runtime_assert(static_cast<usize>(s.quality().objectives.size()) >= fitness().num_objectives());
    s.quality().constraint_value = 0.0;
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

  void log(std::ostream& os, SolutionBase& solution) override final {
    os << '"';
    log_solution(os, solution);
    os << "\",";
    for (usize i = 0; i < objectives.size(); i++) {
      os << solution.quality().objectives(i) << ',';
    }
    if (Y_test.size() > 0) {
      // TODO cache this -> solution gets optional second quality?
      // Then again, one can just call predict using the SKlearn regressor for actual use
      // and for all other experiments the overhead is not an issue yet
      Quality q_test = archive_fitness().worst();
      Array<ScalarType> params;  // TODO fit FC params...
      eval_one(solution, X_test, Y_test, var_Y_test, params, false, q_test);
      for (usize i = 0; i < objectives.size(); i++) {
        os << q_test.objectives(i) << ',';
      }
    }

    fitness().log(os, solution.quality());
  };

  void log_solution(std::ostream& os, const SolutionBase& solution) const override final {
    auto exprs = ctx.to_sympy(solution);
    for (usize i = 0; i < exprs.size(); i++) {
      if (i > 0) {
        os << " , ";
      }
      if (linear_scaling) {
        os << solution.continuous_values()(ctx.num_continuous + 2 * i) << " + ("
           << solution.continuous_values()(ctx.num_continuous + 2 * i + 1) << " * (" << exprs[i] << "))";
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
                Quality& quality) {
    usize expression_size;
    auto out = ctx.compute_outputs(_eval_buffer, solution, X, params, expression_size);

    if (!out.has_value()) {
      solution.quality().objectives.array() = std::numeric_limits<CType>::infinity();
      solution.quality().constraint_value = 1.0;
      return;
    }

    Arr2D<ScalarType> Y_pred = out.value();

    if (linear_scaling) {
      Arr2D<ScalarType> Y_pred_train;
      if (!is_train) {
        Y_pred_train = ctx.compute_outputs(_eval_buffer, solution, X_train, params, expression_size).value();
      }

      Arr2D<ScalarType> A_ls = Arr2D<ScalarType>::Ones(Y_train.rows(), 2);
      for (usize o = 0; o < ctx.num_outputs; o++) {
        A_ls.col(1) = (is_train ? Y_pred : Y_pred_train).col(o);
        solution.continuous_values()(Eigen::seqN(ctx.num_continuous + 2 * o, 2)) =
            A_ls.matrix().colPivHouseholderQr().solve(Y_train.matrix().col(o));
      }
    }

    quality.constraint_value = 0.0;
    for (usize j = 0; j < objectives.size(); j++) {
      if (objectives[j] == "mse") {
        if (linear_scaling) {
          quality.objectives(j) = 0.0;
          for (usize o = 0; o < ctx.num_outputs; o++) {
            CType intercept = solution.continuous_values()(ctx.num_continuous + 2 * o);
            CType slope = solution.continuous_values()(ctx.num_continuous + 2 * o + 1);
            quality.objectives(j) += ((intercept + slope * Y_pred.col(o)) - Y.col(o)).square().mean();
          }
        } else {
          quality.objectives(j) = (Y_pred - Y).square().mean();
        }
      } else if (objectives[j] == "nmse") {
        if (linear_scaling) {
          quality.objectives(j) = 0.0;
          for (usize o = 0; o < ctx.num_outputs; o++) {
            CType intercept = solution.continuous_values()(ctx.num_continuous + 2 * o);
            CType slope = solution.continuous_values()(ctx.num_continuous + 2 * o + 1);
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

  MOFitness _archive_fitness;
  MOFitness _fitness;
  std::shared_ptr<InitBase> _init;
  UnboundedArchive _target;
  usize _num_continuous;
  Vec<CType> _continuous_lower_bounds;
  Vec<CType> _continuous_upper_bounds;
  Vec<CType> _continuous_init_lower_bounds;
  Vec<CType> _continuous_init_upper_bounds;
  Arr2D<ScalarType> _eval_buffer;
};

};  // namespace goblin

#endif /* _GOBLIN_GP_SR_H */
