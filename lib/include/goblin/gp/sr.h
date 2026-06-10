#pragma once
#ifndef _GOBLIN_GP_SR_H
#define _GOBLIN_GP_SR_H

#include <limits>
#include <cassert>
#include <memory>
#include <tuple>
#include <variant>
#include <vector>
#include <string>
#include <optional>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include "goblin/gp/gpu_evaluation/launch_config.h"
#include "goblin/gp/gpu_evaluation/types.h"
#ifdef GOBLIN_HAS_CUDA
#include "goblin/gp/gpu_evaluation/evaluate.h"
#include "goblin/gp/gpu_evaluation/memory.h"
#include "goblin/gp/gpu_evaluation/misc.h"
#endif // GOBLIN_HAS_CUDA

#include "goblin/gp/context.h"
#include "goblin/gp/init.h"
#include "goblin/gp/instance.h"
#include "goblin/lib/archive.h"
#include "goblin/lib/assert.h"
#include "goblin/lib/solution.h"
#include "goblin/lib/types.h"
#include "goblin/lib/init.h"

namespace goblin {

const CType VARIANCE_BOUND = 1e-12;
const CType DEFAULT_GRADIENT_EPSILON = 1e-5;

class SRQuality : public MOQuality {
 public:
  std::unique_ptr<QualityBase> clone() const override { return std::make_unique<SRQuality>(*this); };

  // Linear scaling coefficients per output: row 0 = intercept, row 1 = slope.
  // Scaled prediction: intercept + slope * raw_tree_output.
  Arr2D<CType> ls_params;

  // Lazily computed test-set accuracy, populated on first request via evaluate_test().
  // Declared mutable because it is not part of solution identity and never drives decisions.
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

// Symbolic regression problem: fits a GP tree to minimise one or more objectives
// (e.g. MSE, NMSE, tree size) over (X_train, Y_train) with optional linear scaling
// of the raw tree output.  GPU evaluation is available when GOBLIN_HAS_CUDA is defined.
class SRProblem : public GPInstanceBase {
  // TODO template ScalarType and add a wrapper class so the Python API can delegate
  // at compile time to float, double, mpfr, or autodiff without changing the interface.
  using ScalarType = CType;

 public:
  // TODO the objectives API is ad-hoc: it is convenient for Python bindings but makes
  // adding custom objectives hard.  Dependency injection would be cleaner but risks
  // recomputing the tree output multiple times.  Leaving hardcoded until a better design
  // is found (see how other GP/SR libraries handle this).
  SRProblem(GPContext ctx,
            Arr2D<CType> X_train,
            Arr2D<CType> Y_train,
            std::optional<Arr2D<CType>> X_test = std::nullopt,
            std::optional<Arr2D<CType>> Y_test = std::nullopt,
            // Objectives recorded in the archive (and optimised by default).
            std::variant<std::string, std::vector<std::string>> objectives = "mse",
            // Restrict optimisation to the first N entries of `objectives`; the rest
            // are tracked in the archive only.
            std::optional<usize> objectives_to_optimize = std::nullopt,
            bool linear_scaling = true,
            std::optional<AnyInit> init = std::nullopt,
            CType constant_init_lower_bound = -1.0,
            CType constant_init_upper_bound = 1.0,
            std::optional<std::vector<CType>> target_objectives = std::nullopt,
            std::string gradient_mode = "forward",
            CType gradient_epsilon = DEFAULT_GRADIENT_EPSILON,
            CType archive_epsilon = 0.0,
            std::optional<bool> always_inherit_continuous = std::nullopt,
            std::optional<usize> batch_size = std::nullopt,
            // When set, evaluation runs on the GPU with the specified kernel. Requires GOBLIN_HAS_CUDA.
            std::optional<KernelVersion> kernel_version = std::nullopt)
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
        _always_inherit_continuous(always_inherit_continuous),
        _batch_size(batch_size) {
#ifdef GOBLIN_HAS_CUDA
    _kernel_version = kernel_version;
#else
    __goblin_runtime_assert(!kernel_version.has_value() && "GPU evaluation requires GOBLIN_HAS_CUDA");
    (void)kernel_version;
#endif
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
    // Near-zero variance means R² is undefined; clamp to 1.0 so NMSE falls back to MSE.
    for (isize i = 0; i < var_Y_train.size(); i++) {
      if (std::abs(var_Y_train(i)) < VARIANCE_BOUND) {
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
      for (isize i = 0; i < var_Y_test.size(); i++) {
        if (std::abs(var_Y_test(i)) < VARIANCE_BOUND) {
          var_Y_test(i) = 1.0;
        }
      }
    }

    if (target_objectives.has_value()) {
      register_target(target_objectives.value());
    }

#ifdef GOBLIN_HAS_CUDA
    if (_kernel_version.has_value()) {
      _init_gpu();
    }
#endif
  };

#ifdef GOBLIN_HAS_CUDA
  ~SRProblem() { free_gpu(); }

  /// Switch between CPU (nullopt) and GPU (KernelVersion) evaluation at runtime.
  /// Copies training data to the GPU the first time GPU mode is enabled.
  void set_kernel_version(std::optional<KernelVersion> kernel_version) {
    if (kernel_version.has_value() && d_X == nullptr) {
      _copy_data_to_gpu();
    }
    _kernel_version = kernel_version;
  }

  /// Free all GPU memory. Does not change the kernel version.
  void free_gpu() {
    _free_data_on_gpu();
    _free_solution_on_gpu();
    _free_results_on_gpu();
  }
#endif

  bool adapt(Rng& rng) override final {
    if (_batch_size.has_value() && _batch_size.value() < static_cast<usize>(X_train.rows())) {
      // TODO refactor into a DataLoader/Sampler abstraction with configurable sampling strategies.
      // (Random permutation is simplest; stratified/scattered subsets may better represent the
      // training distribution but earlier experiments on another codebase showed little benefit.)
      auto perm = permute(rng, X_train.rows());
      perm.resize(_batch_size.value());

      X_batch = X_train(perm, Eigen::placeholders::all);
      Y_batch = Y_train(perm, Eigen::placeholders::all);

      var_Y_batch = (Y_batch.rowwise() - Y_batch.colwise().mean()).square().colwise().mean();
      return true;
    }        
    
    return false;
  };

  CRef<Vec<DType>> discrete_domain_sizes() const override final { return ctx.domain_sizes; };

  CRef<Vec<CType>> continuous_lower_bounds() const override final { return _continuous_lower_bounds; };
  CRef<Vec<CType>> continuous_upper_bounds() const override final { return _continuous_upper_bounds; };

  CRef<Vec<CType>> continuous_init_lower_bounds() const override final { return _continuous_init_lower_bounds; };
  CRef<Vec<CType>> continuous_init_upper_bounds() const override final { return _continuous_init_upper_bounds; };

  void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
#ifdef GOBLIN_HAS_CUDA
    if (_kernel_version.has_value()) {
      _evaluate_gpu(solutions, indices);
      return;
    }
#endif
    _evaluate_cpu(rng, solutions, indices);
  };

  void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override final {
    _init->add_random(rng, *this, solutions, count);
#ifndef NDEBUG
    auto *p = dynamic_cast<SRQuality*>(&solutions[solutions.size() - 1].quality());
    assert(p != nullptr && "Quality mismatch");
#endif
  };

  const FitnessBase& fitness() const override final { return _fitness; };

  const ArchiveFitnessBase& archive_fitness() const override final { return _archive_fitness; };

  virtual std::tuple<bool, bool> inherit_discrete(SolutionBase& offspring,
                                                  const SolutionBase& donor,
                                                  const Subset& subset) const override {
    const bool inherit_continuous = _always_inherit_continuous.value_or(ctx.const_repr == ConstantRepr::ERCs ||
                                                                        ctx.const_repr == ConstantRepr::Edges) &&
                                    ctx.const_repr != ConstantRepr::None;

    // Pool constants are not index-paired with discrete variables, so the full pool is inherited.
    const bool inherit_by_index = ctx.const_repr != ConstantRepr::Pool;

    bool any_active_changed = false;
    bool anything_changed = false;

    for (usize i : subset.discrete) {
      if (offspring.discrete_values()(i) != donor.discrete_values()(i)) {
        any_active_changed |= offspring.discrete_active()(i);
        anything_changed = true;
        offspring.discrete_values()(i) = donor.discrete_values()(i);
      }

      // TODO for GCS: inherit child arities + permutations

      if (inherit_continuous && inherit_by_index) {
        // TODO floating-point equality here is not robust; consider a relative+absolute threshold.
        if (offspring.continuous_values()(i) != donor.continuous_values()(i)) {
          any_active_changed |= offspring.continuous_active()(i);
          anything_changed = true;
          offspring.continuous_values()(i) = donor.continuous_values()(i);
        }
      }
    }

    if (inherit_continuous && !inherit_by_index) {
      // Inheriting all continuous variables even when the paired discrete slot is not a constant
      // is imprecise, but earlier experiments showed interpolating continuous mixing doesn't help.
      for (usize i = 0; i < num_continuous(); i++) {
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
    }
    
    return false;
  };

  void log_header(std::ostream& os) const override final {
    for (usize i = 0; i < ctx.num_subexpressions; i++) {
      os << "subtree_" << i << ",";
    }
    os << "unresolved_expressions,";
    os << "expressions,";
    for (const auto& o : objectives) {
      os << o << "_train,";
    }
    if (Y_test.size() > 0) {
      for (const auto& o : objectives) {
        os << o << "_test,";
      }
    }

    archive_fitness().log_header(os);
  };

  void log(std::ostream& os, const SolutionBase& solution) const override final {
    for (const auto& expr : ctx.subtrees_to_sympy(solution)) {
      os << '"' << expr << "\",";
    }
    os << '"';
    log_solution_unresolved(os, solution);
    os << "\",";
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
          os << exprs[i];  // unevaluated solution — no scaling params yet
        } else {
          os << q.ls_params(0, i) << " + (" << q.ls_params(1, i) << " * (" << exprs[i] << "))";
        }
      } else {
        os << exprs[i];
      }
    }
  };

  void log_solution_unresolved(std::ostream& os, const SolutionBase& solution) const {
    auto exprs = ctx.to_sympy_unresolved(solution);
    for (usize i = 0; i < exprs.size(); i++) {
      if (i > 0) {
        os << " , ";
      }
      if (linear_scaling) {
        const auto& q = solution.quality_as<SRQuality>();
        if (static_cast<usize>(q.ls_params.cols()) != ctx.num_outputs) {
          os << exprs[i];
        } else {
          os << q.ls_params(0, i) << " + (" << q.ls_params(1, i) << " * (" << exprs[i] << "))";
        }
      } else {
        os << exprs[i];
      }
    }
  };

  void evaluate_test(const SolutionBase& solution) const {
    const auto& q = solution.quality_as<SRQuality>();
    if (Y_test.size() > 0 && !q.test_quality.has_value()) {
      Solution copy = solution;  // copy needed because active variables are not mutable

      Array<ScalarType> params;  // TODO fit FC params
      auto& cq = copy.quality_as<SRQuality>();
      eval_one(copy, X_test, Y_test, var_Y_test, params, false, cq, cq.ls_params);

      q.test_quality = MOQuality();
      q.test_quality.value().objectives = copy.quality_as<SRQuality>().objectives;
      q.test_quality.value().constraint_value = copy.quality_as<SRQuality>().constraint_value;
    }
  };

  const GPContext& context() const override final { return ctx; }

  std::tuple<std::vector<usize>, u64> gradient_steps(Rng& rng,
                                                     SolutionSetBase& solutions,
                                                     SolutionSetBase& parents,
                                                     const std::span<const usize>& indices,
                                                     usize num_steps) override final;

  GPContext ctx;
  bool linear_scaling;
  std::vector<std::string> objectives;
  Arr2D<ScalarType> X_train;
  Arr2D<ScalarType> Y_train;
  // Variance per output column, precomputed for NMSE normalisation.
  // Near-zero entries are clamped to 1.0 so NMSE degrades gracefully to MSE.
  Array<ScalarType> var_Y_train;
  Arr2D<ScalarType> X_batch;
  Arr2D<ScalarType> Y_batch;
  Array<ScalarType> var_Y_batch;
  Arr2D<ScalarType> X_test;
  Arr2D<ScalarType> Y_test;
  Array<ScalarType> var_Y_test;

 private:
  // --- CPU evaluation ---

  void _evaluate_cpu(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) {
    if (_batch_size.has_value() && X_batch.size() == 0) {
      adapt(rng);
    }

    Array<ScalarType> params;
    for (auto i : indices) {
      auto& q = solutions[i].quality_as<SRQuality>();
      q.test_quality = std::nullopt;  // invalidate cached test quality on every train evaluation
      if (_batch_size.has_value()) {
        eval_one(solutions[i], X_batch, Y_batch, var_Y_batch, params, true, q, q.ls_params);
      } else {
        eval_one(solutions[i], X_train, Y_train, var_Y_train, params, true, q, q.ls_params);
      }
    }
  }

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
            CType intercept = ls_params(0, o);
            CType slope = ls_params(1, o);
            quality.objectives(j) += ((intercept + slope * Y_pred.col(o)) - Y.col(o)).square().mean();
          }
        } else {
          quality.objectives(j) = (Y_pred - Y).square().mean();
        }
      } else if (objectives[j] == "nmse") {
        if (linear_scaling) {
          quality.objectives(j) = 0.0;
          for (usize o = 0; o < ctx.num_outputs; o++) {
            CType intercept = ls_params(0, o);
            CType slope = ls_params(1, o);
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

  // --- GPU evaluation ---
#ifdef GOBLIN_HAS_CUDA
  void _evaluate_gpu(SolutionSetBase& solutions, const std::span<const usize>& indices) {
    usize expression_size;
    size_t num_solutions = indices.size();

    if (num_solutions == 0) {
      return;
    }

    std::vector<u8> node_type;
    std::vector<float> node_value;
    std::vector<bool> overflowed(num_solutions, false);

    size_t k = 0;
    for (auto i : indices) {
      overflowed[k++] = ctx.to_gpu_representation(solutions[i], node_type, node_value, true, expression_size);
    }

    __goblin_runtime_assert(node_type.size() == node_value.size());

    const size_t solution_length = ctx.max_expression_size;
    const LaunchConfig config = LaunchConfig::determine(_kernel_version.value(), num_solutions, _num_datapoints, solution_length, _num_sms);

    config.check();

    _copy_solutions_to_gpu(node_type, node_value);
    _allocate_results_on_gpu(config);

    kernel_wrapper(d_X, d_Y, d_type, d_value, d_partial, d_result, config);

    std::vector<float> result(num_solutions);
    copy_from_device(result.data(), d_result, num_solutions);

    k = 0;
    for (auto i : indices) {
      auto& quality = solutions[i].quality_as<SRQuality>();
      if (overflowed[k]) {
        quality.objectives.array() = std::numeric_limits<CType>::infinity();
        quality.constraint_value = 1.0;
      } else {
        quality.constraint_value = 0.0;
        quality.objectives(0) = result[k];
      }
      k++;
    }
  }

  void _init_gpu() {
    _num_sms = get_gpu_info().num_sms;
    _num_datapoints = X_train.rows();

    _copy_data_to_gpu();
  }

  void _copy_data_to_gpu() {
    Arr2D<float> X32 = X_train.cast<float>();
    Arr2D<float> Y32 = Y_train.cast<float>();
    d_X = allocate_and_copy(X32.data(), X32.size());
    d_Y = allocate_and_copy(Y32.data(), Y32.size());
  }

  void _copy_solutions_to_gpu(std::vector<u8> node_type, std::vector<float> node_value) {
    const size_t num_elements = node_type.size();

    if (_num_solutions_allocated < num_elements) {
      _free_solution_on_gpu();
      d_type  = allocate_and_copy(node_type.data(), node_type.size());
      d_value = allocate_and_copy(node_value.data(), node_value.size());
      _num_solutions_allocated = num_elements;
    } else {
      copy_to_gpu(d_type,  node_type.data(),  node_type.size());
      copy_to_gpu(d_value, node_value.data(), node_value.size());
    }
  }

  void _allocate_results_on_gpu(const LaunchConfig& config) {
    if (config.kernel_version != KernelVersion::SingleBlock) {
      const size_t num_partials =
          (config.kernel_version == KernelVersion::DynamicBlock)
              ? config.num_solutions * config.eval.grid.y
              : config.num_solutions * config.num_datapoints;

      if (_num_partials_allocated < num_partials) {
        free_on_gpu(d_partial);
        d_partial = allocate_on_gpu<float>(num_partials);
        _num_partials_allocated = num_partials;
      }
    }

    if (_num_results_allocated < config.num_solutions) {
      free_on_gpu(d_result);
      d_result = allocate_on_gpu<float>(config.num_solutions);
      _num_results_allocated = config.num_solutions;
    }
  }

  void _free_data_on_gpu() {
    free_on_gpu(d_X);
    free_on_gpu(d_Y);
    d_X = nullptr;
    d_Y = nullptr;
  }

  void _free_solution_on_gpu() {
    free_on_gpu(d_type);
    free_on_gpu(d_value);
    d_type  = nullptr;
    d_value = nullptr;
    _num_solutions_allocated = 0;
  }

  void _free_results_on_gpu() {
    free_on_gpu(d_partial);
    free_on_gpu(d_result);
    d_partial = nullptr;
    d_result  = nullptr;
    _num_partials_allocated = 0;
    _num_results_allocated  = 0;
  }
#endif // GOBLIN_HAS_CUDA

  SRFitness _archive_fitness;
  SRFitness _fitness;
  std::shared_ptr<InitBase> _init;
  UnboundedArchive _target;
  usize _num_continuous{};
  Vec<CType> _continuous_lower_bounds;
  Vec<CType> _continuous_upper_bounds;
  Vec<CType> _continuous_init_lower_bounds;
  Vec<CType> _continuous_init_upper_bounds;

  // CPU fields
  std::string _gradient_mode;
  CType _gradient_epsilon;
  std::optional<bool> _always_inherit_continuous;
  std::optional<usize> _batch_size;
  mutable Arr2D<ScalarType> _eval_buffer;  // reused across evaluations to avoid repeated allocation

#ifdef GOBLIN_HAS_CUDA
  // GPU fields
  std::optional<KernelVersion> _kernel_version;
  int    _num_sms        = 0;
  size_t _num_datapoints = 0;
  // Watermark sizes: GPU buffers only grow, never shrink, to minimise cudaMalloc calls.
  size_t _num_solutions_allocated = 0;
  size_t _num_partials_allocated  = 0;
  size_t _num_results_allocated   = 0;
  float*    d_X       = nullptr;
  float*    d_Y       = nullptr;
  uint8_t*  d_type    = nullptr;
  float*    d_value   = nullptr;
  float*    d_partial = nullptr;
  float*    d_result  = nullptr;
#endif  // GOBLIN_HAS_CUDA
};

};  // namespace goblin

#endif /* _GOBLIN_GP_SR_H */
