#pragma once
#ifndef _GOBLIN_AMALGAM_H
#define _GOBLIN_AMALGAM_H

#include <cassert>
#include <functional>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "goblin/lib/assert.h"
#include "goblin/lib/types.h"
#include "goblin/lib/rng.h"
#include "goblin/lib/budget.h"
#include "goblin/bench/problem.h"
#include "goblin/lib/method.h"

namespace goblin {

class AMaLGaM final : public MethodBase {
 public:
  AMaLGaM(double init_lower_bound = -1.0,
          double init_upper_bound = 1.0,
          double fitness_variance_threshold = 0.0,
          usize max_num_populations = 25,
          usize max_evaluations = 1000000,
          std::optional<double> target_value = std::nullopt,
          std::optional<usize> base_population_size = std::nullopt,
          std::optional<usize> seed = std::nullopt)
      : init_lower_bound(init_lower_bound),
        init_upper_bound(init_upper_bound),
        fitness_variance_threshold(fitness_variance_threshold),
        max_num_populations(max_num_populations),
        max_evaluations(max_evaluations),
        target_value(target_value),
        base_population_size(base_population_size),
        seed(seed) {
    __goblin_runtime_assert(!isna(init_lower_bound));
    __goblin_runtime_assert(!isna(init_upper_bound));
    __goblin_runtime_assert(init_lower_bound < init_upper_bound);
  }

  void set_init_bounds(double lower, double upper) {
    __goblin_runtime_assert(!isna(lower));
    __goblin_runtime_assert(!isna(upper));
    __goblin_runtime_assert(lower < upper);
    init_lower_bound = lower;
    init_upper_bound = upper;
  };

  std::tuple<Vec<double>, double, double> run(
      std::function<std::tuple<double, double>(const Ref<Vec<double>>&)>& function,
      std::vector<std::tuple<double, double>> bounds);

  std::optional<u64> current_generation() const override final;

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) override final {
    if (problem.num_discrete() > 0 || problem.num_continuous() < 1 || problem.num_objectives() > 1) {
      __goblin_runtime_assert(false);  // Problem not supported
    }

    auto archive = std::make_shared<UnboundedArchive>(problem.archive_fitness());
    Rng rng = seeded_rng(seed);

    // make a copy to persist other options between calls on the same instance
    AMaLGaM alg = *this;

    alg.seed = seed;

    if (population_size.has_value()) {
      alg.max_num_populations = 1;
      alg.base_population_size = population_size.value();
    }

    alg.init_lower_bound = problem.continuous_init_lower_bounds().maxCoeff();
    alg.init_upper_bound = problem.continuous_init_upper_bounds().minCoeff();

    if (budget.max_evaluations.has_value()) {
      alg.max_evaluations = budget.max_evaluations.value();
    }

    std::vector<std::tuple<double, double>> bounds;
    for (usize i = 0; i < problem.num_continuous(); i++) {
      bounds.emplace_back(problem.continuous_lower_bounds()(i), problem.continuous_upper_bounds()(i));
    }

    AoSSet s;
    s.add(Solution(problem.archive_fitness().worst(), std::nullopt, Vec<double>::Zero(problem.num_continuous())));
    std::vector<usize> idxs{0};
    std::function<std::tuple<double, double>(const Eigen::Ref<Eigen::VectorXd>& parameters)> fn =
        [&](const Eigen::Ref<Eigen::VectorXd>& parameters) {
          s[0].continuous_values() = parameters;
          problem.evaluate(rng, s, idxs);

          archive->update(s[0], true);
          if (problem.target_reached(*archive)) {
            throw std::runtime_error("");
          }

          return std::make_tuple(s[0].quality().objectives(0), s[0].quality().constraint_value);
        };
    try {
      alg.run(fn, bounds);
    } catch (std::runtime_error& e) {
    }

    // TODO better terminationstatus guess
    // - target reached if target was reached
    // - evaluations
    return std::make_tuple(archive, TerminationStatus::Converged);
  };

 private:
  double init_lower_bound = -1.0;
  double init_upper_bound = 1.0;
  double fitness_variance_threshold = 0.0;
  usize max_num_populations = 25;
  usize max_evaluations = 1000000;
  std::optional<double> target_value = std::nullopt;
  std::optional<usize> base_population_size = std::nullopt;
  std::optional<usize> seed = std::nullopt;
};

};  // namespace goblin

#endif /* _GOBLIN_AMALGAM_H */
