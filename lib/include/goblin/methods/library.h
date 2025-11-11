#pragma once
#ifndef _GOBLIN_GOMEA_LIBRARY_H
#define _GOBLIN_GOMEA_LIBRARY_H

#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <tuple>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include <gomea/src/common/linkage_config.hpp>
#include <gomea/src/discrete/Config.hpp>
#include <gomea/src/discrete/gomeaIMS.hpp>
#include <gomea/src/real_valued/Config.hpp>
#include <gomea/src/real_valued/rv-gomea.hpp>

#include "goblin/bench/problem.h"
#include "goblin/lib/assert.h"
#include "goblin/lib/budget.h"
#include "goblin/lib/method.h"
#include "goblin/lib/rng.h"
#include "goblin/lib/types.h"

namespace goblin {
class DiscreteGOMEA final : public MethodBase {
 public:
  DiscreteGOMEA(std::string_view linkage_model = "LinkageTree",
                std::string similarity_metric = "MI",
                bool gene_invariant = false,  // enables GI-GOMEA
                bool forced_improvements = true,
                bool filter_linkage = false,
                std::optional<usize> max_subset_size = std::nullopt,  // only used for the linkage tree FOS
                usize base_population_size = 2,         // The size of the first population in the multi-start scheme.
                usize max_number_of_populations = 100,  // The maximum number of populations in the multi-start scheme.
                usize subgeneration_factor = 4,         // The subgeneration factor in the multi-start scheme.
                usize max_archive_size = 0) {
    config.generational_statistics = false;
    config.usePartialEvaluations = 0;
    config.AnalyzeFOS = 0;
    config.verbose = false;

    if (linkage_model == "Univariate") {
      config.FOSIndex = gomea::linkage::linkage_model_type::UNIVARIATE;
      linkage_config = gomea::linkage_config_t();
    } else if (linkage_model == "LinkageTree") {
      config.FOSIndex = gomea::linkage::linkage_model_type::LINKAGE_TREE;
      linkage_config = gomea::linkage_config_t(similarity_metric.c_str(), filter_linkage,
                                               max_subset_size.value_or(std::numeric_limits<int>().infinity()), false);
    } else {
      // TODO raise error?
    }
    config.linkage_config = &linkage_config;

    config.gene_invariant = gene_invariant;
    config.useForcedImprovements = forced_improvements ? 1 : 0;

    config.maxArchiveSize = max_archive_size;

    config.maximumNumberOfGOMEAs = max_number_of_populations;
    config.IMSsubgenerationFactor = subgeneration_factor;
    config.basePopulationSize = base_population_size;
  };

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) override final {
    auto archive = std::make_shared<UnboundedArchive>(problem.archive_fitness());
    // copy to make the base options persist over multiple calls
    auto conf = config;

    if (problem.num_discrete() < 1 || problem.num_continuous() > 0 || problem.num_objectives() > 1) {
      __goblin_runtime_assert(false);  // Problem not supported
    }

    std::random_device rd;
    std::uniform_int_distribution<u64> seed_dist(0, std::numeric_limits<u64>::max());
    Rng rng(seed.value_or(seed_dist(rd)), 0);

    if (seed.has_value()) {
      conf.fix_seed = true;
      conf.randomSeed = seed.value();
    }

    if (population_size.has_value()) {
      conf.maximumNumberOfGOMEAs = 1;
      conf.basePopulationSize = static_cast<int>(population_size.value());
    }

    if (budget.max_evaluations.has_value()) {
      conf.maximumNumberOfEvaluations = static_cast<int>(budget.max_evaluations.value());
    }
    if (budget.max_generations.has_value()) {
      conf.maximumNumberOfGenerations = static_cast<int>(budget.max_generations.value());
    }
    if (budget.max_time.has_value()) {
      std::chrono::duration<double> max_time = budget.max_time.value();
      conf.maximumNumberOfSeconds = max_time.count();
    }

    class Wrapper final : public gomea::fitness::fitness_t<char> {
     public:
      Wrapper(Rng& rng, InstanceBase& p, ArchiveBase& a)
          : gomea::fitness::fitness_t<char>(p.num_discrete(), p.discrete_domain_sizes().maxCoeff()),
            rng(rng),
            p(p),
            a(a),
            idxs({0}) {
        initialize();
        s.add(Solution(p.archive_fitness().worst(), Vec<DType>::Zero(p.num_discrete()), std::nullopt));
      };

      void evaluationFunction(gomea::solution_t<char>* solution) {
        for (usize i = 0; i < p.num_discrete(); i++) {
          solution->variables[i] %= static_cast<char>(p.discrete_domain_sizes()(i));
        }
        s[0].discrete_values() =
            Eigen::Map<Eigen::ArrayX<char>>(solution->variables.data(), solution->variables.size()).cast<DType>();
        p.evaluate(rng, s, idxs);
        solution->setObjectiveValue(s[0].quality().objectives(0));
        solution->setConstraintValue(s[0].quality().constraint_value);
        a.update(s[0], true);

        if (p.target_reached(a)) {
          throw gomea::utils::terminationException("");
        }

        this->full_number_of_evaluations++;
        this->number_of_evaluations++;
      };

      void partialEvaluationFunction(gomea::solution_t<char>* parent, gomea::partial_solution_t<char>* solution) {
        s[0].discrete_values() =
            Eigen::Map<Eigen::VectorX<char>>(parent->variables.data(), parent->variables.size()).cast<DType>();
        for (usize i = 0; i < solution->touched_indices.size(); i++) {
          solution->touched_variables[i] %= static_cast<char>(p.discrete_domain_sizes()(solution->touched_indices[i]));
          s[0].discrete_values()(solution->touched_indices[i]) = static_cast<DType>(solution->touched_variables[i]);
        }
        p.evaluate(rng, s, idxs);
        solution->setObjectiveValue(s[0].quality().objectives(0));
        solution->setConstraintValue(s[0].quality().constraint_value);
        a.update(s[0], true);

        if (p.target_reached(a)) {
          throw gomea::utils::terminationException("");
        }

        this->full_number_of_evaluations++;
        this->number_of_evaluations++;
      };

     private:
      Rng& rng;
      InstanceBase& p;
      ArchiveBase& a;
      std::vector<usize> idxs;
      DefaultSolutionSet s;
    };

    Wrapper fn(rng, problem, *archive);

    conf.fitness = &fn;

    instance = std::make_unique<gomea::discrete::gomeaIMS>(&conf);
    try {
      instance->run();
    } catch (gomea::utils::terminationException& ex) {
    }

    // TODO make guess as to why we stopped...
    return std::make_tuple(archive, TerminationStatus::Converged);
  };

  std::optional<u64> current_generation() const override final {
    u64 generations = 0;
    if (instance) {
      for (auto& p : instance->GOMEAs) {
        generations += static_cast<u64>(p->numberOfGenerations);
      }
    }
    return generations;
  };

 private:
  gomea::linkage_config_t linkage_config;
  gomea::discrete::Config config;
  std::unique_ptr<gomea::discrete::gomeaIMS> instance;
};

class RvGOMEA final : public MethodBase {
 public:
  RvGOMEA(std::string_view linkage_model = "Full",
          usize base_population_size = 10,       // The size of the first population in the multi-start scheme.
          usize max_number_of_populations = 25,  // The maximum number of populations in the multi-start scheme.
          usize subgeneration_factor = 8,        // The subgeneration factor in the multi-start scheme.
          usize max_nis = 100,                   // The maximum number of subsequent generations without an
                                                 // improvement while the distribution multiplier is <= 1.0.
          double selection_percentile = 0.35,    // The selection truncation percentile
                                                 // (in [1/population_size,1]).
          double distribution_multiplier_decrease = 0.9,  // The multiplicative distribution multiplier decrease.
          double standard_deviation_threshold = 1.0,  // The maximum ratio of the distance of the average improvement
                                                      // to the mean compared to the distance of one standard deviation
                                                      // before triggering AVS (SDR mechanism).
          double fitness_variance_tolerance = 0.0,    // The minimum fitness variance level that is allowed.
          bool selection_during_gom = true,           // Update the current distribution
                                                      // estimate for each GOM step/FOS subset
          bool update_elitist_during_gom = true       // Update the current elite for each GOM step/FOS subset
  ) {
    config.problem_index = 0;
    config.generational_statistics = false;

    if (linkage_model == "Full") {
      config.FOSIndex = -1;
    } else if (linkage_model == "Univariate") {
      config.FOSIndex = 1;
    } else if (linkage_model == "LinkageTree") {
      config.FOSIndex = -2;
    } else {
      // TODO raise error?
    }

    __goblin_runtime_assert(base_population_size >= 1);
    config.base_population_size = static_cast<int>(base_population_size);

    __goblin_runtime_assert(max_number_of_populations >= 1);
    config.maximum_number_of_populations = static_cast<int>(max_number_of_populations);

    config.number_of_subgenerations_per_population_factor = static_cast<int>(subgeneration_factor);

    config.maximum_no_improvement_stretch = max_nis;

    __goblin_runtime_assert(static_cast<int>(selection_percentile * config.base_population_size) > 0 &&
                            selection_percentile < 1.0);
    config.tau = selection_percentile;

    config.distribution_multiplier_decrease = distribution_multiplier_decrease;
    config.st_dev_ratio_threshold = standard_deviation_threshold;
    config.fitness_variance_tolerance = fitness_variance_tolerance;
  };

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) override final {
    if (problem.num_discrete() > 0 || problem.num_continuous() < 1 || problem.num_objectives() > 1) {
      __goblin_runtime_assert(false);  // Problem not supported
    }

    std::random_device rd;
    std::uniform_int_distribution<u64> seed_dist(0, std::numeric_limits<u64>::max());
    Rng rng(seed.value_or(seed_dist(rd)), 0);

    auto archive = std::make_shared<UnboundedArchive>(problem.archive_fitness());
    // copy to make the base options persist over multiple calls
    auto conf = config;

    if (seed.has_value()) {
      conf.fix_seed = true;
      conf.random_seed = seed.value();
    }

    if (population_size.has_value()) {
      conf.maximum_number_of_populations = 1;
      conf.base_population_size = static_cast<int>(population_size.value());
    }

    conf.lower_user_range = problem.continuous_init_lower_bounds().maxCoeff();
    conf.upper_user_range = problem.continuous_init_upper_bounds().minCoeff();

    if (budget.max_evaluations.has_value()) {
      conf.maximum_number_of_evaluations = static_cast<double>(budget.max_evaluations.value());
    }
    if (budget.max_generations.has_value()) {
      conf.maximum_number_of_generations = static_cast<int>(budget.max_generations.value());
    }
    if (budget.max_time.has_value()) {
      std::chrono::duration<double> max_time = budget.max_time.value();
      conf.maximum_number_of_seconds = max_time.count();
    }

    conf.number_of_variables = static_cast<int>(problem.num_continuous());

    class Wrapper final : public gomea::fitness::fitness_t<double> {
     public:
      Wrapper(Rng& rng, InstanceBase& p, ArchiveBase& a)
          : gomea::fitness::fitness_t<double>(p.num_continuous()), rng(rng), p(p), a(a), idxs({0}) {
        initialize();
        s.add(Solution(p.archive_fitness().worst(), std::nullopt, Vec<CType>::Zero(p.num_continuous())));
      };

      void evaluationFunction(gomea::solution_t<double>* solution) {
        s[0].continuous_values() = Eigen::Map<Eigen::VectorXd>(solution->variables.data(), solution->variables.size());
        p.evaluate(rng, s, idxs);
        solution->setObjectiveValue(s[0].quality().objectives(0));
        solution->setConstraintValue(s[0].quality().constraint_value);
        a.update(s[0], true);

        if (p.target_reached(a)) {
          throw gomea::utils::terminationException("");
        }

        this->full_number_of_evaluations++;
        this->number_of_evaluations++;
      };

      void partialEvaluationFunction(gomea::solution_t<double>* parent, gomea::partial_solution_t<double>* solution) {
        s[0].continuous_values() = Eigen::Map<Eigen::VectorXd>(parent->variables.data(), parent->variables.size());
        for (usize i = 0; i < solution->touched_indices.size(); i++) {
          s[0].continuous_values()(solution->touched_indices[i]) = solution->touched_variables[i];
        }
        p.evaluate(rng, s, idxs);
        solution->setObjectiveValue(s[0].quality().objectives(0));
        solution->setConstraintValue(s[0].quality().constraint_value);
        a.update(s[0], true);

        if (p.target_reached(a)) {
          throw gomea::utils::terminationException("");
        }

        this->full_number_of_evaluations++;
        this->number_of_evaluations++;
      };

      double getLowerRangeBound(int dimension) { return p.continuous_lower_bounds()(dimension); };

      double getUpperRangeBound(int dimension) { return p.continuous_upper_bounds()(dimension); };

     private:
      Rng& rng;
      InstanceBase& p;
      ArchiveBase& a;
      std::vector<usize> idxs;
      DefaultSolutionSet s;
    };

    Wrapper fn(rng, problem, *archive);

    conf.fitness = &fn;
    conf.initializeFOSFromIndex(conf.FOSIndex);

    instance = std::make_unique<gomea::realvalued::rvg_t>(&conf);
    try {
      instance->run();
    } catch (gomea::utils::terminationException& ex) {
    }

    delete conf.linkage_config;

    // TODO make guess as to why we stopped...
    return std::make_tuple(archive, TerminationStatus::Converged);
  }

  std::optional<u64> current_generation() const override final {
    u64 generations = 0;
    if (instance) {
      for (auto& p : instance->populations) {
        generations += static_cast<u64>(p->number_of_generations);
      }
    }
    return generations;
  };

 private:
  gomea::realvalued::Config config;
  std::unique_ptr<gomea::realvalued::rvg_t> instance;
};
};  // namespace goblin

#endif /* _GOBLIN_GOMEA_LIBRARY_H */
