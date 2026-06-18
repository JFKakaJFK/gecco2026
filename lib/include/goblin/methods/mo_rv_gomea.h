#pragma once
#ifndef _GOBLIN_MO_RV_GOMEA_H
#define _GOBLIN_MO_RV_GOMEA_H

#include <cassert>
#include <functional>
#include <optional>
#include <random>
#include <stdexcept>
#include <tuple>
#include <variant>
#include <vector>

#include <Eigen/Dense>

#include "goblin/bench/problem.h"
#include "goblin/lib/budget.h"
#include "goblin/lib/linkage_model.h"
#include "goblin/lib/method.h"
#include "goblin/lib/rng.h"
#include "goblin/lib/types.h"

namespace goblin {

class MoRvGOMEA final : public MethodBase {
   public:
    MoRvGOMEA(
        /// Initial population size
        std::optional<usize> initial_population_size = std::nullopt,
        /// Maximum number of populations
        usize max_num_populations = 25,
        /// Number of generations performed before starting larger populations
        usize subgeneration_factor = 8,
        /// The target number of solutions in the approximation front
        usize target_archive_size = 100,
        /// Selection percentile of solutions to use to estimate sampling distributions
        double selection_percentile = 0.35,
        /// The distribution shrinking factor used to adapt the sampling distribution scale (increasing factor is always
        /// 1/shrinking factor)
        double distribution_multiplier_decrease = 0.9,
        /// The standard deviation threshold used to trigger variance scaling
        double std_deviation_ratio_threshold = 1.0,
        /// The maximum number of subsequent generations without and improvement with distribution multiplier <= 1.0
        std::optional<usize> max_no_improvement_stretch = std::nullopt,
        /// The initial number of mixing components
        std::optional<usize> initial_num_clusters = std::nullopt,
        /// The linkage model to use, either a static FOS or one of `"LinkageTree" (default), "Full", "Univariate",
        /// "StaticLinkageTree", "BFLT" (Bounded Fixed Linkage Tree), "RandomBFLT", "RandomLinkageTree",
        /// "StaticRandomLinkageTree"`
        std::variant<std::string, StaticFOS> linkage_model = "LinkageTree",
        /// The maximum number of variables in a subset
        std::optional<usize> max_subset_size = std::nullopt,
        /// Determines if boundary repair is used for box-constrained problems
        bool boundary_repair = false,
        /// Enables the forced improvements mechanism
        bool forced_improvements = false,
        /// If disabled all evaluations are full evaluations
        bool partial_evaluations = true)
        : initial_population_size(initial_population_size),
          max_num_populations(max_num_populations),
          subgeneration_factor(subgeneration_factor),
          target_archive_size(target_archive_size),
          selection_percentile(selection_percentile),
          distribution_multiplier_decrease(distribution_multiplier_decrease),
          std_deviation_ratio_threshold(std_deviation_ratio_threshold),
          max_no_improvement_stretch(max_no_improvement_stretch),
          initial_num_clusters(initial_num_clusters),
          linkage_model(linkage_model),
          max_subset_size(max_subset_size),
          boundary_repair(boundary_repair),
          forced_improvements(forced_improvements),
          partial_evaluations(partial_evaluations) {
        if (initial_population_size.value_or(1) < 1) {
            throw std::runtime_error("Initial population size must be >= 1.");
        }

        if (initial_num_clusters.value_or(1) < 1) {
            throw std::runtime_error("Number of clusters must be >= 1.");
        }

        if (target_archive_size < 1) {
            throw std::runtime_error("Target archive size must be >= 1.");
        }

        if (selection_percentile <= 0.0 || selection_percentile >= 1.0) {
            throw std::runtime_error("Selection percentile must be in (0, 1].");
        }
    };

    std::optional<u64> current_generation() const override final;

    std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
        InstanceBase& problem,
        const Budget& budget,
        std::optional<u64> seed = std::nullopt,
        std::optional<usize> population_size = std::nullopt) override final;

   private:
    std::optional<usize> initial_population_size{};
    usize max_num_populations{};
    usize subgeneration_factor{};
    usize target_archive_size{};
    double selection_percentile{};
    double distribution_multiplier_decrease{};
    double std_deviation_ratio_threshold{};
    std::optional<usize> max_no_improvement_stretch{};
    std::optional<usize> initial_num_clusters{};
    std::variant<std::string, StaticFOS> linkage_model{};
    std::optional<usize> max_subset_size{};
    bool boundary_repair{};
    bool forced_improvements{};
    bool partial_evaluations{};
};

};  // namespace goblin

#endif /* _GOBLIN_MO_RV_GOMEA_H */
