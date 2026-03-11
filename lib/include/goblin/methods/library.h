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
#include <stdexcept>
#include <vector>

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

// Doesn't work yet since we store the full class, not a pointer...
// // forward declaration to avoid pulling in the library headers in the header
// namespace gomea {
// struct linkage_config_t;

// namespace discrete {
// struct Config;
// struct gomeaIMS;
// };  // namespace discrete

// namespace realvalued {
// struct Config;
// struct rvg_t;
// };  // namespace realvalued
// };  // namespace gomea

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
                usize max_archive_size = 0,
                std::string fos_order = "default"  // parallel, fixed
  );

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) override final;

  std::optional<u64> current_generation() const override final;

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
  );

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) override final;

  std::optional<u64> current_generation() const override final;

 private:
  gomea::realvalued::Config config;
  std::unique_ptr<gomea::realvalued::rvg_t> instance;
};
};  // namespace goblin

#endif /* _GOBLIN_GOMEA_LIBRARY_H */
