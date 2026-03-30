#include <iostream>
#include <cassert>
#include <print>

#include "doctest/doctest.h"
#include <Eigen/Dense>

#include "goblin/bench/functions/combinators.h"
#include "goblin/bench/functions/discrete.h"
#include "goblin/bench/problem.h"
#include "goblin/bench/tracked.h"
#include "goblin/methods/library.h"
#include "goblin/methods/mixed.h"

using namespace std::chrono_literals;
using namespace goblin;

TEST_CASE("goblin::methods::mixed_ia") {
  const usize N = 25;  // 150;
  const double VTR = N;
  BenchmarkInstance instance(std::make_shared<LeadingOnes>(N));
  instance.set_init(std::make_shared<RandomInit>());
  instance.register_target({VTR});

  Budget budget(/* max_evaluations = */ 1000000);

  PopulationOptions options;
  options.forced_improvements = true;
  options.donor_search_proportion = 0.0;

  auto gomea = MixedGOMEA(
      options, RvOptions{.enabled = false},
      IMSOptions{.initial_population_size = 2, .max_num_populations = 25, .restart_stale_populations = false},
      std::make_shared<LinkageTreeFOS>(
          /* metric = */ "mi",
          /* intron_strategy = */ "any_active",
          /* merge_continuous = */ true,
          /* num_continuous_bins = */ std::nullopt,
          /* filter_parent_threshold = */ std::nullopt,
          /* filter_children_threshold = */ std::nullopt,
          /* filter_root = */ true),
      std::make_shared<FullFOS>(), std::make_shared<AMaLGaMSamplingModel>());

  u64 seed = 11765961851908202236ull;
  // auto [front, status] = Tracked::run(instance, gomea, budget, TrackingOptions("mixed_ia.csv"), seed);
  auto [front, status] = gomea.run(instance, budget, seed);

  std::println("Status: {}", format_as(status));

  REQUIRE(front->empty() == false);
  REQUIRE(front->so_solution(0).quality_as<MOQuality>().objectives[0] == VTR);
}
