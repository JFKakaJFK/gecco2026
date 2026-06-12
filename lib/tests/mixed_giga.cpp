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

TEST_CASE("goblin::methods::mixed_giga") {
  // BenchmarkInstance instance(
  //     std::vector<std::shared_ptr<ObjectiveBase>>{std::make_shared<OneMax>(5)});
  const usize REPEATS = 4;
  BenchmarkInstance instance(std::make_shared<Repeat>(std::make_shared<DeceptiveTrap>(5), REPEATS));
  double vtr = 5.0 * REPEATS;
  instance.add_target(vtr);

  Budget budget(/* max_evaluations = */ 100000, /* max_generations = */ 100);

  // SUBCASE("Library") {
  //   auto gomea = DiscreteGOMEA("LinkageTree", "MI", /* gene_invariant = */ true);
  //   auto [front, status] = Tracked::run(instance, gomea, budget, TrackingOptions("discrete_giga.csv"));

  //   REQUIRE(front->empty() == false);
  //   REQUIRE(front->so_solution(0).quality_as<MOQuality>().objectives[0] == vtr);
  // };

  SUBCASE("Mixed") {
    auto gomea = MixedGOMEA(PopulationOptions{.gene_invariant = true}, RvOptions(), IMSOptions(),
                            std::make_shared<LinkageTreeFOS>(
                                /* metric = */ "mi"),
                            std::make_shared<FullFOS>(), std::make_shared<AMaLGaMSamplingModel>());
    auto [front, status] = Tracked::run(instance, gomea, budget, TrackingOptions("mixed_giga.csv"));

    REQUIRE(front->empty() == false);
    REQUIRE(front->so_solution(0).quality_as<MOQuality>().objectives[0] == vtr);
  };
}
