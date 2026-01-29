#include <iostream>
#include "doctest/doctest.h"
#include <Eigen/Dense>

#include "goblin/bench/functions/mixed.h"
#include "goblin/bench/functions/continuous.h"
#include "goblin/bench/problem.h"
#include "goblin/methods/mixed.h"
#include "goblin/methods/library.h"

using namespace goblin;

TEST_CASE("goblin::methods::mixed_mixed") {
  BenchmarkInstance sphere(std::vector<std::shared_ptr<ObjectiveBase>>{std::make_shared<LeadingSpheres>(5)});
  sphere.register_target({1e-8});
  sphere.set_initial_bounds(-1.0, 1.0);

  REQUIRE(sphere.num_objectives() == 1);
  // REQUIRE(sphere.num_continuous() == 2);

  Budget budget(/* max_evaluations = */ 1000000);

  auto mixed_full = MixedGOMEA(PopulationOptions(),
                               RvOptions{
                                   .intron_aware = true,
                                   // .max_nis = 20
                               },
                               IMSOptions{.initial_population_size = 10,
                                          .subgeneration_factor = 8,
                                          // .max_num_populations = 1,
                                          .restart_stale_populations = true},
                               std::make_shared<UnivariateFOS>(), std::make_shared<UnivariateFOS>());
  auto front = std::get<0>(mixed_full.run(sphere, budget));

  REQUIRE(front->empty() == false);
  std::println("{} @ {}", sphere.format_solution(front->so_solution(0)),
               sphere.fitness().format(front->so_solution(0).quality()));
  CHECK(static_cast<const MOQuality&>(front->so_solution(0).quality()).objectives[0] <= 1e-8);
}
