#include <iostream>
#include "doctest/doctest.h"
#include <Eigen/Dense>

#include "goblin/bench/functions/continuous.h"
#include "goblin/bench/problem.h"
#include "goblin/methods/mixed.h"
#include "goblin/methods/library.h"

using namespace goblin;

TEST_CASE("goblin::methods::mixed_rv") {
  BenchmarkInstance sphere(std::vector<std::shared_ptr<ObjectiveBase>>{std::make_shared<Sphere>(2)});
  sphere.register_target({1e-8});
  sphere.set_initial_bounds(100.0, 110.0);

  REQUIRE(sphere.num_objectives() == 1);
  // REQUIRE(sphere.num_continuous() == 2);

  Budget budget(/* max_evaluations = */ 10000);

  auto rvg = RvGOMEA();

  auto [front, _] = rvg.run(sphere, budget);

  REQUIRE(front->empty() == false);
  CHECK(front->so_solution(0).quality().objectives[0] <= 1e-8);

  auto mixed_full = MixedGOMEA(PopulationOptions(), RvOptions{.max_nis = 100},
                               IMSOptions{
                                   .initial_population_size = 10, .max_num_populations = 1
                                   // .initial_population_size = 10,
                                   // .subgeneration_factor = 8,
                               },
                               std::make_shared<LinkageTreeFOS>(), std::make_shared<FullFOS>());
  front = std::get<0>(mixed_full.run(sphere, budget));

  REQUIRE(front->empty() == false);
  CHECK(front->so_solution(0).quality().objectives[0] <= 1e-8);

  for (usize i = 0; i <= 1; i++) {
    auto mixed_lt = MixedGOMEA(PopulationOptions(), RvOptions{.intron_aware = i > 0, .max_nis = 100},
                               IMSOptions{
                                   .initial_population_size = 10, .max_num_populations = 1
                                   // .initial_population_size = 10,
                                   // .subgeneration_factor = 8,
                               },
                               std::make_shared<LinkageTreeFOS>(), std::make_shared<LinkageTreeFOS>());
    front = std::get<0>(mixed_lt.run(sphere, budget));

    REQUIRE(front->empty() == false);
    CHECK(front->so_solution(0).quality().objectives[0] <= 1e-8);
  }
}
