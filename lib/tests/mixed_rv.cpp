#include <iostream>
#include "doctest/doctest.h"

#define EIGEN_MAX_ALIGN_BYTES 0
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Dense>

#include "goblin/bench/functions/continuous.h"
#include "goblin/bench/problem.h"
#include "goblin/methods/mixed.h"
#include "goblin/methods/library.h"

using namespace goblin;

TEST_CASE("goblin::methods::mixed_rv") {
  BenchmarkInstance sphere(std::vector<std::shared_ptr<ObjectiveBase>>{std::make_shared<Sphere>(2)});
  sphere.add_target(1e-8);
  sphere.set_initial_bounds(100.0, 110.0);

  REQUIRE(sphere.num_objectives() == 1);
  // REQUIRE(sphere.num_continuous() == 2);

  Budget budget(/* max_evaluations = */ 10000);

  auto mixed_full = MixedGOMEA(PopulationOptions(), RvOptions{.max_nis = 100},
                               IMSOptions{
                                   .initial_population_size = 10,
                                   .max_num_populations = 1,
                                   .subgeneration_factor = 8,
                               },
                               std::make_shared<LinkageTreeFOS>(), std::make_shared<FullFOS>());

  // auto [front, _] = mixed_full.run(sphere, budget);

  auto rvg = RvGOMEA(
      /* linkage_model = */ "Full",
      /* base_population_size = */ 10,
      /* max_number_of_populations = */ 1);

  auto [front, _] = rvg.run(sphere, budget);

  REQUIRE(front->empty() == false);
  REQUIRE(front->so_solution(0).quality_as<MOQuality>().objectives[0] <= 1e-8);

  front = std::get<0>(mixed_full.run(sphere, budget));

  REQUIRE(front->empty() == false);
  REQUIRE(front->so_solution(0).quality_as<MOQuality>().objectives[0] <= 1e-8);

  for (usize i = 0; i <= 1; i++) {
    auto mixed_lt = MixedGOMEA(PopulationOptions(), RvOptions{.intron_aware = i > 0, .max_nis = 100},
                               IMSOptions{
                                   .initial_population_size = 10,
                                   .max_num_populations = 1,
                                   .subgeneration_factor = 8,
                               },
                               std::make_shared<LinkageTreeFOS>(), std::make_shared<LinkageTreeFOS>());
    front = std::get<0>(mixed_lt.run(sphere, budget));

    REQUIRE(front->empty() == false);
    REQUIRE(front->so_solution(0).quality_as<MOQuality>().objectives[0] <= 1e-8);
  }
}
