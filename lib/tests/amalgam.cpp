#include <iostream>
#include "doctest/doctest.h"
#include <Eigen/Dense>

#include "goblin/bench/functions/continuous.h"
#include "goblin/bench/problem.h"
#include "goblin/methods/amalgam.h"

using namespace goblin;

TEST_CASE("goblin::methods::amalgam") {
  BenchmarkInstance sphere(std::vector<std::shared_ptr<ObjectiveBase>>{std::make_shared<Sphere>(2)});
  sphere.set_initial_bounds(100.0, 110.0);
  sphere.register_target({1e-8});

  REQUIRE(sphere.num_objectives() == 1);
  REQUIRE(sphere.num_continuous() == 2);

  Budget budget(/* max_evaluations = */ 10000);

  auto alg = AMaLGaM();
  // alg.set_init_bounds(100, 110);
  auto [front, _] = alg.run(sphere, budget, /* seed = */ std::nullopt, /* population_size = */ 10);

  REQUIRE(front->empty() == false);
  CHECK(front->so_solution(0).quality_as<MOQuality>().objectives[0] <= 1e-8);
}
