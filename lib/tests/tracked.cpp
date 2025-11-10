#include <print>

#include "doctest/doctest.h"
#include <Eigen/Dense>

#include "goblin/bench/functions/continuous.h"
#include "goblin/bench/problem.h"
#include "goblin/bench/tracked.h"
#include "goblin/methods/amalgam.h"

using namespace goblin;

TEST_CASE("goblin::bench::tracked") {
  BenchmarkInstance sphere(std::vector<std::shared_ptr<ObjectiveBase>>{std::make_shared<Sphere>(2)});
  sphere.register_target({1e-8});

  REQUIRE(sphere.num_objectives() == 1);
  REQUIRE(sphere.num_continuous() == 2);

  Budget budget(/* max_evaluations = */ 10000);

  auto alg = AMaLGaM();
  alg.set_init_bounds(100, 110);

  // log will be at build/lib/tests/goblin_tracked_log.csv
  auto [front, status] = Tracked::run(sphere, alg, budget, TrackingOptions("goblin_tracked_log.csv"));

  std::println("Status: {}", format_as(status));

  REQUIRE(front.empty() == false);
  CHECK(front.so_solution(0).quality().objectives[0] <= 1e-8);
}
